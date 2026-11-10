"""回撤降杠杆（DD 协议）单元测试（强化计划第五步）。

守护重点（最易错、最影响 A/B 证据可信度的桩子）：
  1. 滞回状态机：回撤 > dd_entry 触发（严格大于）、[dd_exit, dd_entry] 带内保持、
     < dd_exit 恢复——防临界区来回抖动
  2. 高水位跟踪：新高抬 HWM，再触发从新高量起
  3. 乘子叠加：DD 与 vol-target 独立相乘（两组合级缩放器可交换）
  4. 默认 no-op：enabled=false → 恒 1.0（深回撤无效）——引擎零回归
  5. 引擎接线：_run_date_loop 带 drawdown 配置时，净值跌破阈值后全权重 × dd_scale
  6. 尾部指标纯函数：longest_drawdown_days 峰→谷→复峰语义 / 尾部未恢复计到末尾

不依赖网络 / 不碰缓存：本地构造数据，纯函数可复现。

运行：python -m unittest tests.futures.test_dd_protocol
"""
from __future__ import annotations

import unittest

import pandas as pd

# 私有库路径（研究脚本已移入 junior-quant-alpha）
import sys as _sys
from pathlib import Path as _Path
PRIVATE_PROBES = _Path(__file__).resolve().parents[2] / 'private'
for _d in ('probes', 'experiments'):
    _p = PRIVATE_PROBES / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
        break

from jq_framework.backtest.futures_walk_forward import _run_date_loop
from jq_framework.risk.futures_risk import FuturesRiskManager
from futures_dd_evidence import (
    _variant_cfg,
    longest_drawdown_days,
    tail_metrics,
)
from jq_framework.strategies.base import Signal

_DD_CFG = {"drawdown": {"enabled": True, "dd_entry": 0.10,
                        "dd_exit": 0.05, "dd_scale": 0.5}}
_MR = {"RB": 0.11}


def _probe_strat():
    """恒多探针策略（无 carry 依赖、不看行情）——引擎接线测试用。"""
    class _AlwaysLong:
        def on_bar(self, bars):
            return Signal(direction="long", target_pos=1.0)
    return _AlwaysLong()


class DDStateMachineTest(unittest.TestCase):
    """滞回状态机语义（直测 FuturesRiskManager.apply）。"""

    def _w(self, rm, nav):
        out, _ = rm.apply({"RB": 1.0}, _MR, nav_t=nav)
        return out["RB"]

    def test_entry_strictly_above_threshold(self):
        rm = FuturesRiskManager(dict(_DD_CFG))
        self.assertEqual(self._w(rm, 1.0), 1.0)
        self.assertEqual(self._w(rm, 0.95), 1.0)      # 回撤 5%（=exit 不触发）
        self.assertEqual(self._w(rm, 0.90), 1.0)      # 回撤恰 10%：严格 > 不触发
        self.assertAlmostEqual(self._w(rm, 0.899), 0.5)  # 10.1% > entry → ×0.5

    def test_hysteresis_band_keeps_active(self):
        rm = FuturesRiskManager(dict(_DD_CFG))
        self._w(rm, 1.0)                               # 建立高水位
        self._w(rm, 0.85)                              # 触发（回撤 15%）
        self.assertAlmostEqual(self._w(rm, 0.93), 0.5)  # 回撤 7%（带内）保持
        self.assertAlmostEqual(self._w(rm, 0.95), 0.5)  # 回撤恰 5%（=exit 不恢复）

    def test_recovery_below_exit(self):
        rm = FuturesRiskManager(dict(_DD_CFG))
        self._w(rm, 1.0)
        self._w(rm, 0.85)
        self.assertEqual(self._w(rm, 0.96), 1.0)       # 回撤 4% < exit → 恢复

    def test_hwm_tracks_new_highs(self):
        rm = FuturesRiskManager(dict(_DD_CFG))
        self._w(rm, 1.0)
        self._w(rm, 1.2)                               # 新高 → HWM=1.2
        self.assertEqual(rm._hwm, 1.2)
        self.assertEqual(self._w(rm, 1.12), 1.0)       # 自 1.2 回撤 6.7% 未触发
        self.assertAlmostEqual(self._w(rm, 1.05), 0.5)  # 回撤 12.5% > entry → 触发
        self.assertAlmostEqual(self._w(rm, 1.06), 0.5)  # 带内保持

    def test_invalid_nav_keeps_state(self):
        rm = FuturesRiskManager(dict(_DD_CFG))
        self._w(rm, 1.0)
        self._w(rm, 0.80)                              # 触发
        self.assertAlmostEqual(self._w(rm, None), 0.5)  # 无效 nav 保持当前档
        self.assertAlmostEqual(rm._drawdown_scale(0.0), 0.5)
        rm2 = FuturesRiskManager(dict(_DD_CFG))
        self.assertEqual(self._w(rm2, None), 1.0)      # 未触发时无效 nav → 1.0

    def test_disabled_is_noop(self):
        # 默认 enabled=false：深回撤也无缩放（引擎零回归的根）
        rm = FuturesRiskManager({"vol_target": {"enabled": True, "w_min": 0.2}})
        out, _ = rm.apply({"RB": 1.0}, _MR, nav_t=0.5)
        self.assertGreaterEqual(out["RB"], 0.2 - 1e-9)  # 只剩 vol-target 的下限
        self.assertEqual(rm._drawdown_scale(0.5), 1.0)
        self.assertEqual(rm._hwm, None)                  # 状态不更新

    def test_composes_with_vol_target(self):
        # vol-target 满档（sigma_target 极大 → 恒 w_max=1.0）+ DD 触发 → 乘子 0.5
        rm = FuturesRiskManager({"vol_target": {"enabled": True, "sigma_target": 10.0,
                                                "w_max": 1.0, "w_min": 0.2},
                                 **dict(_DD_CFG)})
        self.assertAlmostEqual(self._w(rm, 1.0), 1.0)
        self.assertAlmostEqual(self._w(rm, 0.80), 0.5)   # 1.0 × 0.5

    def test_diagnostics_exposes_dd(self):
        rm = FuturesRiskManager(dict(_DD_CFG))
        self.assertTrue(rm.diagnostics()["dd_enabled"])
        self.assertFalse(FuturesRiskManager({}).diagnostics()["dd_enabled"])


class DDLinearModeTest(unittest.TestCase):
    """linear 形态（比例降险）：scale = max(floor, 1 − dd/tol)，连续无滞回。"""

    _LIN = {"enabled": True, "mode": "linear", "dd_tol": 0.20, "dd_floor": 0.25}

    def _w(self, rm, nav):
        out, _ = rm.apply({"RB": 1.0}, _MR, nav_t=nav)
        return out["RB"]

    def test_linear_scale_ladder(self):
        # tol=20%/floor=0.25：dd 5%/10%/15% → ×0.75/×0.5/×0.25；≥20% 地板
        rm = FuturesRiskManager({"drawdown": dict(self._LIN)})
        self._w(rm, 1.0)
        self.assertAlmostEqual(self._w(rm, 0.95), 0.75)   # dd 5%
        self.assertAlmostEqual(self._w(rm, 0.90), 0.50)   # dd 10%
        self.assertAlmostEqual(self._w(rm, 0.85), 0.25)   # dd 15%
        self.assertAlmostEqual(self._w(rm, 0.75), 0.25)   # dd 25% → 地板
        self.assertEqual(self._w(rm, 1.0), 1.0)           # 新高复满仓

    def test_linear_is_stateless_no_band(self):
        # 连续缩放无滞回带：dd 7% → 0.65；回补至 dd 3% → 0.85（渐进复险）
        rm = FuturesRiskManager({"drawdown": dict(self._LIN)})
        self._w(rm, 1.0)
        self.assertAlmostEqual(self._w(rm, 0.93), 0.65)
        self.assertAlmostEqual(self._w(rm, 0.97), 0.85)

    def test_linear_invalid_nav_keeps_last(self):
        rm = FuturesRiskManager({"drawdown": dict(self._LIN)})
        self._w(rm, 1.0)
        self.assertAlmostEqual(self._w(rm, 0.80), 0.25)   # dd 20% → 地板
        self.assertAlmostEqual(self._w(rm, None), 0.25)   # 无效 nav 保持最近档

    def test_linear_floor_zero_kills(self):
        rm = FuturesRiskManager({"drawdown": {"enabled": True, "mode": "linear",
                                              "dd_tol": 0.20, "dd_floor": 0.0}})
        self._w(rm, 1.0)
        self.assertAlmostEqual(self._w(rm, 0.79), 0.0)    # dd ≥ tol → 清仓
        self.assertAlmostEqual(self._w(rm, 0.90), 0.50)   # 回补即渐进恢复

    def test_mode_validation(self):
        with self.assertRaises(ValueError):
            FuturesRiskManager({"drawdown": {"enabled": True, "mode": "fancy"}})
        with self.assertRaises(ValueError):
            FuturesRiskManager({"drawdown": {"enabled": True, "mode": "linear",
                                             "dd_tol": 0.0}})

    def test_default_mode_is_single(self):
        # 不写 mode → single（既有语义与测试零回归的根）
        rm = FuturesRiskManager(dict(_DD_CFG))
        self.assertEqual(rm.dd_mode, "single")
        self._w(rm, 1.0)
        self.assertAlmostEqual(self._w(rm, 0.85), 0.5)
        self.assertTrue(rm.diagnostics()["dd_mode"] == "single")


class EngineDDWiringTest(unittest.TestCase):
    """_run_date_loop 接线：净值跌破 dd_entry 后全权重 × dd_scale。"""

    def _run(self, risk_cfg, n=40):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="B"),
                           "close": [100.0] * n})
        df["settle"] = df["close"]
        idx = list(pd.to_datetime(df["date"]).values)
        roi = pd.Series(-0.01, index=idx)              # 恒多策略每日亏 1%
        vol = pd.Series(0.15, index=idx)
        return _run_date_loop(
            data={"A": df}, roi={"A": roi}, vol={"A": vol},
            dates=idx, strat=_probe_strat(), class_name="dd_probe",
            need_carry=False, carry_map={}, margin_ratios={"A": 0.1},
            risk_cfg=risk_cfg, commission=0.0, slippage=0.0,
            weight_method="equal", rebalance_days=1, attribution=False,
            n_vars=1, roll_masks={})

    def test_dd_halves_weights_after_threshold(self):
        off = self._run({})
        on = self._run(dict(_DD_CFG))
        w_off = off["weights_log"]["A"]
        w_on = on["weights_log"]["A"]
        # 无 vol-target/杠杆/预算：DD off 除首日（bars 空 → hold → 0）恒满仓
        self.assertEqual(w_off[0], 0.0)
        self.assertTrue(all(abs(w - 1.0) < 1e-9 for w in w_off[1:]))
        # 触发前逐日与 off 一致（确定性前缀）；触发后恒半仓
        pre = [i for i, w in enumerate(w_on) if abs(w - 0.5) < 1e-9]
        self.assertTrue(pre, "应存在降杠杆日")
        first = pre[0]
        self.assertEqual(w_on[0], 0.0)
        self.assertTrue(all(abs(w - 1.0) < 1e-9 for w in w_on[1:first]))
        self.assertTrue(all(abs(w - 0.5) < 1e-9 for w in w_on[first:]))
        # 首日 nav=1.0、次日才起亏 → 触发不早于第 11 日（0.99^9≈0.914>0.90）
        self.assertGreaterEqual(first, 10)
        # 亏损减半 → DD on 的终值更高
        self.assertGreater(on["nav_final"], off["nav_final"])

    def test_disabled_cfg_matches_no_cfg(self):
        # enabled=false 的显式配置 ≡ 无 drawdown 键（零回归）
        a = self._run({"drawdown": {"enabled": False}})
        b = self._run({})
        self.assertAlmostEqual(a["nav_final"], b["nav_final"], places=12)


class LongestDrawdownDaysTest(unittest.TestCase):
    """最长回撤持续期纯函数：峰→谷→复峰间隔；尾部未恢复计到末尾。"""

    def test_monotonic_up_is_zero(self):
        self.assertEqual(longest_drawdown_days(pd.Series([1.0, 1.1, 1.2, 1.3])), 0)

    def test_single_dip_and_recovery(self):
        self.assertEqual(longest_drawdown_days(pd.Series([1.0, 0.9, 1.0])), 2)

    def test_recovery_requires_new_high(self):
        # 0.99 < 1.0 未复峰：同一段回撤延续到 idx3 才恢复 → 3
        self.assertEqual(longest_drawdown_days(pd.Series([1.0, 0.9, 0.99, 1.0])), 3)

    def test_unrecovered_tail_counts_to_end(self):
        self.assertEqual(longest_drawdown_days(pd.Series([1.0, 0.9, 0.8, 0.85])), 3)

    def test_two_episodes_longest_wins(self):
        nav = pd.Series([1.0, 0.95, 1.0, 0.9, 0.85, 0.9, 1.0])
        self.assertEqual(longest_drawdown_days(nav), 4)   # 第二段峰2→复峰6

    def test_short_series_is_zero(self):
        self.assertEqual(longest_drawdown_days(pd.Series([1.0])), 0)
        self.assertEqual(longest_drawdown_days(pd.Series([], dtype=float)), 0)


class TailMetricsTest(unittest.TestCase):
    """tail_metrics：最差单日 / 最差 5 日 / 最长回撤持续期。"""

    def _df(self, vals):
        return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(vals)),
                             "nav": vals})

    def test_short_or_empty_is_nan(self):
        for nav_df in (None, pd.DataFrame(), self._df([1.0] * 5)):
            m = tail_metrics(nav_df)
            self.assertTrue(all(pd.isna(v) for v in m.values()))

    def test_crash_day_metrics(self):
        m = tail_metrics(self._df([1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.95, 1.0]))
        self.assertAlmostEqual(m["worst_day"], -0.1)
        self.assertAlmostEqual(m["worst_5d"], 0.9 / 1.0 - 1.0)   # 5 日窗含崩盘日
        self.assertEqual(m["longest_dd_days"], 3)                # 峰4→复峰7

    def test_worst_5d_spans_window(self):
        vals = [1.0, 1.0, 1.0, 1.0, 1.0, 0.99, 0.98, 0.97, 0.96, 1.05]
        m = tail_metrics(self._df(vals))
        self.assertAlmostEqual(m["worst_5d"], 0.96 / 1.0 - 1.0)


class VariantCfgTest(unittest.TestCase):
    """证据脚本变体构造：深拷贝隔离 + 参数覆写合并。"""

    def _cfg(self):
        return {"futures": {"risk": {"vol_target": {"enabled": True, "w_min": 0.20},
                                     "drawdown": {"enabled": False, "mode": "single",
                                                  "dd_entry": 0.10, "dd_exit": 0.05,
                                                  "dd_scale": 0.5}}}}

    def test_base_disables_dd(self):
        c = _variant_cfg(self._cfg(), None, None)
        self.assertFalse(c["futures"]["risk"]["drawdown"]["enabled"])
        self.assertEqual(c["futures"]["risk"]["vol_target"]["w_min"], 0.20)

    def test_overrides_merge_and_enable(self):
        dd = {"mode": "linear", "dd_tol": 0.15, "dd_floor": 0.25}
        c = _variant_cfg(self._cfg(), 0.0, dd)
        self.assertTrue(c["futures"]["risk"]["drawdown"]["enabled"])
        self.assertEqual(c["futures"]["risk"]["drawdown"]["dd_tol"], 0.15)
        self.assertEqual(c["futures"]["risk"]["vol_target"]["w_min"], 0.0)
        # 未覆写的键保留（合并而非整体替换）
        self.assertEqual(c["futures"]["risk"]["drawdown"]["dd_entry"], 0.10)

    def test_original_not_mutated(self):
        cfg = self._cfg()
        _variant_cfg(cfg, 0.10, {"mode": "linear", "dd_tol": 0.2, "dd_floor": 0.0})
        self.assertEqual(cfg["futures"]["risk"]["vol_target"]["w_min"], 0.20)
        self.assertFalse(cfg["futures"]["risk"]["drawdown"]["enabled"])
        self.assertEqual(cfg["futures"]["risk"]["drawdown"]["mode"], "single")

    def test_missing_sections_created(self):
        c = _variant_cfg({}, None, {"mode": "single", "dd_entry": 0.05})
        self.assertTrue(c["futures"]["risk"]["drawdown"]["enabled"])


if __name__ == "__main__":
    unittest.main()
