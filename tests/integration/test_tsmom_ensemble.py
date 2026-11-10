"""TSMOM 多周期趋势集成单元测试（强化计划第四步）。

守护重点（最易错、最影响参数曲面结论的桩子）：
  1. 合成方向确定性：三周期全同向 / 2-3 多数票 / 分裂(含 hold 腿计 0) → long/short/hold
  2. 腿委托等价：集成内部腿方向 = 同参数 TsmomStrategy 单点方向（容器不重复实现）
  3. factor_legs 挂载：三单点 + 集成自身四腿（一次归因跑出参数曲面四张卡）
  4. 引擎归因接线：_run_date_loop --attribution 逐日簿记四腿虚拟 NAV；
     250 日腿在历史不足 251 根前恒平（hold→0 分），集成腿吃多数票方向
  5. 健康卡套件选择：--suite tsmom_ens 走 tsmom_ens_cta 载体 + 四因子；
     governance 默认套件行为不变（治理卡零回归）

不依赖网络 / 不碰缓存：本地构造 bars DataFrame + mock 数据入口，纯函数可复现。

运行：python -m unittest tests.futures.test_tsmom_ensemble
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
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
from jq_alpha.strategies.cta.tsmom import TsmomStrategy
from jq_alpha.strategies.cta.tsmom_ensemble import TsmomEnsembleStrategy


def make_bars(close: list[float]) -> pd.DataFrame:
    """构造最小 bars DataFrame[date, close, settle]（settle=close，settle 口径生效）。"""
    n = len(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({"date": dates, "close": [float(c) for c in close]})
    df["settle"] = df["close"]
    return df


def _run_engine(strat, df: pd.DataFrame, attribution: bool = True) -> dict:
    """手造单品种数据入口，驱动真实 _run_date_loop（同 test_carry_leak / S1 范式）。"""
    idx = list(pd.to_datetime(df["date"]).values)
    roi = pd.Series(0.01, index=idx)          # 每日 +1%（与涨跌构造解耦，便于断言）
    vol = pd.Series(0.15, index=idx)
    return _run_date_loop(
        data={"A": df}, roi={"A": roi}, vol={"A": vol},
        dates=idx, strat=strat, class_name="tsmom_ens_cta",
        need_carry=False, carry_map={}, margin_ratios={"A": 0.1},
        risk_cfg={}, commission=0.0, slippage=0.0,
        weight_method="equal", rebalance_days=1, attribution=attribution,
        n_vars=1, roll_masks={})


class EnsembleDirectionTest(unittest.TestCase):
    """合成分 = mean(sign(20d), sign(60d), sign(250d))，方向 = sign(合成分)。"""

    def _ens(self, **kw):
        return TsmomEnsembleStrategy(params={"lookbacks": [20, 60, 250],
                                             "price": "settle", "target_pos": 1.0, **kw})

    def test_long_when_all_three_up(self):
        # 单调涨 260 根 → 三腿全多 → score +1 → long，满仓
        bars = make_bars([100 * (1.0 + 0.001 * i) for i in range(260)])
        sig = self._ens().on_bar(bars)
        self.assertEqual(sig.direction, "long")
        self.assertEqual(sig.target_pos, 1.0)

    def test_long_when_majority_two_of_three(self):
        # 前 200 根跌、后 60 根涨：20d/60d 多、250d 空 → score +1/3 → 多数票多
        bars = make_bars([200 - i for i in range(200)] + [100 + i for i in range(60)])
        sig = self._ens().on_bar(bars)
        self.assertEqual(sig.direction, "long")

    def test_short_when_majority_two_of_three(self):
        # 镜像：前 200 根涨、后 60 根跌 → 20d/60d 空、250d 多 → score −1/3 → 空
        bars = make_bars([100 + i for i in range(200)] + [300 - i for i in range(60)])
        sig = self._ens().on_bar(bars)
        self.assertEqual(sig.direction, "short")

    def test_hold_when_split_and_short_history(self):
        # 100 根（250 腿 hold 计 0）+ 末 20 根跌、60 日涨 → (−1+1+0)/3 = 0 → hold
        bars = make_bars([100] * 40 + [100 + 2.5 * i for i in range(40)]
                         + [200 - 2.5 * i for i in range(20)])
        sig = self._ens().on_bar(bars)
        self.assertEqual(sig.direction, "hold")
        self.assertIn("无方向", sig.note)

    def test_hold_when_insufficient_for_all(self):
        # 30 根平盘：20d 腿 ret=0 hold、60d/250d 数据不足 → 全 0 → hold
        sig = self._ens().on_bar(make_bars([100] * 30))
        self.assertEqual(sig.direction, "hold")

    def test_target_pos_clamped(self):
        bars = make_bars([100 * (1.0 + 0.001 * i) for i in range(260)])
        self.assertLessEqual(self._ens(target_pos=2.5).on_bar(bars).target_pos, 1.0)
        self.assertGreaterEqual(self._ens(target_pos=-1.0).on_bar(bars).target_pos, 0.0)

    def test_note_shows_leg_scores(self):
        # note 含逐腿分数与均值（人工核对口径，与 tsmom 同风格）
        bars = make_bars([200 - i for i in range(200)] + [100 + i for i in range(60)])
        note = self._ens().on_bar(bars).note
        self.assertIn("tsmom_20:+1", note)
        self.assertIn("tsmom_250:-1", note)
        self.assertIn("+0.33", note)


class EnsembleLegDelegationTest(unittest.TestCase):
    """容器不重复实现：集成内部腿 = 同参数 TsmomStrategy 单点。"""

    def test_leg_directions_equal_standalone_tsmom(self):
        bars = make_bars([200 - i for i in range(200)] + [100 + i for i in range(60)])
        ens = TsmomEnsembleStrategy()
        for lb, leg in ens._legs.items():
            solo = TsmomStrategy(params={"lookback": int(lb.split("_")[1]),
                                         "price": "settle", "target_pos": 1.0})
            self.assertEqual(leg.on_bar(bars).direction, solo.on_bar(bars).direction,
                             f"腿 {lb} 与单点 {lb} 方向不一致")

    def test_factor_legs_wiring(self):
        # factor_legs = 三单点 + 集成自身（引擎 --attribution 一次跑出四张卡）
        ens = TsmomEnsembleStrategy()
        self.assertEqual(set(ens.factor_legs),
                         {"tsmom_20", "tsmom_60", "tsmom_250", "tsmom_ens"})
        for k, leg in ens.factor_legs.items():
            if k == "tsmom_ens":
                self.assertIs(leg, ens)  # 集成自身作为归因腿（on_bar 纯函数可重复调）
            else:
                self.assertIsInstance(leg, TsmomStrategy)
                self.assertEqual(leg.params["lookback"], int(k.split("_")[1]))

    def test_score_semantics_hold_leg_counts_zero(self):
        # _score：long→+1 / short→−1 / hold/flat→0（与 tsmom_carry 容器同语义）
        ens = TsmomEnsembleStrategy()
        self.assertEqual(ens._score(ens, make_bars([100 * (1.0 + 0.001 * i)
                                                    for i in range(260)])), 1.0)
        self.assertEqual(ens._score(ens, make_bars([100] * 30)), 0.0)


class EngineAttributionWiringTest(unittest.TestCase):
    """引擎归因接线：_run_date_loop --attribution 簿记四腿虚拟 NAV。"""

    def test_four_legs_logged_daily_with_warmup_semantics(self):
        # 单调涨 320 根：逐日 factor_attr_log 四键齐备；
        # 250 腿在历史 <251 根前恒 1.0（hold→0 分，腿间独立簿记）；
        # 集成腿在 20/60 多数票下吃到 +1% 日收益 → 期末 > 1。
        bars = make_bars([100 * (1.0 + 0.001 * i) for i in range(320)])
        res = _run_engine(TsmomEnsembleStrategy(), bars, attribution=True)
        log = res["factor_attr_log"]
        self.assertTrue(len(log) > 250, "应逐日簿记归因（含 warm-up 段）")
        keys = {"tsmom_20", "tsmom_60", "tsmom_250", "tsmom_ens"}
        for row in log:
            self.assertTrue(keys <= set(row.keys()),
                            f"归因行缺腿：{sorted(keys - set(row.keys()))}")
        # 第 i 行的日期 = dates[i+1]；当日策略可见 bars = 该日期前的全部（< 251 根 → 250 腿 hold）
        dates = list(pd.to_datetime(bars["date"]).values)
        for i, row in enumerate(log):
            bars_seen = i + 1  # dates[i+1] 前有 i+1 根（t 从 1 起、bars 严格 < dt）
            if bars_seen < 251:
                self.assertEqual(row["tsmom_250"], 1.0,
                                 f"第 {i} 行 250 腿应 hold（仅 {bars_seen} 根历史）")
        self.assertGreater(log[-1]["tsmom_250"], 1.0)   # 热身后多数同向 → 腿增值
        self.assertGreater(log[-1]["tsmom_ens"], 1.0)   # 集成腿吃到趋势收益
        self.assertGreater(log[-1]["tsmom_20"], 1.0)

    def test_ensemble_leg_tracks_own_signals_not_single_leg(self):
        # 深跌 250 根 + 平 30 根 + 尾 20 根小涨（roi=真实 pct_change）：
        # 期末 20d/60d 多、250d 空（涨不回 250 日前高位）→ 多数票 ens 多。
        # 趋势段 20/60 腿顺势增值；250 腿在尾段逆势（仍空）亏损 →
        # 各腿按自身方向分独立簿记，ens 跟多数票不跟任何单点腿。
        close = ([300 - i for i in range(250)]      # 深跌：300→51
                 + [50] * 30                        # 平台
                 + [50 + i for i in range(20)])     # 尾段小涨：50→69
        bars = make_bars(close)
        roi = pd.Series(bars["close"].pct_change().fillna(0.0).values,
                        index=pd.to_datetime(bars["date"]).values)
        vol = pd.Series(0.15, index=roi.index)
        idx = list(roi.index)
        res = _run_date_loop(
            data={"A": bars}, roi={"A": roi}, vol={"A": vol},
            dates=idx, strat=TsmomEnsembleStrategy(), class_name="tsmom_ens_cta",
            need_carry=False, carry_map={}, margin_ratios={"A": 0.1},
            risk_cfg={}, commission=0.0, slippage=0.0,
            weight_method="equal", rebalance_days=1, attribution=True,
            n_vars=1, roll_masks={})
        last = res["factor_attr_log"][-1]
        self.assertGreater(last["tsmom_20"], 1.0)      # 短腿吃跌段、多腿吃尾段
        self.assertGreater(last["tsmom_60"], 1.0)
        self.assertLess(last["tsmom_250"], 1.0)        # 尾段仍空 → 逆势亏损
        self.assertGreater(last["tsmom_ens"], 1.0)     # 多数票全程顺势
        self.assertGreater(last["tsmom_ens"], last["tsmom_250"])

    def test_attribution_off_no_legs_logged(self):
        bars = make_bars([100 * (1.0 + 0.001 * i) for i in range(60)])
        res = _run_engine(TsmomEnsembleStrategy(), bars, attribution=False)
        self.assertEqual(res["factor_attr_log"], [])


class HealthSuiteSelectionTest(unittest.TestCase):
    """健康卡套件选择：tsmom_ens 走专用载体+四因子；governance 默认行为不变。"""

    def test_tsmom_ens_suite_uses_ensemble_carrier(self):
        from scripts.futures_factor_health import (
            _SUITE_GOVERNANCE, _SUITE_TSMOM_ENS, _TSMOM_ENS_CARRIER,
            _TSMOM_ENS_FACTORS, health_card,
        )
        cfg = {"strategies": {}}
        with mock.patch("scripts.futures_factor_health._cards_from_carrier",
                        return_value=([], {})) as m:
            health_card(cfg, 504, 126, 126, "built", "vol_parity",
                        suite=_SUITE_TSMOM_ENS)
            m.assert_called_once_with(cfg, _TSMOM_ENS_CARRIER, _TSMOM_ENS_FACTORS,
                                      504, 126, 126, "built", "vol_parity",
                                      read_only=False)
            # read_only 透传到引擎（cache-only 复核跑法）
            m.reset_mock()
            health_card(cfg, 504, 126, 126, "built", "vol_parity",
                        suite=_SUITE_TSMOM_ENS, read_only=True)
            m.assert_called_once_with(cfg, _TSMOM_ENS_CARRIER, _TSMOM_ENS_FACTORS,
                                      504, 126, 126, "built", "vol_parity",
                                      read_only=True)

    def test_governance_suite_unchanged(self):
        # 默认套件：triple_factor 容器 + 单因子载体（与本改动前行为一致，治理卡零回归）
        from scripts.futures_factor_health import (
            _FACTORS, _EXTRA_CARRIERS, _SUITE_GOVERNANCE, health_card,
        )
        cfg = {"strategies": {c: {"module": "jq_alpha.strategies.cta.tsmom",
                                  "class": "TsmomStrategy"} for c in _EXTRA_CARRIERS}}
        with mock.patch("scripts.futures_factor_health._cards_from_carrier",
                        return_value=([], {})) as m:
            health_card(cfg, 504, 126, 126, "built", "vol_parity",
                        suite=_SUITE_GOVERNANCE)
            called = [args[0][1] for args in m.call_args_list]  # 第二位=载体名
            self.assertEqual(called, ["triple_factor_cta"] + list(_EXTRA_CARRIERS))
            self.assertEqual(m.call_args_list[0][0][2], _FACTORS)


if __name__ == "__main__":
    unittest.main()
