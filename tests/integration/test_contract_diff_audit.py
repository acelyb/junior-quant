"""futures_contract_diff_audit.py 单测（期货CTA_强化计划 第二步：切换前差异审计）。

守护重点：
  1. X0 含换月跳空 → 250 日累计收益翻符号 → 信号方向不一致被逐日点名
  2. built 兜底品种（无 adj_close）标"无自建数据（X0兜底）"跳过，不产生假一致
  3. 两口径序列逐日同收益 → 0 不一致（跳空是差异的唯一来源）
  4. 预热期（< lookback+1 bars）不算可比日

手造几何序列 + mock 数据源（网络无关）；carry 注入空序列（carry 腿两实例同分，
差异只来自 TSMOM 腿——与脚本设计口径一致）。
运行：python -m unittest tests.futures.test_contract_diff_audit
"""
from __future__ import annotations

# 私有库路径（研究脚本已移入 junior-quant-alpha）
import sys as _sys
from pathlib import Path as _Path
PRIVATE_PROBES = _Path(__file__).resolve().parents[2] / 'private'
for _d in ('probes', 'experiments'):
    _p = PRIVATE_PROBES / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
        break

import unittest
from unittest import mock

import numpy as np
import pandas as pd

import scripts.futures_contract_diff_audit as audit

_STRAT_CFG = {
    "type": "cta", "module": "jq_alpha.strategies.cta.tsmom_carry",
    "class": "TsmomCarryStrategy",
    "params": {"weight": 0.5,
               "tsmom": {"lookback": 250, "price": "settle", "target_pos": 1.0},
               "carry": {"smooth": 5, "target_pos": 1.0}},
}


def _x0_frame(settle: pd.Series) -> pd.DataFrame:
    """新浪 X0 主连风格 8 列帧。"""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(settle), freq="B"),
        "open": settle.to_numpy(), "high": settle.to_numpy(),
        "low": settle.to_numpy(), "close": settle.to_numpy(),
        "settle": settle.to_numpy(), "volume": 1000.0, "hold": 1000.0,
    })


def _built_frame(adj_return: pd.Series) -> pd.DataFrame:
    """自建换月帧（build_contract_series 输出风格）。"""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(adj_return), freq="B"),
        "close": np.cumprod(1.0 + adj_return.to_numpy()) * 100.0,
        "adj_close": np.cumprod(1.0 + adj_return.to_numpy()) * 100.0,
        "adj_return": adj_return.to_numpy(),
        "hold": 1000.0, "settle": 0.0,
        "main_contract": "XX2401", "is_roll": False,
    })


def _drift(n: int, daily: float) -> pd.Series:
    return pd.Series(np.cumprod([1.0] + [1.0 + daily] * (n - 1)) * 100.0)


class AuditVarietyTest(unittest.TestCase):
    """mock 三数据源，直接驱动 audit_variety。"""

    def _run_audit(self, x0: pd.DataFrame | None, built: pd.DataFrame | None) -> dict:
        with mock.patch.object(audit, "get_futures_main_continuous",
                               return_value=x0), \
             mock.patch.object(audit, "build_contract_series",
                               return_value=built), \
             mock.patch.object(audit, "get_roll_yield_series",
                               return_value=pd.DataFrame()):
            return audit.audit_variety("XX", _STRAT_CFG)

    def test_roll_jump_flips_direction_and_is_reported(self):
        """X0 在第 50 日有 +40% 换月跳空：built 一路阴跌(short)，X0 的 250 日窗
        含跳空时翻 long → 可比日全部不一致，明细行 x0=long/built=short。"""
        n = 300
        daily = -0.0005                      # built：日跌 0.05%，250 日 ≈ −11.8%
        b = _drift(n, daily)
        x = b.copy()
        x.iloc[50:] *= 1.40                  # X0：跳空 +40%（复权序列没有这一跳）
        res = self._run_audit(_x0_frame(x), _built_frame(
            pd.Series([0.0] + [daily] * (n - 1))))
        self.assertEqual(res["status"], "ok")
        self.assertGreater(res["n_days"], 0)
        self.assertGreater(res["n_mismatch"], 0)
        self.assertEqual(res["n_mismatch"], res["n_days"])  # 可比窗全程含跳空
        for row in res["rows"]:
            self.assertEqual(row["x0_dir"], "long")
            self.assertEqual(row["built_dir"], "short")

    def test_identical_series_zero_mismatch(self):
        """两口径逐日同收益（X0 无跳空、built adj_return 同几何）→ 0 不一致。"""
        n = 300
        s = _drift(n, 0.001)
        res = self._run_audit(_x0_frame(s), _built_frame(
            pd.Series([0.0] + [0.001] * (n - 1))))
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_mismatch"], 0)
        self.assertEqual(res["n_days"], 299 - 250)  # 预热后全部可比
        self.assertEqual(res["rows"], [])

    def test_fallback_variety_marked_and_skipped(self):
        """built 兜底返 X0 原始帧（无 adj_close）→ 无自建数据，不产假一致。"""
        n = 300
        res = self._run_audit(_x0_frame(_drift(n, 0.001)),
                              _x0_frame(_drift(n, 0.001)))
        self.assertEqual(res["status"], "无自建数据（X0兜底）")
        self.assertEqual(res["n_days"], 0)
        self.assertEqual(res["n_mismatch"], 0)
        self.assertEqual(res["rows"], [])

    def test_warmup_days_not_comparable(self):
        """预热期（双方均 < 251 bars → hold）不算可比日：n_days 从首个满 251 bar
        的日期起算，而非全轴 299 日。"""
        n = 300
        s = _drift(n, 0.001)
        res = self._run_audit(_x0_frame(s), _built_frame(
            pd.Series([0.0] + [0.001] * (n - 1))))
        # 日期轴 300 日，t=1..299；bars < dt 需 ≥ 251 → 首个可比 t=251（dt 起）
        self.assertEqual(res["n_days"], 299 - 250)


class LookbackTest(unittest.TestCase):
    def test_lookback_threshold(self):
        # tsmom.on_bar 判据 len(price) < lookback+1 → hold；审计预热门槛同根
        self.assertEqual(audit._lookback(_STRAT_CFG), 251)
        self.assertEqual(audit._lookback({"params": {}}), 251)  # 缺省 250


if __name__ == "__main__":
    unittest.main()
