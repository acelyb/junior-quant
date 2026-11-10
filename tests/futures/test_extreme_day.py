"""tests/futures/test_extreme_day.py — 中短线 Phase 0a 极端日标注与敏感性的守护。

覆盖三块：
  1. extreme_day_mask：因果分位代理——当日极端不抬自身门槛（无前视）、
     热身期（有效观测 < min_obs）宁漏勿猜、NaN 收益不标。
     数据可用性背景：规则表无涨跌停字段（futures_rules_latest 仅
     variety/margin_ratio/multiplier），故用收益分位作停板/跳空的保守代理。
  2. _extreme_day_stats：组合级口径——极端日 = 任一持仓品种（|w|>0）当日
     被标；占比 / 收益占比 / 剔除极端日后复利年化；空仓/空 NAV 安全返回。
  3. 引擎接线：run_futures_portfolio 的 stats 含新键（追加键、不改共享
     _stats——A股零回归），且极端日被正确剔除（剔除后年化 < 全量年化）。
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import jq_framework.backtest.futures_walk_forward as wf
from jq_framework.backtest.futures_walk_forward import (
    _extreme_day_stats,
    extreme_day_mask,
)


def _roi_series(vals: list[float], periods: int) -> pd.Series:
    """N 个收益值 → index 为交易日轴的 roi 序列。"""
    idx = pd.date_range("2024-01-01", periods=periods, freq="B")
    return pd.Series(vals, index=idx[:len(vals)])


def _alt_small(n: int, spike_at: int | None = None, spike: float = 0.10) -> list[float]:
    """前 n 天交替 ±1% 的小收益，spike_at 处放一个大收益。"""
    vals = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]
    if spike_at is not None:
        vals[spike_at] = spike
    return vals


class ExtremeDayMaskTest(unittest.TestCase):
    """因果分位代理（默认 99 分位 / 504 窗 / 126 最小观测）。"""

    def test_spike_after_warmup_flagged_exactly_once(self):
        # 热身期后（≥126 个历史观测）的单日大涨被标、且只标这一天
        m = extreme_day_mask(_roi_series(_alt_small(205, spike_at=200), 205))
        self.assertTrue(bool(m.iloc[200]))
        self.assertFalse(bool(m.iloc[150]))
        self.assertEqual(int(m.sum()), 1)

    def test_warmup_not_flagged(self):
        # 热宁漏勿猜：min_obs 内的极端日不标（无足够历史定门槛）
        m = extreme_day_mask(_roi_series(_alt_small(130, spike_at=5, spike=0.50), 130))
        self.assertFalse(bool(m.iloc[5]), "热身期内的极端日不应标注")
        self.assertEqual(int(m.sum()), 0)

    def test_min_obs_knob_controls_warmup(self):
        # 同一序列：min_obs=10 时第 20 天的尖峰可标；min_obs=126 时不标
        s = _roi_series(_alt_small(30, spike_at=20), 30)
        self.assertTrue(bool(extreme_day_mask(s, min_obs=10).iloc[20]))
        self.assertEqual(int(extreme_day_mask(s, min_obs=126).sum()), 0)

    def test_causality_future_change_leaves_past_flags(self):
        # 无前视：把第 150 天改成大涨，不改变之前任何一天的标注
        a = extreme_day_mask(_roi_series(_alt_small(205), 205))
        b = extreme_day_mask(_roi_series(_alt_small(205, spike_at=150), 205))
        self.assertTrue((a.iloc[:150] == b.iloc[:150]).all())
        self.assertFalse(bool(a.iloc[150]))
        self.assertTrue(bool(b.iloc[150]), "改变当天自身被标（当天的极端是事实）")

    def test_nan_roi_not_flagged(self):
        # 缺收益日不标，且不污染滚动窗口（后续尖峰仍可标）
        vals = _alt_small(205, spike_at=200)
        vals[170] = float("nan")
        m = extreme_day_mask(_roi_series(vals, 205))
        self.assertFalse(bool(m.iloc[170]))
        self.assertTrue(bool(m.iloc[200]))


def _merged(rets: list[float], weights: dict[str, list[float]],
            n_hist: int = 205) -> tuple[dict, dict[str, pd.Series]]:
    """手造 _splice_nav 形态的 merged + 配套 roi（尖峰在最后一根）。

    roi 共 205 天：nav 只取最后 4 天（201..204），尖峰在 204（= nav 最后一天）。
    """
    idx = pd.date_range("2024-01-01", periods=n_hist, freq="B")
    roi = {"RB": _roi_series(_alt_small(n_hist, spike_at=n_hist - 1), n_hist)}
    nav_dates = idx[-len(rets):]
    nav, nav_series = 1.0, []
    for d, r in zip(nav_dates, rets):
        nav *= (1.0 + r)
        nav_series.append({"date": d, "nav": nav})
    merged = {"nav_series": nav_series, "port_ret_series": list(rets),
              "weights_log": {k: list(v) for k, v in weights.items()}}
    return merged, roi


class ExtremeDayStatsTest(unittest.TestCase):
    """组合级统计：任一持仓品种当日极端才计为组合极端日。"""

    def test_counts_ratio_share_and_ex_annualized(self):
        rets = [0.01, -0.01, 0.01, 0.10]  # 末日尖峰被标
        merged, roi = _merged(rets, {"RB": [0.5, 0.5, 0.5, 0.5]})
        st = _extreme_day_stats(merged, roi)
        self.assertEqual(st["extreme_days"], 1)
        self.assertAlmostEqual(st["extreme_day_ratio"], 0.25)
        # 收益占比 = 尖峰日收益 / 总收益 = 0.10 / 0.11
        self.assertAlmostEqual(st["extreme_ret_share"], 0.10 / 0.11)
        # 剔除后年化 = 前三日复利（1.01×0.99×1.01）按 252/4 年化
        ex_nav = 1.01 * 0.99 * 1.01
        self.assertAlmostEqual(st["ann_return_ex_extreme"],
                               ex_nav ** (252.0 / 4) - 1.0)

    def test_zero_weight_on_extreme_day_not_counted(self):
        # 极端日当天空仓该品种 → 不计组合极端日，剔除后 = 全量
        rets = [0.01, -0.01, 0.01, 0.10]
        merged, roi = _merged(rets, {"RB": [0.5, 0.5, 0.5, 0.0]})
        st = _extreme_day_stats(merged, roi)
        self.assertEqual(st["extreme_days"], 0)
        self.assertAlmostEqual(st["ann_return_ex_extreme"],
                               (1.01 * 0.99 * 1.01 * 1.10) ** (252.0 / 4) - 1.0)
        self.assertAlmostEqual(st["extreme_ret_share"], 0.0)

    def test_unheld_variety_spike_ignored(self):
        # 尖峰品种在 weights_log 里但当日零仓 → 不计；不在 weights_log → 不算
        rets = [0.01, -0.01, 0.01, 0.10]
        merged, roi = _merged(rets, {"RB": [0.5, 0.5, 0.5, 0.5],
                                     "CU": [0.0, 0.0, 0.0, 0.0]})
        idx = pd.date_range("2024-01-01", periods=205, freq="B")
        roi["CU"] = _roi_series(_alt_small(205, spike_at=204), 205)  # CU 同日尖峰但零仓
        st = _extreme_day_stats(merged, roi)
        self.assertEqual(st["extreme_days"], 1)
        merged2, roi2 = _merged(rets, {"RB": [0.5, 0.5, 0.5, 0.5]})
        st2 = _extreme_day_stats(merged2, roi2)
        self.assertAlmostEqual(st["ann_return_ex_extreme"], st2["ann_return_ex_extreme"])

    def test_total_return_near_zero_share_is_nan(self):
        rets = [0.01, -0.01, 0.01, -0.01]  # 总和恰 0 → 占比无意义
        # 末日尖峰日零仓（roi 尖峰仍在，但不计组合极端日）→ 无极端日、占比 NaN
        merged, roi = _merged(rets, {"RB": [0.5, 0.5, 0.5, 0.0]})
        st = _extreme_day_stats(merged, roi)
        self.assertEqual(st["extreme_days"], 0)
        self.assertTrue(np.isnan(st["extreme_ret_share"]))

    def test_empty_nav_safe(self):
        merged = {"nav_series": [], "port_ret_series": [], "weights_log": {}}
        st = _extreme_day_stats(merged, {})
        self.assertEqual(st["extreme_days"], 0)
        self.assertEqual(st["extreme_day_ratio"], 0.0)
        self.assertTrue(np.isnan(st["extreme_ret_share"]))
        self.assertEqual(st["ann_return_ex_extreme"], 0.0)


class EngineWiringTest(unittest.TestCase):
    """run_futures_portfolio stats 追加新键（additive，A股共享 _stats 不动）。"""

    def test_stats_has_extreme_keys_and_excludes_spike(self):
        n = 160
        idx = pd.date_range("2025-01-05", periods=n, freq="B")
        settle = 100.0 + 0.01 * np.arange(n, dtype=float)
        settle[-1] = settle[-2] * 1.10  # 末日 +10% 尖峰（159 个观测 ≥ min_obs=126）
        df = pd.DataFrame({"date": idx, "open": settle, "high": settle,
                           "low": settle, "close": settle, "settle": settle,
                           "volume": 1000.0})

        class _FakeStrat:
            factor_legs: dict = {}

            def on_bar(self, bars):
                from jq_framework.strategies.base import Signal
                return Signal(direction="long", price=float(bars["close"].iloc[-1]))

        orig = (wf.get_futures_main_continuous, wf.get_roll_yield_series,
                wf.get_futures_rules, wf.build_contract_series, wf.build_strategy)
        wf.get_futures_main_continuous = lambda v, contract="X0", read_only=False: df.copy()
        wf.get_roll_yield_series = lambda v, read_only=False: pd.DataFrame()
        wf.get_futures_rules = lambda read_only=False: {}
        wf.build_contract_series = lambda v, read_only=False: pd.DataFrame()
        wf.build_strategy = lambda sc: (_FakeStrat(), "fake")
        try:
            cfg = {"strategies": {"tsmom_cta": {"module": "x", "class": "Y",
                                                 "params": {}}},
                   "futures": {"symbols": ["A0"], "rebalance_days": 1,
                               "margin": 0.1, "cost": {}, "risk": {}}}
            res = wf.run_futures_portfolio(cfg, "tsmom_cta", contract="X0",
                                           weight_method="equal", verbose=False)
        finally:
            (wf.get_futures_main_continuous, wf.get_roll_yield_series,
             wf.get_futures_rules, wf.build_contract_series,
             wf.build_strategy) = orig

        st = res["stats"]
        for k in ("extreme_days", "extreme_day_ratio", "extreme_ret_share",
                  "ann_return_ex_extreme", "avg_hold_days", "max_hold_days",
                  "n_runs"):
            self.assertIn(k, st)
        # 单日 +10% 尖峰：恰好一个组合极端日（其余日收益单调递减、必不超 99 分位）
        self.assertEqual(st["extreme_days"], 1)
        self.assertAlmostEqual(st["extreme_day_ratio"], 1.0 / (n - 1))
        # 剔除正尖峰日 → 剔除后年化严格低于全量年化
        self.assertLess(st["ann_return_ex_extreme"], st["ann_return"])
        # 收益占比：尖峰日贡献占主导（其余日收益 ~0.01% 量级）
        self.assertGreater(st["extreme_ret_share"], 0.8)
        # 0d 同号串：全程做多（首记录日空仓断开）→ 单串 158 天（段末截断照计）
        self.assertEqual(st["n_runs"], 1)
        self.assertAlmostEqual(st["avg_hold_days"], float(n - 2))
        self.assertEqual(st["max_hold_days"], n - 2)


if __name__ == "__main__":
    unittest.main()
