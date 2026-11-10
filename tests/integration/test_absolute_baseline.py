"""tests/futures/test_absolute_baseline.py — 第一步绝对绩效基准的守护。

覆盖三块：
  1. _splice_nav 基准口径修正（2026-10-14）：基准须按自身段终值复利链拼接，
     不再误乘策略 cum（旧口径拼出"策略权益×段内基准增长"混合曲线）。
  2. _per_window_summary：分窗段级摘要（起止/天数/策略与基准段内终值），
     attribution 关闭时日期不再为 None；归因开启时与 nav_series 同日期轴。
  3. 报表纯函数：window_metrics（段内几何年化/超额）、overall_metrics
     （全样本聚合、成本占比、Calmar/换手）与 build_rows（w01..ALL 行组装）。
"""
from __future__ import annotations

import math
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

from jq_framework.backtest.futures_walk_forward import (
    _per_window_summary,
    _splice_nav,
)
from futures_absolute_baseline import (
    build_rows,
    overall_metrics,
    window_metrics,
)


def _fake_seg(nav_vals, bench_vals, dates):
    """手造一段 _run_date_loop 聚合结果（attribution 关闭形态）。"""
    return {
        "nav_series": [{"date": d, "nav": v} for d, v in zip(dates, nav_vals)],
        "bench_series": [{"date": d, "nav": v} for d, v in zip(dates, bench_vals)],
        "port_ret_series": [0.0] * len(dates),
        "turnover_series": [{"date": d, "turnover": 0.0} for d in dates],
        "cost_series": [{"date": d, "cost": 0.0} for d in dates],
        "margin_series": [{"date": d, "margin": 0.0} for d in dates],
        "attribution": {"RB": [0.0] * len(dates)},
        "weights_log": {"RB": [0.0] * len(dates)},
        "factor_attr_log": [],
        "nav_final": nav_vals[-1],
    }


class SpliceBenchChainingTest(unittest.TestCase):
    """拼接基准独立复利链（第一步口径修正的回归锚）。"""

    def setUp(self):
        d1 = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
              pd.Timestamp("2024-01-04")]
        d2 = [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-02")]
        # 段1：策略 1.0→1.21（+21%），基准 1.0→1.02（+2%）
        # 段2：策略 1.0→1.05（+5%），基准 1.0→1.01（+1%）
        self.seg1 = _fake_seg([1.0, 1.1, 1.21], [1.0, 0.99, 1.02], d1)
        self.seg2 = _fake_seg([1.0, 1.05], [1.0, 1.01], d2)
        self.out = _splice_nav([self.seg1, self.seg2])

    def test_nav_chains_strategy_finals(self):
        # 策略 NAV 拼接 = 段终值连乘：1.21 × 1.05
        self.assertAlmostEqual(self.out["nav_final"], 1.21 * 1.05)
        self.assertAlmostEqual(self.out["nav_series"][-1]["nav"], 1.21 * 1.05)

    def test_bench_chains_own_finals_not_strategy_cum(self):
        # 基准拼接 = 基准段终值连乘：1.02 × 1.01。
        # 旧口径误乘策略 cum → 1.21 × 1.01（混合曲线），此断言即防回退。
        self.assertAlmostEqual(self.out["bench_series"][-1]["nav"], 1.02 * 1.01)
        self.assertNotAlmostEqual(self.out["bench_series"][-1]["nav"], 1.21 * 1.01)

    def test_first_segment_bench_unchanged(self):
        # 首段 cum_bench=1：基准原值不动（段内基准从 1.0 起独立累计）
        self.assertAlmostEqual(self.out["bench_series"][0]["nav"], 1.0)
        self.assertAlmostEqual(self.out["bench_series"][2]["nav"], 1.02)

    def test_within_segment_bench_scaled_by_prior_bench_cum(self):
        # 段2 基准各值 = 段1 基准终值 × 段2 原值（1.02 × 1.01）
        self.assertAlmostEqual(self.out["bench_series"][3]["nav"], 1.02)
        self.assertAlmostEqual(self.out["bench_series"][4]["nav"], 1.02 * 1.01)


class PerWindowSummaryTest(unittest.TestCase):
    """段级摘要：attribution 关闭时日期可用 + 基准终值暴露。"""

    def setUp(self):
        d1 = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
              pd.Timestamp("2024-01-04")]
        d2 = [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-02")]
        self.segs = [_fake_seg([1.0, 1.1, 1.21], [1.0, 0.99, 1.02], d1),
                     _fake_seg([1.0, 1.05], [1.0, 1.01], d2)]

    def test_dates_and_sizes_without_attribution(self):
        pw = _per_window_summary(self.segs)
        self.assertEqual(len(pw), 2)
        self.assertEqual(pw[0]["start"], pd.Timestamp("2024-01-02"))
        self.assertEqual(pw[0]["end"], pd.Timestamp("2024-01-04"))
        self.assertEqual(pw[0]["n_days"], 3)
        self.assertEqual(pw[1]["n_days"], 2)

    def test_finals_exposed(self):
        pw = _per_window_summary(self.segs)
        self.assertAlmostEqual(pw[0]["nav_final"], 1.21)
        self.assertAlmostEqual(pw[0]["bench_final"], 1.02)
        self.assertAlmostEqual(pw[1]["nav_final"], 1.05)
        self.assertAlmostEqual(pw[1]["bench_final"], 1.01)

    def test_attribution_on_same_date_axis(self):
        # 归因开启：factor_attr_log 有值，日期仍取 nav_series（同一天循环产出）
        segs = [dict(s) for s in self.segs]
        for s, dates in zip(segs, [[pd.Timestamp("2024-01-02")] * 3,
                                   [pd.Timestamp("2024-07-01")] * 2]):
            s["factor_attr_log"] = [{"date": d, "nav": 1.0, "tsmom": 1.0}
                                    for d in dates]
        pw = _per_window_summary(segs)
        self.assertEqual(pw[0]["end"], pd.Timestamp("2024-01-04"))
        self.assertEqual(len(pw[0]["factor_attr_log"]), 3)


class WindowMetricsTest(unittest.TestCase):
    """段内几何年化 / 基准年化 / 超额。"""

    def test_half_year_window(self):
        m = window_metrics(1.10, 1.0, 126)
        self.assertAlmostEqual(m["ann_return"], 1.10 ** 2 - 1)   # +21%
        self.assertAlmostEqual(m["bench_ann"], 0.0)
        self.assertAlmostEqual(m["excess_ann"], 0.21)

    def test_full_year_window_negative_bench(self):
        m = window_metrics(1.0, 0.90, 252)
        self.assertAlmostEqual(m["ann_return"], 0.0)
        self.assertAlmostEqual(m["bench_ann"], -0.10)
        self.assertAlmostEqual(m["excess_ann"], 0.10)

    def test_invalid_inputs_nan(self):
        for nav, bench, n in [(0.0, 1.0, 126), (1.0, 0.0, 126), (1.0, 1.0, 0)]:
            m = window_metrics(nav, bench, n)
            self.assertTrue(math.isnan(m["ann_return"]))
            self.assertTrue(math.isnan(m["excess_ann"]))


def _res(ann, bench=0.05, sharpe=0.6, mdd=-0.20, n_days=1512):
    return {"stats": {"ann_return": ann, "bench_ann": bench, "sharpe": sharpe,
                      "max_drawdown": mdd, "n_days": n_days,
                      # Phase 0a/0d 引擎追加键（ALL 行透传）
                      "extreme_day_ratio": 0.03, "extreme_ret_share": 0.25,
                      "ann_return_ex_extreme": ann - 0.01,
                      "avg_hold_days": 42.0, "max_hold_days": 120, "n_runs": 9},
            "turnover_series": pd.DataFrame({"turnover": [0.1, 0.3]}),
            "per_window": [
                {"start": pd.Timestamp("2024-01-02"), "end": pd.Timestamp("2024-07-01"),
                 "n_days": 126, "nav_final": 1.10, "bench_final": 1.00},
                {"start": pd.Timestamp("2024-07-02"), "end": pd.Timestamp("2024-12-31"),
                 "n_days": 126, "nav_final": 0.95, "bench_final": 1.02},
            ],
            "nav_df": pd.DataFrame({
                "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31")],
                "nav": [1.0, 1.045]}),
            }


class OverallMetricsTest(unittest.TestCase):
    """全样本聚合：成本拖累/占比、Calmar、换手年化、×2 与 ×3 压力列。"""

    def test_basic_aggregation(self):
        om = overall_metrics(_res(0.20), _res(0.16),
                             _res(0.12, sharpe=0.45, mdd=-0.25),
                             _res(0.08, sharpe=0.35, mdd=-0.28))
        self.assertAlmostEqual(om["ann_return"], 0.16)
        self.assertAlmostEqual(om["ann_return_gross"], 0.20)
        self.assertAlmostEqual(om["cost_drag_ann"], 0.04)
        self.assertAlmostEqual(om["cost_share"], 0.04 / 0.20)
        self.assertAlmostEqual(om["excess_ann"], 0.16 - 0.05)
        self.assertAlmostEqual(om["calmar"], 0.16 / 0.20)
        self.assertAlmostEqual(om["ann_turnover"], 0.2 * 252)
        # ×2 压力档
        self.assertAlmostEqual(om["ann_return_cost2x"], 0.12)
        self.assertAlmostEqual(om["cost_drag_ann_cost2x"], 0.08)
        self.assertAlmostEqual(om["cost_share_cost2x"], 0.08 / 0.20)
        self.assertAlmostEqual(om["calmar_cost2x"], 0.12 / 0.25)
        # ×3 止损滑点敏感档（Phase 0b）
        self.assertAlmostEqual(om["ann_return_cost3x"], 0.08)
        self.assertAlmostEqual(om["cost_drag_ann_cost3x"], 0.12)
        self.assertAlmostEqual(om["cost_share_cost3x"], 0.12 / 0.20)
        self.assertAlmostEqual(om["calmar_cost3x"], 0.08 / 0.28)
        self.assertAlmostEqual(om["excess_ann_cost3x"], 0.08 - 0.05)

    def test_share_nan_when_gross_nonpositive(self):
        om = overall_metrics(_res(-0.05), _res(-0.08), _res(-0.10), _res(-0.12))
        self.assertAlmostEqual(om["cost_drag_ann"], 0.03)
        self.assertTrue(math.isnan(om["cost_share"]))
        self.assertTrue(math.isnan(om["cost_share_cost3x"]))

    def test_mdd_zero_calmar_nan(self):
        om = overall_metrics(_res(0.20), _res(0.16, mdd=0.0),
                             _res(0.12, mdd=0.0), _res(0.08, mdd=0.0))
        self.assertTrue(math.isnan(om["calmar"]))
        self.assertTrue(math.isnan(om["calmar_cost3x"]))


class BuildRowsTest(unittest.TestCase):
    """行组装：逐窗口 w01..wNN + ALL；×2/×3 窗口列来自对应成本档结果。"""

    def test_rows_layout(self):
        rows = build_rows("tsmom_carry_cta", _res(0.20), _res(0.16),
                          _res(0.12), _res(0.08))
        self.assertEqual([r["scope"] for r in rows], ["w01", "w02", "ALL"])
        w1 = rows[0]
        self.assertAlmostEqual(w1["ann_return"], 1.10 ** 2 - 1)
        self.assertAlmostEqual(w1["bench_ann"], 0.0)
        self.assertAlmostEqual(w1["ann_return_cost2x"], 1.10 ** 2 - 1)
        self.assertAlmostEqual(w1["ann_return_cost3x"], 1.10 ** 2 - 1)
        all_row = rows[-1]
        self.assertAlmostEqual(all_row["ann_return"], 0.16)
        self.assertAlmostEqual(all_row["nav_final"], 1.045)
        self.assertAlmostEqual(all_row["sharpe"], 0.6)
        self.assertAlmostEqual(all_row["ann_return_cost3x"], 0.08)
        # Phase 0a/0d 追加键从 ×1 stats 透传进 ALL 行
        self.assertAlmostEqual(all_row["extreme_day_ratio"], 0.03)
        self.assertAlmostEqual(all_row["extreme_ret_share"], 0.25)
        self.assertAlmostEqual(all_row["ann_return_ex_extreme"], 0.15)
        self.assertAlmostEqual(all_row["avg_hold_days"], 42.0)
        self.assertEqual(all_row["max_hold_days"], 120)
        self.assertEqual(all_row["n_runs"], 9)

    def test_x2_window_differs_when_finals_differ(self):
        # ×2 的 per_window 终值不同 → w 行 ×2 列不同（两档成本真实分开）
        res_x2 = _res(0.12)
        res_x2["per_window"] = [
            {"start": pd.Timestamp("2024-01-02"), "end": pd.Timestamp("2024-07-01"),
             "n_days": 126, "nav_final": 1.05, "bench_final": 1.00},
            {"start": pd.Timestamp("2024-07-02"), "end": pd.Timestamp("2024-12-31"),
             "n_days": 126, "nav_final": 0.93, "bench_final": 1.02},
        ]
        rows = build_rows("s", _res(0.20), _res(0.16), res_x2, _res(0.08))
        self.assertAlmostEqual(rows[0]["ann_return_cost2x"], 1.05 ** 2 - 1)
        self.assertAlmostEqual(rows[1]["ann_return_cost2x"], 0.93 ** 2 - 1)

    def test_x3_window_differs_when_finals_differ(self):
        # ×3 的 per_window 终值不同 → w 行 ×3 列不同（Phase 0b 第三压力档）
        res_x3 = _res(0.08)
        res_x3["per_window"] = [
            {"start": pd.Timestamp("2024-01-02"), "end": pd.Timestamp("2024-07-01"),
             "n_days": 126, "nav_final": 1.03, "bench_final": 1.00},
            {"start": pd.Timestamp("2024-07-02"), "end": pd.Timestamp("2024-12-31"),
             "n_days": 126, "nav_final": 0.90, "bench_final": 1.02},
        ]
        rows = build_rows("s", _res(0.20), _res(0.16), _res(0.12), res_x3)
        self.assertAlmostEqual(rows[0]["ann_return_cost3x"], 1.03 ** 2 - 1)
        self.assertAlmostEqual(rows[1]["ann_return_cost3x"], 0.90 ** 2 - 1)
        # ×2 列不受 ×3 结果影响
        self.assertAlmostEqual(rows[0]["ann_return_cost2x"], 1.10 ** 2 - 1)


if __name__ == "__main__":
    unittest.main()
