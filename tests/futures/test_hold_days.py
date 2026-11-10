"""tests/futures/test_hold_days.py — 中短线 Phase 0d 同号串持仓统计的守护。

口径（hold_days_stats）：每品种权重序列的连续同号串（0 仓断串、变号即
前串结束开新串）；串长 ≈ 每日重判框架下单次持仓天数。全部串等权平均，
段末/样本末被窗口截断的尾串照计（偏短方向、各策略同窗口下可比）。
"""
from __future__ import annotations

import unittest

from jq_framework.backtest.futures_walk_forward import hold_days_stats


class HoldDaysStatsTest(unittest.TestCase):

    def test_single_run(self):
        st = hold_days_stats({"RB": [0.5] * 5})
        self.assertAlmostEqual(st["avg_hold_days"], 5.0)
        self.assertEqual(st["max_hold_days"], 5)
        self.assertEqual(st["n_runs"], 1)

    def test_sign_flip_starts_new_run_without_zero(self):
        # 变号日 = 前串结束 + 新串开始（反手不产生空仓日，也断串）
        st = hold_days_stats({"RB": [0.5, 0.5, -0.5]})
        self.assertEqual(st["n_runs"], 2)
        self.assertAlmostEqual(st["avg_hold_days"], 1.5)
        self.assertEqual(st["max_hold_days"], 2)

    def test_zero_breaks_run(self):
        st = hold_days_stats({"RB": [0.5, 0.5, 0.0, 0.5]})
        self.assertEqual(st["n_runs"], 2)
        self.assertAlmostEqual(st["avg_hold_days"], 1.5)
        self.assertEqual(st["max_hold_days"], 2)

    def test_tail_run_censored_but_counted(self):
        # 样本末仍持仓（无 0 收尾）：尾串照计（截断偏短，如实口径）
        st = hold_days_stats({"RB": [0.5, 0.5]})
        self.assertEqual(st["n_runs"], 1)
        self.assertAlmostEqual(st["avg_hold_days"], 2.0)

    def test_multi_variety_pooled(self):
        st = hold_days_stats({"RB": [1.0, 1.0, 0.0], "CU": [-1.0]})
        self.assertEqual(st["n_runs"], 2)
        self.assertAlmostEqual(st["avg_hold_days"], 1.5)
        self.assertEqual(st["max_hold_days"], 2)

    def test_all_zero_and_empty(self):
        for wl in [{"RB": [0.0] * 4}, {}]:
            st = hold_days_stats(wl)
            self.assertEqual(st["avg_hold_days"], 0.0)
            self.assertEqual(st["max_hold_days"], 0)
            self.assertEqual(st["n_runs"], 0)

    def test_short_variety_mixed_with_long(self):
        # 1c gate 场景：短线腿串短、长线腿串长 → 均值介于其间
        st = hold_days_stats({
            "RB": [1.0] * 100,        # 1 串 100 天
            "CU": [1.0, -1.0] * 10,   # 20 串各 1 天
        })
        self.assertEqual(st["n_runs"], 21)
        self.assertAlmostEqual(st["avg_hold_days"], 120.0 / 21)
        self.assertEqual(st["max_hold_days"], 100)


if __name__ == "__main__":
    unittest.main()
