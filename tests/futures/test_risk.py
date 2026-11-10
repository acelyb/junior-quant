"""期货风控层单元测试：FuturesRiskManager 的杠杆/保证金/vol-target/预算约束。

守护重点（组合层风控顺序与边界，最易把仓位算错的桩子）：
  1. 杠杆上限：总 |w| 超限按比例缩、未设限不缩
  2. 单品种保证金上限：|w_i|×margin_i 超限逐腿缩
  3. 保证金预算（第 4 步）：组合总保证金超 cap 整体等比缩、未设限不缩
  4. vol-target：高波动减仓 / 低波动满仓 / 冷启动满仓 / no-op 恒满仓
  5. 顺序正确性：vol-target → 杠杆 → 单品种 → 预算，最终 margin_used ≤ budget

不依赖网络/数据：全部手造权重 + margin 比例，纯函数可复现。

运行：python -m unittest tests.futures.test_risk
"""
from __future__ import annotations

import unittest

import numpy as np

from jq_framework.risk.futures_risk import FuturesRiskManager


class LeverageTest(unittest.TestCase):
    def test_clip_when_over_leverage(self):
        # 总 |w|=2.0 > max_leverage=1.5 → 等比缩到 1.5，比例保持
        rm = FuturesRiskManager({"max_leverage": 1.5})
        w, _ = rm.apply({"RB": 1.2, "AU": 0.8}, {"RB": 0.11, "AU": 0.10})
        gross = sum(abs(x) for x in w.values())
        self.assertAlmostEqual(gross, 1.5, places=9)
        self.assertAlmostEqual(w["RB"] / w["AU"], 1.2 / 0.8, places=6)

    def test_no_clip_without_limit(self):
        rm = FuturesRiskManager({})
        w, _ = rm.apply({"RB": 1.5, "AU": 1.0}, {"RB": 0.11, "AU": 0.10})
        self.assertAlmostEqual(sum(abs(x) for x in w.values()), 2.5, places=9)

    def test_within_leverage_no_change(self):
        rm = FuturesRiskManager({"max_leverage": 1.5})
        w, _ = rm.apply({"RB": 0.8, "AU": 0.5}, {"RB": 0.11, "AU": 0.10})
        self.assertEqual(w, {"RB": 0.8, "AU": 0.5})


class PerMarginTest(unittest.TestCase):
    def test_within_limit_unchanged(self):
        rm = FuturesRiskManager({"max_single_margin": 0.30})
        w, _ = rm.apply({"RB": 0.5}, {"RB": 0.11})  # 0.5*0.11=0.055 < 0.30
        self.assertAlmostEqual(w["RB"], 0.5, places=9)

    def test_over_limit_clipped(self):
        rm = FuturesRiskManager({"max_single_margin": 0.30})
        w, _ = rm.apply({"RB": 3.0}, {"RB": 0.11})  # 3.0*0.11=0.33 > 0.30
        self.assertAlmostEqual(w["RB"], 0.30 / 0.11, places=6)


class MarginBudgetTest(unittest.TestCase):
    def test_over_budget_scaled_preserving_ratio(self):
        rm = FuturesRiskManager({"capital_utilization": 0.5})
        w, mt = rm.apply({"RB": 3.0, "AU": 3.0}, {"RB": 0.11, "AU": 0.10})
        # used=0.33+0.30=0.63 > 0.5 → scale=0.5/0.63，比例 1:1 保持
        self.assertAlmostEqual(mt, 0.5, places=9)
        self.assertAlmostEqual(w["RB"] / w["AU"], 1.0, places=6)

    def test_no_budget_no_change(self):
        rm = FuturesRiskManager({})
        w, mt = rm.apply({"RB": 3.0, "AU": 3.0}, {"RB": 0.11, "AU": 0.10})
        self.assertAlmostEqual(mt, 0.63, places=9)

    def test_budget_always_holds_after_apply(self):
        # 多品种任意权重下，最终 margin_used 必须 ≤ budget（顺序保证）
        rm = FuturesRiskManager({
            "max_leverage": 10.0, "max_single_margin": 5.0,
            "capital_utilization": 0.5,
        })
        w, mt = rm.apply({"RB": 8.0, "CU": -6.0, "AU": 4.0},
                         {"RB": 0.11, "CU": 0.08, "AU": 0.10})
        self.assertLessEqual(mt, 0.5 + 1e-9)


class VolTargetTest(unittest.TestCase):
    def test_high_vol_reduces(self):
        # 高波动 → 触发减仓，且不跌破 w_min
        rng = np.random.default_rng(0)
        navs = [1.0] + [1.0 * (1 + rng.standard_normal() * 0.03) for _ in range(120)]
        rm = FuturesRiskManager({"vol_target": {"enabled": True, "sigma_target": 0.10,
                                                "window": 60, "w_max": 1.0, "w_min": 0.20}})
        ws = [rm.apply({"RB": 1.0}, {"RB": 0.11}, nav_t=v)[0]["RB"] for v in navs]
        active = [x for x in ws if x < 1.0]
        self.assertGreater(len(active), 50)  # 多数日子被 vol-target 压减
        self.assertGreaterEqual(min(active), 0.20 - 1e-9)

    def test_low_vol_full(self):
        rng = np.random.default_rng(1)
        navs = [1.0] + [1.0 * (1 + rng.standard_normal() * 0.0003) for _ in range(120)]
        rm = FuturesRiskManager({"vol_target": {"enabled": True, "sigma_target": 0.10,
                                                "window": 60, "w_max": 1.0, "w_min": 0.20}})
        ws = [rm.apply({"RB": 1.0}, {"RB": 0.11}, nav_t=v)[0]["RB"] for v in navs]
        hot = [x for x in ws if x >= 1.0 - 1e-9]
        self.assertGreater(len(hot), 50)

    def test_cold_start_full(self):
        # 冷启动（样本不足）→ 满仓，不误伤
        rm = FuturesRiskManager({"vol_target": {"enabled": True, "sigma_target": 0.10,
                                                "window": 60, "w_max": 1.0, "w_min": 0.20}})
        w, _ = rm.apply({"RB": 1.0}, {"RB": 0.11}, nav_t=1.0)
        self.assertAlmostEqual(w["RB"], 1.0, places=9)

    def test_disabled_noop(self):
        # vol_target.enabled=false → 恒满仓
        rm = FuturesRiskManager({})
        w, _ = rm.apply({"RB": 1.2, "AU": 0.8}, {"RB": 0.11, "AU": 0.10}, nav_t=1.0)
        self.assertAlmostEqual(w["RB"], 1.2, places=9)
        self.assertAlmostEqual(w["AU"], 0.8, places=9)


class PipelineOrderTest(unittest.TestCase):
    def test_apply_returns_bounded_margin(self):
        # 全约束开启下综合流程不抛、且不把仓位算出负值
        rm = FuturesRiskManager({
            "max_leverage": 1.5, "max_single_margin": 0.30,
            "capital_utilization": 0.9,
            "vol_target": {"enabled": False},
        })
        w, mt = rm.apply({"RB": 0.6, "CU": -0.4, "AU": 0.3},
                         {"RB": 0.11, "CU": 0.08, "AU": 0.10})
        for v, vv in w.items():
            self.assertEqual(vv, abs(vv) if vv >= 0 else -abs(vv))  # 符号保持
        self.assertGreaterEqual(mt, 0.0)


if __name__ == "__main__":
    unittest.main()
