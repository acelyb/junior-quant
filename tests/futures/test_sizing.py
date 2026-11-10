"""期货头寸换算层单元测试：pos_size / apply_margin_budget / futures_trade_cost。

守护重点（信号 → 实际手数的换算正确性，最易超资金预算或算错名义的桩子）：
  1. pos_size：手数=floor(|w|·nav/(price·mult)) 向下取整，名义/保证金手算核实
  2. floor 取整 + <1 手归零（平仓，不产出 margin）
  3. multiplier 兜底：缺省用 default_multiplier / 都不给 → 跳过该品种
  4. apply_margin_budget：超预算等比缩仓保比例、无 cap no-op
  5. futures_trade_cost：多空换向 |+0.5→−0.5| 计 1.0 / 建仓 / 清仓 / 零成本

不依赖网络：全部手造权重/价格/保证金比例，纯函数可复现。

运行：python -m unittest tests.futures.test_sizing
"""
from __future__ import annotations

import unittest

from jq_framework.portfolio.futures_sizing import (
    apply_margin_budget,
    futures_trade_cost,
    margin_used,
    pos_size,
)


NAV = 1_000_000


class PosSizeTest(unittest.TestCase):
    def test_hand_checked_long(self):
        # w=0.5, price=4000, mult=10, margin=0.11
        # 名义/手=40000；手数=floor(0.5*1e6/40000)=12；名义=480000；保证金=52800
        out = pos_size({"RB": 0.5}, NAV, {"RB": 4000.0},
                       {"RB": 10.0}, {"RB": 0.11})
        rb = out["RB"]
        self.assertEqual(rb["lots"], 12)
        self.assertEqual(rb["direction"], 1)
        self.assertAlmostEqual(rb["notional"], 480000, places=6)
        self.assertAlmostEqual(rb["margin"], 52800, places=6)

    def test_short_direction(self):
        # 负权重 → direction=-1，手数/名义与同幅多头一致
        out = pos_size({"CU": -0.5}, NAV, {"CU": 70000.0},
                       {"CU": 5.0}, {"CU": 0.08})
        cu = out["CU"]
        self.assertEqual(cu["direction"], -1)
        self.assertEqual(cu["lots"], 1)  # 0.5*1e6/(70000*5)=1.428 → 1

    def test_floor_and_sub_lot_zero(self):
        # w=0.05 → 0.05*1e6/40000=1.25 → 1 手；w=0.001 → 0 归零
        out2 = pos_size({"RB": 0.05}, NAV, {"RB": 4000.0},
                        {"RB": 10.0}, {"RB": 0.11})
        self.assertEqual(out2["RB"]["lots"], 1)
        out3 = pos_size({"RB": 0.001}, NAV, {"RB": 4000.0},
                        {"RB": 10.0}, {"RB": 0.11})
        self.assertEqual(out3["RB"]["lots"], 0)
        self.assertEqual(out3["RB"]["direction"], 0)
        self.assertEqual(out3["RB"]["margin"], 0.0)  # <1 手归零不产出保证金

    def test_multiplier_fallback(self):
        # mult 缺失用 default_multiplier=5 生效
        out = pos_size({"CU": 0.5}, NAV, {"CU": 70000.0}, {},
                       {"CU": 0.08}, default_multiplier=5)
        self.assertEqual(out["CU"]["lots"], 1)  # 1.428→1
        # 都不给（无 mult 无 default）→ 跳过该品种
        out2 = pos_size({"ZN": 0.5}, NAV, {"ZN": 100.0}, {}, {"ZN": 0.1})
        self.assertNotIn("ZN", out2)

    def test_zero_weight_skipped(self):
        out = pos_size({"RB": 0.0}, NAV, {"RB": 4000.0}, {"RB": 10.0}, {"RB": 0.11})
        self.assertNotIn("RB", out)  # w==0 直接跳过

    def test_bad_price_skipped(self):
        out = pos_size({"RB": 0.5}, NAV, {"RB": 0.0}, {"RB": 10.0}, {"RB": 0.11})
        self.assertNotIn("RB", out)  # price<=0 跳过


class MarginBudgetTest(unittest.TestCase):
    def test_scales_ratio_preserved(self):
        # used=0.6+0.6=1.2 > 0.9 → scale=0.75，比例 1:1 保持
        ws = apply_margin_budget({"RB": 2.0, "AU": 2.0},
                                 {"RB": 0.3, "AU": 0.3}, cap_util=0.9)
        self.assertAlmostEqual(margin_used(ws, {"RB": 0.3, "AU": 0.3}), 0.9, places=9)
        self.assertAlmostEqual(ws["RB"] / ws["AU"], 1.0, places=9)

    def test_noop_without_cap(self):
        self.assertEqual(apply_margin_budget({"RB": 0.5}, {"RB": 0.11}, None),
                         {"RB": 0.5})
        self.assertEqual(apply_margin_budget({"RB": 0.5}, {"RB": 0.11}, 0),
                         {"RB": 0.5})  # cap<=0 no-op
        self.assertEqual(apply_margin_budget({}, {"RB": 0.11}, 0.9), {})

    def test_under_budget_unchanged(self):
        w = {"RB": 0.5}
        self.assertEqual(apply_margin_budget(w, {"RB": 0.11}, cap_util=0.9), w)


class TradeCostTest(unittest.TestCase):
    def test_flip_average(self):
        # 多空换向 |+0.5→−0.5| 计 rebal=1.0, buy=0(prev非0), sell=0(target非0)
        # cost = 1.0*0.0003/2 = 0.00015
        cost = futures_trade_cost({"RB": 0.5}, {"RB": -0.5}, 0.0001, 0.0002)
        self.assertAlmostEqual(cost, 0.00015, places=12)

    def test_open_position(self):
        # 建仓：prev 空→target 0.5：buy=0.5, sell=0, rebal=0.5 → (1.0)*0.0003/2
        cost = futures_trade_cost({}, {"RB": 0.5}, 0.0001, 0.0002)
        self.assertAlmostEqual(cost, 0.00015, places=12)

    def test_zero_rate(self):
        self.assertEqual(futures_trade_cost({"RB": 0.5}, {"RB": -0.5}, 0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
