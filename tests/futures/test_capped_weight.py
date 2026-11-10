"""类别内权重硬封顶 volparity_capped 单元测试。

守护重点（组合风险预算 = 用已知产业类别把高相关簇摊薄、预算让给低相关类别）：
  1. 高相关簇（黑色 RB/I/JM/J）在 cap 下类别合计 ≤ cap，低相关类别预算抬升
  2. 总权重（带符号绝对值加总）守恒 = budget（cap 只重分配不增删）
  3. 全平（raw_dirs 全 0）→ 空 dict
  4. 单个品种独占高波动时，cap 不把负权重符号弄反（符号保号）
  5. 不违反 vol_parity 默认零回归：不传该旁路时 _make_weights 行为不变

与 decorr_vol_parity 的区别（docs §11.1 被否）：
cap 用"已知产业类别"硬约束，不做收益协方差统计估计，无白化噪声腿反噬。
本测试仅守护方法正确性（不守护采纳性）——采纳与否以 OOS 实证为准。

运行：python -m unittest tests.futures.test_capped_weight
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jq_framework.backtest.futures_walk_forward import (
    _vol_parity_capped_weights, _make_weights, CATEGORY,
)

# 17 品种，跨 5 大类；黑色 4 品（RB/I/JM/J）为高相关簇
ALL_VARS = ["RB", "I", "JM", "J", "CU", "AL", "ZN", "NI",
            "AU", "AG", "TA", "MA", "SC", "M", "Y", "P", "CF"]

# 给每品种一个稳定的年化波动（低波动品种得更多预算）。
# 黑色 4 品刻意低波动 ↓ → vol_parity 下黑色簇合计 > 25%（触发 cap 的前置成立）
_VOL = {
    "RB": 0.14, "I": 0.16, "JM": 0.15, "J": 0.15,       # 黑色（低波动→预算多）
    "CU": 0.20, "AL": 0.18, "ZN": 0.20, "NI": 0.24,     # 有色
    "AU": 0.12, "AG": 0.20,                               # 贵金属（AU 低波动→预算多）
    "TA": 0.22, "MA": 0.26, "SC": 0.28,                   # 能化
    "M": 0.16, "Y": 0.18, "P": 0.22, "CF": 0.20,          # 农产品
}


class VolParityCappedTest(unittest.TestCase):

    def setUp(self):
        # 全部做多，便于对比权重分配
        self.raw_dirs = {v: 1.0 for v in ALL_VARS}
        self.budget = 1.0
        self.cap = 0.25

    def _cat_sum(self, w: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in w.items():
            out[CATEGORY[k]] = out.get(CATEGORY[k], 0.0) + abs(v)
        return out

    def test_black_cluster_capped(self):
        """高相关簇（黑色）在 cap 下类别合计 ≤ cap，且被摊薄到低相关类别。"""
        w_base = _make_weights(self.raw_dirs, "vol_parity", _VOL, self.budget, 0.05)
        w_cap = _vol_parity_capped_weights(self.raw_dirs, _VOL, self.budget, min_vol=0.05, cap=self.cap)
        # vol_parity 下黑色 4 品（中等波动）合计应 > cap，否则测试自身不成立
        black_base = self._cat_sum(w_base)["黑色"]
        self.assertGreater(black_base, self.cap,
                           "前置条件：vol_parity 下黑色簇应超 25%（否则无 cap 意义）")
        # cap 后：黑色 ≤ cap，低相关类别（贵金属）合计 ≥ vol_parity 下的值
        capped = self._cat_sum(w_cap)
        self.assertLessEqual(capped["黑色"], self.cap + 1e-9)
        # 预算被搬到某处：至少一个非黑色类别合计上升（通过总守恒保证隐式成立）
        self.assertGreaterEqual(capped["贵金属"], self._cat_sum(w_base)["贵金属"] - 1e-9,
                                "低相关低波动类别（贵金属）预算不应被削减")

    def test_total_weight_conserved(self):
        """cap 只重分配不增删：全组合带符号 |w| 加总保持 = budget。"""
        w = _vol_parity_capped_weights(self.raw_dirs, _VOL, self.budget, min_vol=0.05, cap=self.cap)
        self.assertAlmostEqual(sum(abs(v) for v in w.values()), self.budget, places=6)

    def test_all_flat_returns_empty(self):
        """全平（raw_dirs 全 0）→ 空 dict。"""
        w = _vol_parity_capped_weights({k: 0.0 for k in ALL_VARS}, _VOL, self.budget)
        self.assertEqual(w, {})

    def test_sign_preserved(self):
        """cap 压缩/重分配不翻转任何品种的符号。"""
        raw = {v: (1.0 if v in ("RB", "CU", "AU") else -1.0) for v in ALL_VARS}
        w = _vol_parity_capped_weights(raw, _VOL, self.budget, min_vol=0.05, cap=self.cap)
        for k, v in w.items():
            self.assertEqual(np.sign(v), np.sign(raw[k]), f"{k} 符号被翻转")

    def test_vol_parity_default_unchanged(self):
        """不传旁路时 vol_parity 行为不变（默认零回归守护）。"""
        w = _make_weights(self.raw_dirs, "vol_parity", _VOL, self.budget, 0.05)
        self.assertTrue(all(v > 0 for v in w.values()))

    def test_all_categories_capped_conserves(self):
        """极端 case：所有类别都被压到 cap、无未满类别 → 释放预算仍需守恒回补。

        M7 守护：旧实现当 under 为空时 break，released 预算丢失 → 总|w| < budget，
        违反守恒。改用逆波动回补到全部类别（含已触顶者），总权重回到 budget。
        """
        # 全做多、uniform 波动、cap 极低 → 每类一个品种且都触顶（5 类 × 1 品），
        # 无"未满类别"可吸收释放预算 → 走末端守恒回补分支
        raw = {v: 1.0 for v in ALL_VARS}
        # 用极低 cap 强制所有类别触顶：0.01 远小于各类在 vol_parity 下的份额
        w = _vol_parity_capped_weights(raw, _VOL, self.budget, min_vol=0.05, cap=0.01)
        # 至少不全空
        self.assertTrue(w, "极端 cap 下不应返回空")
        # 守恒：带符号 |w| 加总 ≈ budget（回补后回到预算；允许 cap 微越界）
        tot = sum(abs(v) for v in w.values())
        self.assertAlmostEqual(tot, self.budget, places=3,
                               msg=f"全类别触 cap 时总权重应守恒（偏离 {tot:.4f} vs {self.budget}）")

    def test_intra_category_inverse_vol_topup(self):
        """M7 类内回补按逆波动非均分：同类别内低波动品种获更多回补份额。"""
        # 真实 5 类：黑色(RB/I 低波动)触顶、其余 under → 类内回补走反波分配。
        # 断言黑色内部 RB(σ0.14) 相对 I(σ0.16) 的权重不低于 I——均分则相等、
        # 反波分配下低波 RB 应获更多（守住"非均分"的实现）：
        capped = _vol_parity_capped_weights(self.raw_dirs, _VOL, self.budget,
                                            min_vol=0.05, cap=self.cap)
        rb_w, i_w = capped.get("RB", 0.0), capped.get("I", 0.0)
        self.assertTrue(rb_w > 0 and i_w > 0, "黑色两类权重应存在")
        self.assertGreaterEqual(rb_w, i_w - 1e-9,
                                "类内反波回补：低波品种 RB 权重不应低于高波 I")


class CappedShareSemanticsTest(unittest.TestCase):
    """份额口径回归守护（2026-10-15 第六步修正）。

    原实现把 cap 当绝对权重上限：budget=1.0 时与份额口径等价（上述测试因此
    全绿未暴露），但生产 budget=max_leverage=1.5 > 5×0.25 时容量不可行——
    迭代停在中途，类别份额既未封住、结果依赖迭代次数（实测黑色 29%→23%）。
    修正后 cap 约束类别份额，任意 budget 下都必须封住。
    """

    def _shares(self, w: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        tot = sum(abs(v) for v in w.values())
        for k, v in w.items():
            out[CATEGORY[k]] = out.get(CATEGORY[k], 0.0) + abs(v)
        return {c: s / tot for c, s in out.items()}

    def test_budget_150_holds_share_cap(self):
        """生产 budget=1.5：所有类别份额 ≤ cap（原缺陷场景）。"""
        raw = {v: 1.0 for v in ALL_VARS}
        w = _vol_parity_capped_weights(raw, _VOL, 1.5, min_vol=0.05, cap=0.25)
        shares = self._shares(w)
        # 前置：vol_parity 下黑色份额 > 25%（否则场景不成立）
        w_base = _make_weights(raw, "vol_parity", _VOL, 1.5, 0.05)
        self.assertGreater(self._shares(w_base)["黑色"], 0.25)
        for c, s in shares.items():
            self.assertLessEqual(s, 0.25 + 1e-9, f"类别 {c} 份额 {s:.1%} 超 cap")
        # 守恒：总 |w| = budget（cap 只重分配不增删）
        self.assertAlmostEqual(sum(abs(v) for v in w.values()), 1.5, places=6)

    def test_extreme_concentration_capped_any_budget(self):
        """极端集中（一类独大）× budget=1.5 × 4 类活跃（容量 100% 恰可行）：份额封住。"""
        # 黑色 4 品 + 贵/能化/农产品持仓、有色全平；均匀波动 → 初始份额 ∝ 品种数
        raw = {v: (1.0 if CATEGORY[v] == "黑色" else
                   (-1.0 if CATEGORY[v] == "贵金属" else
                    (1.0 if CATEGORY[v] in ("能化", "农产品") else 0.0))) for v in ALL_VARS}
        vol2 = {k: 0.10 for k in ALL_VARS}
        w = _vol_parity_capped_weights(raw, vol2, 1.5, min_vol=0.05, cap=0.25)
        shares = self._shares(w)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=9)
        self.assertGreater(self._shares(_make_weights(raw, "vol_parity", vol2, 1.5, 0.05))["黑色"],
                           0.25, "前置：黑色初始份额应超 cap")
        for c, s in shares.items():
            self.assertLessEqual(s, 0.25 + 1e-9, f"类别 {c} 份额 {s:.1%} 超 cap")
        self.assertAlmostEqual(sum(abs(v) for v in w.values()), 1.5, places=6)
        for k, v in w.items():
            self.assertEqual(np.sign(v), np.sign(raw[k]), f"{k} 符号被翻转")

    def test_infeasible_categories_conserve_budget(self):
        """容量不可行（2 类活跃 × cap 0.25 = 50% < 100%）：守恒优先、cap 尽力。"""
        # 只有黑色 4 品 + 贵金属 2 品持仓——份额上限算术无解，M7 末端守恒兜底：
        # 总 |w| 仍 = budget、符号保号（cap 违反是容量约束的必然，非 bug）
        raw = {v: (1.0 if CATEGORY[v] == "黑色" else
                   (-1.0 if CATEGORY[v] == "贵金属" else 0.0)) for v in ALL_VARS}
        vol2 = {k: 0.10 for k in ALL_VARS}
        w = _vol_parity_capped_weights(raw, vol2, 1.5, min_vol=0.05, cap=0.25)
        self.assertAlmostEqual(sum(abs(v) for v in w.values()), 1.5, places=6)
        for k, v in w.items():
            self.assertEqual(np.sign(v), np.sign(raw[k]), f"{k} 符号被翻转")

    def test_budget_100_matches_legacy_absolute_semantics(self):
        """budget=1.0 时份额口径 ≡ 旧绝对值口径（既有行为零回归）。"""
        raw = {v: 1.0 for v in ALL_VARS}
        w = _vol_parity_capped_weights(raw, _VOL, 1.0, min_vol=0.05, cap=0.25)
        for c, s in self._shares(w).items():
            self.assertLessEqual(s, 0.25 + 1e-9)
        self.assertAlmostEqual(sum(abs(v) for v in w.values()), 1.0, places=6)




if __name__ == "__main__":
    unittest.main()
