"""tests/futures/test_breakout.py — 中短线 Phase 1d 突破状态机的守护。

口径（模块 docstring 的边界逐条）：入场=突破 entry_n 日高/低（阈值只含 t-1
及以前）、显式报向从不出 hold、反手(①)优先于通道出场(②)优先于时间止损(③)、
最长 max_days 个报向日、**时间止损离场后同向全新突破可重入**（事件重入——
TimeStopWrapper 同向不追规则做不到的核心属性）、每品种状态独立。
"""
from __future__ import annotations

import unittest

import pandas as pd

from jq_alpha.strategies.cta.breakout import BreakoutStrategy

# 单测用小参数（默认 entry_n=20 太长）；比例关系与生产一致：exit_n < entry_n
P = {"entry_n": 5, "exit_n": 3, "max_days": 4, "price": "settle", "target_pos": 1.0}


def _bars(prices, holds=None) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=len(prices)),
                       "settle": [float(p) for p in prices],
                       "close": [float(p) for p in prices]})
    if holds is not None:
        df["hold"] = [float(h) for h in holds]
    return df


def _run(prices, params=P, var="RB", holds=None) -> list[str]:
    """沿价格路径逐日前缀驱动（引擎口径：每品种每日恰好一次 on_bar）。"""
    s = BreakoutStrategy(params=params)
    s.set_active_var(var)
    return [s.on_bar(_bars(prices[:i + 1], holds[:i + 1] if holds else None)).direction
            for i in range(len(prices))]


class BreakoutStateMachineTest(unittest.TestCase):

    def test_insufficient_bars_flat(self):
        # 通道阈值不可计算 → 显式 flat（状态机从不出 hold），不 crash
        self.assertEqual(_run([10.0] * 5), ["flat"] * 5)

    def test_entry_on_upside_break(self):
        # 突破 5 日高（严格 >，恰好等于不触发）→ long
        dirs = _run([10.0] * 5 + [12.0])
        self.assertEqual(dirs, ["flat"] * 5 + ["long"])

    def test_no_entry_when_touching_channel_exactly(self):
        # 因果性/严格性：c == 前 5 日高 → 不入场（阈值不含当日）
        dirs = _run([10.0, 11.0, 12.0, 13.0, 14.0, 14.0])
        self.assertEqual(dirs[-1], "flat")

    def test_entry_on_downside_break(self):
        dirs = _run([10.0] * 5 + [8.0])
        self.assertEqual(dirs[-1], "short")

    def test_hold_days_tick_without_trigger(self):
        # 无出场/反手触发 → 维持多头且时钟走（连续报 long）
        dirs = _run([10.0] * 5 + [12.0, 12.1, 12.2])
        self.assertEqual(dirs, ["flat"] * 5 + ["long", "long", "long"])

    def test_time_stop_exactly_max_days(self):
        # 最长 max_days(4) 个报向日：第 5 个报向日起 flat
        dirs = _run([10.0] * 5 + [12.0] * 8)
        self.assertEqual(dirs, ["flat"] * 5 + ["long"] * 4 + ["flat"] * 4)

    def test_exit_on_short_channel_break(self):
        # ② 跌破 3 日低（但未破 5 日低——深低 8 在 5 日窗内、3 日窗外）→ 平仓 flat
        dirs = _run([7.0, 10.0, 8.0, 10.0, 10.0, 11.0, 9.5])
        self.assertEqual(dirs, ["flat"] * 5 + ["long", "flat"])

    def test_flip_on_opposite_entry_break(self):
        # ① 持多中跌破 5 日低（含②的 3 日低）→ 反手 short（时钟重开）
        dirs = _run([10.0] * 5 + [12.0, 7.0])
        self.assertEqual(dirs, ["flat"] * 5 + ["long", "short"])
        # note 确认时钟重开
        s = BreakoutStrategy(params=P)
        s.set_active_var("RB")
        s.on_bar(_bars([10.0] * 5 + [12.0]))
        sig = s.on_bar(_bars([10.0] * 5 + [12.0, 7.0]))
        self.assertIn("days=1", sig.note)

    def test_reentry_on_fresh_same_direction_break(self):
        # 核心属性（事件重入）：时间止损离场后，同向**全新突破**须能重入
        # ——TimeStopWrapper 的"同向不追"会吞掉这个事件，本类原生实现不吞。
        dirs = _run([10.0] * 5 + [12.0] * 7 + [13.0])
        # i=5 入场、i=6..8 持仓(days 2..4)、i=9 时间止损、i=10..11 观望(12==12 不触发)、
        # i=12 突破前 5 日高(12) → 重入 long
        self.assertEqual(dirs, ["flat"] * 5 + ["long"] * 4 + ["flat"] * 3 + ["long"])

    def test_short_side_symmetry(self):
        # 空头侧对称：跌破入场 → 突破 3 日高出场（峰 13 在 3 日窗外、12 在 5 日窗内）
        dirs = _run([13.0, 10.0, 12.0, 10.0, 10.0, 9.0, 10.5])
        self.assertEqual(dirs, ["flat"] * 5 + ["short", "flat"])
        dirs = _run([10.0] * 5 + [8.0] * 7 + [7.0])
        self.assertEqual(dirs, ["flat"] * 5 + ["short"] * 4 + ["flat"] * 3 + ["short"])
        dirs = _run([10.0] * 5 + [8.0, 13.0])
        self.assertEqual(dirs[-1], "long")                          # 反手

    def test_per_variety_state_independent(self):
        s = BreakoutStrategy(params=P)
        s.set_active_var("RB")
        s.on_bar(_bars([10.0] * 5 + [12.0]))                        # RB 入场 days=1
        s.on_bar(_bars([10.0] * 5 + [12.0, 12.1]))                  # RB days=2
        s.set_active_var("CU")
        sig = s.on_bar(_bars([10.0] * 5 + [11.0]))                  # CU 独立入场
        self.assertEqual(sig.direction, "long")
        self.assertIn("days=1", sig.note)
        s.set_active_var("RB")
        sig = s.on_bar(_bars([10.0] * 5 + [12.0, 12.1, 12.2]))      # RB 时钟不被 CU 打断
        self.assertIn("days=3", sig.note)

    def test_never_reports_hold(self):
        # 任何状态下都不出 hold（0c 语义债在策略内消化）
        paths = [[10.0] * 5 + [12.0, 7.0, 13.0], [10.0] * 4,
                 [10.0] * 5 + [12.0] * 9, [10.0] * 5 + [8.0] * 9]
        for p in paths:
            for d in _run(p):
                self.assertIn(d, ("long", "short", "flat"))

    def test_factor_legs_empty(self):
        # 有状态：禁止自引用出腿（--attribution 双跳防护）
        self.assertEqual(BreakoutStrategy(params=P).factor_legs, {})


class OiConfirmTest(unittest.TestCase):
    """oi_confirm 增仓确认（2026-10-26 OI/量确认突破探针采纳档）的守护。

    语义：只约束空仓→开仓（反手/出场不受影响）；Δhold/hold[t-1] > 阈值才入场；
    hold 缺失/无效 → 宁漏勿猜不入场；oi_confirm=None（默认）行为与既有逐字一致。
    """

    POI = {**P, "oi_confirm": 0.01}

    def test_default_off_identical_behavior(self):
        # 默认 None：无 hold 列也照常入场（与既有 breakout 逐字一致——零回归）
        dirs = _run([10.0] * 5 + [12.0], params=P)
        self.assertEqual(dirs[-1], "long")

    def test_oi_confirmed_entry(self):
        # 突破日增仓 2% > 1% 阈值 → 入场
        dirs = _run([10.0] * 5 + [12.0], params=self.POI,
                    holds=[100.0] * 5 + [102.0])
        self.assertEqual(dirs[-1], "long")

    def test_oi_unconfirmed_blocks_entry(self):
        # 突破日减仓 → 不入场（flat）；状态保持空仓
        dirs = _run([10.0] * 5 + [12.0, 12.1], params=self.POI,
                    holds=[100.0] * 5 + [98.0, 98.0])
        self.assertEqual(dirs[-1], "flat")
        # 次日仍在通道上方但 Δhold=0 未过阈 → 仍不入（确认按当日增量算）
        dirs2 = _run([10.0] * 5 + [12.0, 13.0], params=self.POI,
                     holds=[100.0] * 5 + [98.0, 98.0])
        self.assertEqual(dirs2[-1], "flat")

    def test_oi_threshold_strictly_greater(self):
        # Δhold 恰等于阈值（1%）→ 不入场（严格 >，与通道突破同严格性口径）
        dirs = _run([10.0] * 5 + [12.0], params=self.POI,
                    holds=[100.0] * 5 + [101.0])
        self.assertEqual(dirs[-1], "flat")

    def test_oi_missing_hold_blocks_entry(self):
        # hold 列缺失 → 确认条件不可算，宁漏勿猜不入场
        dirs = _run([10.0] * 5 + [12.0], params=self.POI, holds=None)
        self.assertEqual(dirs[-1], "flat")

    def test_oi_zero_prev_hold_blocks_entry(self):
        # hold[t-1] ≤ 0（脏数据）→ 不可算不入场
        dirs = _run([10.0] * 5 + [12.0], params=self.POI,
                    holds=[0.0] * 5 + [50.0])
        self.assertEqual(dirs[-1], "flat")

    def test_oi_downside_break_confirmed_short(self):
        # 空头对称：跌破 5 日低 + 增仓 → short
        dirs = _run([10.0] * 5 + [8.0], params=self.POI,
                    holds=[100.0] * 5 + [103.0])
        self.assertEqual(dirs[-1], "short")

    def test_oi_exit_not_gated(self):
        # 出场不受确认约束：持仓中跌破短窗通道照常出场（即便当日减仓）
        # 路径：flat×5 → 入场(增仓确认过) → 盘整2日 → 跌破 3 日低（当日减仓）→ 仍出场
        dirs = _run([10.0] * 5 + [12.0, 12.1, 11.8, 11.5], params=self.POI,
                    holds=[100.0, 100.0, 100.0, 100.0, 100.0,
                           102.0,   # 入场日增仓
                           90.0,    # 出场候选日减仓——不受确认约束
                           90.0, 90.0])
        self.assertEqual(dirs, ["flat"] * 5 + ["long"] * 3 + ["flat"])

    def test_oi_reversal_not_gated(self):
        # 反手不受确认约束：持多期间跌破 5 日低 → 立即反手 short（即便减仓）
        dirs = _run([10.0] * 5 + [12.0, 7.0], params=self.POI,
                    holds=[100.0] * 5 + [102.0, 90.0])
        self.assertEqual(dirs[-1], "short")

    def test_oi_after_time_stop_reentry_gated(self):
        # 时间止损离场后的同向全新突破仍受确认约束（重入=空仓→开仓）
        prices = [10.0] * 5 + [12.0] * 5 + [12.5]      # flat×5, long×4, flat(到期), 次日再突破
        holds = [100.0] * 5 + [102.0] + [100.0] * 4 + [99.0]   # 再突破日减仓 → 拦下
        dirs = _run(prices, params=self.POI, holds=holds)
        self.assertEqual(dirs, ["flat"] * 5 + ["long"] * 4 + ["flat", "flat"])
