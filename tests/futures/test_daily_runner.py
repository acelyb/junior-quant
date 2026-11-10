"""每日信号 runner 纯逻辑单元测试：白名单收敛 + 符号换算 + 手数渲染。

守护重点：
  1. 期货白名单收敛：tsmom_carry_cta 出货，reversal/triple_factor 被否策略被过滤
  2. futures_strategies 缺省/None 时 → 不设白名单，全部期货策略出货（向后兼容）
  3. A 股 stock_strategies 留空 = 默认全部 stock 策略出货（零回归）
  4. A 股白名单非空时按白名单收敛
  5. format_report 手数渲染（审计 #19）：期货记录 market=策略 type（"cta"），
     渲染按 lots 键出（原判 r["market"]=="futures" 恒假 → 手数算好不渲染）
  6. carry 覆盖 fail-loud（第三步）：_carry_status 四态分类 + 覆盖率聚合 + 渲染
  _futures_variety：新浪主连 'RB0' → 品种 'RB'（去尾 0），普通符号原样

不依赖网络：只打纯函数 _is_strategy_allowed / _futures_variety / format_report /
_carry_status / _carry_coverage_line。

运行：python -m unittest tests.futures.test_daily_runner
"""
from __future__ import annotations

import unittest

import pandas as pd

from jq_framework.signals.daily_runner import (
    _carry_coverage_line,
    _carry_status,
    _futures_variety,
    _is_strategy_allowed,
    format_report,
)


class FuturesWhitelistTest(unittest.TestCase):
    def test_allows_whitelisted(self):
        # tsmom_carry_cta 在白名单 → 放行
        ds = {"futures_strategies": ["tsmom_carry_cta"]}
        self.assertTrue(_is_strategy_allowed("futures", "tsmom_carry_cta", ds))

    def test_filters_rejected_strategies(self):
        # 被否策略(reversal/triple_factor)注册但不进每日信号（治理收敛）
        ds = {"futures_strategies": ["tsmom_carry_cta"]}
        self.assertFalse(_is_strategy_allowed("futures", "reversal_cta", ds))
        self.assertFalse(_is_strategy_allowed("futures", "triple_factor_cta", ds))

    def test_empty_futures_list_filters_all(self):
        # 空白名单 = 无任何期货策略出货（显式收敛到无）
        ds = {"futures_strategies": []}
        self.assertFalse(_is_strategy_allowed("futures", "tsmom_carry_cta", ds))

    def test_missing_futures_strategies_defaults_all(self):
        # key 缺省 → 不设白名单，全部期货策略出货（向后兼容）
        self.assertTrue(_is_strategy_allowed("futures", "tsmom_carry_cta", {}))
        self.assertTrue(_is_strategy_allowed("futures", "reversal_cta", {}))


class StockWhitelistTest(unittest.TestCase):
    def test_empty_stock_list_allows_all(self):
        # A 股 stock_strategies 留空 = 默认全部 stock 策略出货（零回归）
        ds = {"stock_strategies": []}
        self.assertTrue(_is_strategy_allowed("stock", "dual_ma_stock", ds))
        self.assertTrue(_is_strategy_allowed("stock", "any_strategy", ds))

    def test_nonempty_converges(self):
        ds = {"stock_strategies": ["dual_ma_stock"]}
        self.assertTrue(_is_strategy_allowed("stock", "dual_ma_stock", ds))
        self.assertFalse(_is_strategy_allowed("stock", "other_strategy", ds))


class FuturesVarietyTest(unittest.TestCase):
    def test_strips_trailing_zero(self):
        self.assertEqual(_futures_variety("RB0"), "RB")
        self.assertEqual(_futures_variety("SC0"), "SC")

    def test_keeps_plain_symbol(self):
        self.assertEqual(_futures_variety("RB"), "RB")
        self.assertEqual(_futures_variety(""), "")


def _cta_rec(**over) -> dict:
    """合成一条期货每日信号记录（market=策略 type "cta"，含 _pos_fields 注入的四键）。"""
    rec = {"strategy": "tsmom_carry_cta", "market": "cta", "symbol": "RB0",
           "direction": "long", "target_pos": 1.0, "price": 3500.0,
           "note": "TSMOM: 250日累计收益+12.3% > 0 → 多",
           "latest_date": "2026-10-13", "factor": 0.6,
           "nav": 1_000_000.0, "lots": 3, "notional": 105_000.0, "margin": 10_500.0}
    rec.update(over)
    return rec


class RenderLotsTest(unittest.TestCase):
    """审计 #19：手数/名义/保证金算好了必须渲染出来（期货线"可执行手数清单"定位）。"""

    def test_cta_record_renders_lots(self):
        # market="cta"（策略 type）+ lots → 渲染出手数行（原判 =="futures" 恒假）
        report = format_report([_cta_rec()])
        self.assertIn("手数", report)
        self.assertIn("3 手", report)
        self.assertIn("名义", report)
        self.assertIn("保证金", report)

    def test_stock_record_without_lots_renders_no_lots_line(self):
        # A 股记录（无 lots 键）不出手数行——零回归
        stock = {"strategy": "dual_ma_stock", "market": "stock", "symbol": "000021",
                 "direction": "long", "target_pos": 0.1, "price": 10.5,
                 "note": "金叉", "latest_date": "2026-10-13"}
        report = format_report([stock])
        self.assertNotIn("手数", report)
        self.assertNotIn("名义", report)

    def test_error_record_renders_no_lots_line(self):
        # 错误记录（_err_record 形态：无 lots 键）不出手数行，不与告警语义混淆
        err = {"strategy": "tsmom_carry_cta", "market": "cta", "symbol": "RB0",
               "direction": "error", "target_pos": 0, "price": None,
               "note": "出错: 无数据", "latest_date": "-"}
        report = format_report([err])
        self.assertNotIn("手数", report)

    def test_zero_lots_still_rendered(self):
        # _pos_fields 失败兜底（lots=0）也渲染——失败可见优于静默消失
        report = format_report([_cta_rec(lots=0, notional=0.0, margin=0.0)])
        self.assertIn("手数", report)
        self.assertIn("0 手", report)


class RenderContractSourceTest(unittest.TestCase):
    """第二步口径 fail-loud：built 模式记录渲染口径行，兜底品种带 ⚠️。

    contract 键仅 built 模式注入（X0 模式/A 股/错误记录均无此键 → 不渲染，零回归）。
    """

    def test_built_record_renders_plain(self):
        report = format_report([_cta_rec(contract="built")])
        self.assertIn("口径", report)
        self.assertIn("built 复权", report)
        self.assertNotIn("⚠️", report)

    def test_fallback_record_renders_warning(self):
        report = format_report([_cta_rec(contract="X0-fallback")])
        self.assertIn("口径", report)
        self.assertIn("⚠️ X0 兜底（含换月跳空）", report)

    def test_record_without_key_renders_no_line(self):
        # X0 模式（无 contract 键）与 A 股/错误记录都不出口径行——零回归
        report = format_report([_cta_rec()])
        self.assertNotIn("口径", report)


def _carry_df(start: str, end: str) -> pd.DataFrame:
    """合成展期收益率序列帧（date/roll_yield 两列，日频）。"""
    d = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": d, "roll_yield": 0.001})


class CarryStatusTest(unittest.TestCase):
    """第三步 carry fail-loud：_carry_status 四态分类（纯函数，网络无关）。"""

    def test_no_source(self):
        # 序列空 → 无源（腿=0，组合该品种实为 0.5×TSMOM 半信号）
        st = _carry_status(None, pd.Timestamp("2026-10-14"))
        self.assertEqual(st["status"], "无源")
        self.assertFalse(st["active"])
        self.assertIsNone(st["start"])
        self.assertIsNone(st["lag_days"])
        st2 = _carry_status(pd.DataFrame(columns=["date", "roll_yield"]),
                            pd.Timestamp("2026-10-14"))
        self.assertEqual(st2["status"], "无源")
        self.assertFalse(st2["active"])

    def test_full_since_2018(self):
        # 起点 2018（≤ 分界 2021-01-01）→ 完整(2018起)，active
        st = _carry_status(_carry_df("2018-01-02", "2026-10-13"),
                           pd.Timestamp("2026-10-14"))
        self.assertEqual(st["status"], "完整(2018起)")
        self.assertTrue(st["active"])
        self.assertEqual(st["lag_days"], 1)

    def test_short_history_since_2021_09(self):
        # 起点 2021-09（> 分界）→ 短史(2021-09起)，active
        st = _carry_status(_carry_df("2021-09-22", "2026-10-13"),
                           pd.Timestamp("2026-10-14"))
        self.assertEqual(st["status"], "短史(2021-09起)")
        self.assertTrue(st["active"])

    def test_stale_series_inactive(self):
        # 末值滞后价格 > 14 天 → 过期（腿信号吃旧值），active=False
        st = _carry_status(_carry_df("2018-01-02", "2026-09-01"),
                           pd.Timestamp("2026-10-14"))
        self.assertTrue(st["status"].startswith("过期"))
        self.assertIn("滞后", st["status"])
        self.assertEqual(st["lag_days"], 43)
        self.assertFalse(st["active"])

    def test_lag_within_tolerance_stays_active(self):
        # 滞后 ≤ 14 天（容差内）→ 仍按起点分组、active
        st = _carry_status(_carry_df("2018-01-02", "2026-10-01"),
                           pd.Timestamp("2026-10-14"))
        self.assertEqual(st["status"], "完整(2018起)")
        self.assertTrue(st["active"])
        self.assertEqual(st["lag_days"], 13)


class CarryCoverageLineTest(unittest.TestCase):
    """第三步 carry fail-loud：_carry_coverage_line 聚合（无 carry 记录→None）。"""

    def test_no_carry_records_returns_none(self):
        # 全部记录无 carry 键（A 股/非 carry 策略）→ 不输出——零回归
        self.assertIsNone(_carry_coverage_line([_cta_rec()]))

    def test_empty_records_returns_none(self):
        self.assertIsNone(_carry_coverage_line([]))

    def test_mixed_groups_aggregated(self):
        # 三组混合：完整 1 / 短史 1 / 无源 1 → 2/3 活跃
        recs = [
            _cta_rec(carry={"status": "完整(2018起)", "active": True}),
            _cta_rec(carry={"status": "短史(2021-09起)", "active": True}),
            _cta_rec(carry={"status": "无源", "active": False}),
        ]
        line = _carry_coverage_line(recs)
        self.assertIn("carry 覆盖: 2/3 品种活跃", line)
        self.assertIn("完整 1", line)
        self.assertIn("短史 1", line)
        self.assertIn("无源 1", line)

    def test_stale_grouped_as_expired(self):
        # 过期单列一组、不进活跃分子
        recs = [
            _cta_rec(carry={"status": "完整(2018起)", "active": True}),
            _cta_rec(carry={"status": "过期(滞后30天)", "active": False}),
        ]
        line = _carry_coverage_line(recs)
        self.assertIn("carry 覆盖: 1/2 品种活跃", line)
        self.assertIn("过期 1", line)


class RenderCarryStatusTest(unittest.TestCase):
    """第三步 carry fail-loud：format_report 渲染 carry 行（有键才渲染）。"""

    def test_no_source_renders_half_signal_warning(self):
        # 无源 → 半信号提示（静默降级必须可见）
        report = format_report([_cta_rec(
            carry={"status": "无源", "active": False})])
        self.assertIn("carry", report)
        self.assertIn("无源 → carry腿=0（实为0.5×TSMOM 半信号）", report)

    def test_sourced_renders_status_only(self):
        report = format_report([_cta_rec(
            carry={"status": "完整(2018起)", "active": True})])
        self.assertIn("carry  : 完整(2018起)", report)
        self.assertNotIn("半信号", report)

    def test_no_carry_key_renders_no_line(self):
        # 非 carry 策略/A 股/错误记录无 carry 键 → 不渲染（策略名含 carry 子串不算）
        report = format_report([_cta_rec()])
        self.assertNotIn("carry  :", report)
        self.assertNotIn("半信号", report)
        self.assertNotIn("carry 覆盖", report)

    def test_report_ends_with_coverage_summary(self):
        # 汇总段末尾渲染覆盖率行（有 carry 记录才渲染）
        recs = [
            _cta_rec(carry={"status": "完整(2018起)", "active": True}),
            _cta_rec(carry={"status": "无源", "active": False}),
        ]
        report = format_report(recs)
        self.assertIn("carry 覆盖: 1/2 品种活跃", report)


if __name__ == "__main__":
    unittest.main()
