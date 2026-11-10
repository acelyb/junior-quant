"""季节性/日历因子（Seasonal）策略层单元测试。

守护重点（区别于价格衍生因子的本质：只看日历位置，不看价格路径形状）：
  - 历史同月均值符号 → long/short；样本不足 / 近零中性带 → hold
  - 无前视铁律：**进行中的当月不算自己的观测**（当月暴跌不影响过去同月定的方向）
  - 目标月 = 最后一根 bar 所在月（换月切换语义）
  - 月度收益跨月缺口 → NaN（不把"两月收益"冒充单月季节性）
  - settle 优先于 close 的价格口径（与 tsmom 一致）
  - 纯函数 month_end_price / month_returns 的分组与缺口语义

不依赖网络 / 不碰缓存：本地构造多年度 bars DataFrame，纯函数可复现。

运行：python -m unittest tests.futures.test_seasonal
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jq_alpha.strategies.cta.seasonal import (
    SeasonalStrategy,
    month_end_price,
    month_returns,
)


def monthly_bars(month_rets: dict[tuple[int, int], float], days_per_month: int = 3,
                 last_month_days: int | None = None) -> pd.DataFrame:
    """按 {(年, 月): 月度收益} 构造 bars：月内 days_per_month 根 bar 线性走到月末价。

    月末价 = 上月末价 × (1 + r) 精确成立 → month_returns 复算出的即 r 本身。
    last_month_days 可让最后一个月残缺（模拟"进行中的当月"只有几根 bar）。
    """
    keys = sorted(month_rets.keys())
    dates: list = []
    px: list[float] = []
    level = 100.0
    for i, k in enumerate(keys):
        r = month_rets[k]
        end_px = level * (1.0 + r)
        n = days_per_month if (last_month_days is None or i < len(keys) - 1) \
            else last_month_days
        dts = pd.bdate_range(f"{k[0]}-{k[1]:02d}-01", periods=n)
        for j in range(n):
            dates.append(dts[j])
            px.append(level + (end_px - level) * (j + 1) / n)
        level = end_px
    return pd.DataFrame({"date": dates, "close": px, "settle": px})


def full_years(start_year: int, end_month_key: tuple[int, int],
               oct_ret: float = 0.0, other_ret: float = 0.0,
               nov_ret: float | None = None) -> dict[tuple[int, int], float]:
    """start_year 至 end_month_key 的逐月收益：10 月固定 oct_ret，11 月固定 nov_ret
    （缺省 = other_ret），其余月 other_ret。"""
    out: dict[tuple[int, int], float] = {}
    y, m = start_year, 1
    while (y, m) <= end_month_key:
        r = oct_ret if m == 10 else (nov_ret if (m == 11 and nov_ret is not None) else other_ret)
        out[(y, m)] = r
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


class SeasonalHelpersTest(unittest.TestCase):
    """纯函数：月末分组 + 月度收益（含跨月缺口语义）。"""

    def test_month_end_price_takes_last_bar_per_month(self):
        idx = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08",
                              "2020-02-03", "2020-02-04"])
        me = month_end_price(pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx))
        self.assertEqual(me[(2020, 1)], 12.0)
        self.assertEqual(me[(2020, 2)], 14.0)

    def test_month_returns_basic(self):
        me = pd.Series([100.0, 110.0, 99.0], index=pd.MultiIndex.from_tuples(
            [(2020, 1), (2020, 2), (2020, 3)]))
        r = month_returns(me)
        self.assertTrue(np.isnan(r[(2020, 1)]))  # 首月无上月
        self.assertAlmostEqual(r[(2020, 2)], 0.10)
        self.assertAlmostEqual(r[(2020, 3)], 99.0 / 110.0 - 1.0)

    def test_month_returns_gap_month_is_nan(self):
        # 2020-03 整月缺数据 → 04 月收益 NaN（不跨缺口拼"两月收益"）
        me = pd.Series([100.0, 105.0, 115.5, 121.0], index=pd.MultiIndex.from_tuples(
            [(2020, 1), (2020, 2), (2020, 4), (2020, 5)]))
        r = month_returns(me)
        self.assertTrue(np.isnan(r[(2020, 4)]))
        self.assertAlmostEqual(r[(2020, 5)], 121.0 / 115.5 - 1.0)

    def test_month_returns_december_to_january_spans_year(self):
        # 1 月的上月 = 上年 12 月（跨年衔接）
        me = pd.Series([100.0, 120.0, 114.0], index=pd.MultiIndex.from_tuples(
            [(2020, 12), (2021, 1), (2021, 2)]))
        r = month_returns(me)
        self.assertAlmostEqual(r[(2021, 1)], 0.20)
        self.assertAlmostEqual(r[(2021, 2)], 114.0 / 120.0 - 1.0)


class SeasonalStrategyTest(unittest.TestCase):

    def test_hold_when_insufficient_years(self):
        # 2020/2021 两个 10 月观测 < min_years=3 → hold（与 TSMOM 预热期同语义）
        rets = full_years(2020, (2022, 10), oct_ret=0.05)
        bars = monthly_bars(rets)
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(bars)
        self.assertEqual(sig.direction, "hold")
        self.assertIn("无方向", sig.note)

    def test_long_when_month_seasonally_up(self):
        # 2020-2023 四个 10 月各 +5% → 均值 +5% → 10 月做多
        rets = full_years(2020, (2024, 10), oct_ret=0.05)
        bars = monthly_bars(rets)
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(bars)
        self.assertEqual(sig.direction, "long")
        self.assertEqual(sig.target_pos, 1.0)
        self.assertIn("4年均值", sig.note)

    def test_short_when_month_seasonally_down(self):
        rets = full_years(2020, (2024, 10), oct_ret=-0.05)
        bars = monthly_bars(rets)
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(bars)
        self.assertEqual(sig.direction, "short")

    def test_in_progress_month_is_not_own_observation(self):
        # 无前视铁律：当月（进行中）暴跌 −30%，但过去四个 10 月 +5% → 仍做多
        rets = full_years(2020, (2024, 10), oct_ret=0.05)
        rets[(2024, 10)] = -0.30
        bars = monthly_bars(rets, last_month_days=2)  # 当月只有 2 根 bar
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(bars)
        self.assertEqual(sig.direction, "long")

    def test_target_month_follows_last_bar_month(self):
        # 换月切换：最后一根 bar 在 11 月 → 用历史 11 月（+6%→多），不用 10 月（−5%）
        rets = full_years(2020, (2024, 11), oct_ret=-0.05, nov_ret=0.06)
        bars = monthly_bars(rets, last_month_days=2)
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(bars)
        self.assertEqual(sig.direction, "long")
        # 同一段历史，最后 bar 落在 10 月（10 月为进行中）→ 用历史 10 月 → 空
        rets_oct = full_years(2020, (2024, 10), oct_ret=-0.05, nov_ret=0.06)
        sig_oct = SeasonalStrategy(params={"min_years": 3}).on_bar(monthly_bars(rets_oct))
        self.assertEqual(sig_oct.direction, "short")

    def test_hold_in_neutral_band(self):
        # 四个 10 月各 +0.5%，均值 +0.5% < eps=1% → 近零中性带 hold
        rets = full_years(2020, (2024, 10), oct_ret=0.005)
        sig = SeasonalStrategy(params={"min_years": 3, "eps": 0.01}).on_bar(monthly_bars(rets))
        self.assertEqual(sig.direction, "hold")
        self.assertIn("近零", sig.note)

    def test_hold_when_flat_history(self):
        # 全部月份 0% → 均值 0 → hold
        rets = full_years(2020, (2024, 10))
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(monthly_bars(rets))
        self.assertEqual(sig.direction, "hold")

    def test_settle_preferred_over_close(self):
        # settle 路径历史 10 月 +5%、close 路径 −5% → 信号跟 settle（多）
        up = monthly_bars(full_years(2020, (2024, 10), oct_ret=0.05))
        down = monthly_bars(full_years(2020, (2024, 10), oct_ret=-0.05))
        bars = pd.DataFrame({"date": up["date"], "close": down["close"], "settle": up["settle"]})
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(bars)
        self.assertEqual(sig.direction, "long")

    def test_hold_when_no_price_columns(self):
        bars = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
        sig = SeasonalStrategy(params={}).on_bar(bars)
        self.assertEqual(sig.direction, "hold")
        self.assertIn("无可用价格序列", sig.note)

    def test_dirty_settle_rows_filtered_with_dates(self):
        # settle 列含 0/NaN 脏行：连日期一起过滤（价格↔日期对齐是月末分组的前提）
        base = monthly_bars(full_years(2020, (2024, 10), oct_ret=0.05))
        dirty = base.copy()
        dirty.loc[dirty.index[:3], "settle"] = 0.0          # 月初脏行
        dirty.loc[dirty.index[3], "settle"] = np.nan
        sig = SeasonalStrategy(params={"min_years": 3}).on_bar(dirty)
        # 脏行剔除后 10 月月末价仍由干净行决定 → 方向不变
        self.assertEqual(sig.direction, "long")

    def test_factor_legs_and_name_for_governance(self):
        # 归因载体：单因子策略自身即腿（治理卡 --attribution 通用归因的接入口）
        strat = SeasonalStrategy(params={})
        self.assertEqual(strat.factor_name, "seasonal")
        self.assertEqual(set(strat.factor_legs.keys()), {"seasonal"})
        self.assertIs(strat.factor_legs["seasonal"], strat)


if __name__ == "__main__":
    unittest.main()
