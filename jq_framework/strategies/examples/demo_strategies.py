"""示例策略：教学用，展示 StrategyBase 的最小实现。

写自己的策略，照抄这里的结构即可：继承 StrategyBase，实现 on_bar()。

⚠️ 契约提醒（框架会强制检查）：
  1. 不得保存跨 bar 状态——分窗引擎会逐段重建实例，状态会丢
  2. direction ∈ {"long", "short", "flat", "hold"}
  3. target_pos ∈ [0, 1]
"""
from __future__ import annotations

import pandas as pd

from jq_framework.strategies.base import Signal, StrategyBase


class DualMaExample(StrategyBase):
    """双均线：快线上穿慢线做多，下穿做空。

    参数:
      fast: 快线窗口（默认 5）
      slow: 慢线窗口（默认 20）
    """

    default_params = {"fast": 5, "slow": 20}

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])

        if len(bars) < slow + 1:
            return Signal(direction="hold", note=f"数据不足(需 {slow + 1} 根)")

        close = bars["close"]
        ma_f = close.rolling(fast).mean()
        ma_s = close.rolling(slow).mean()

        prev_f, cur_f = ma_f.iloc[-2], ma_f.iloc[-1]
        prev_s, cur_s = ma_s.iloc[-2], ma_s.iloc[-1]
        price = float(close.iloc[-1])

        if prev_f <= prev_s and cur_f > cur_s:
            return Signal(direction="long", target_pos=1.0, price=price,
                          note=f"金叉 MA{fast} 上穿 MA{slow}")
        if prev_f >= prev_s and cur_f < cur_s:
            return Signal(direction="short", target_pos=1.0, price=price,
                          note=f"死叉 MA{fast} 下穿 MA{slow}")
        return Signal(direction="hold", note="无交叉")


class DonchianExample(StrategyBase):
    """唐奇安通道突破：突破 N 日高点做多，跌破 N 日低点做空。

    这是 Turtle 的经典形态，也是最有名的趋势跟踪原型。
    """

    default_params = {"entry_n": 20, "exit_n": 10}

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        entry_n = int(self.params["entry_n"])
        exit_n = int(self.params["exit_n"])

        if len(bars) < entry_n + 1:
            return Signal(direction="hold", note=f"数据不足(需 {entry_n + 1} 根)")

        close = bars["close"]
        # 用前一根为止的区间，避免用当根自身的极值（那是未来函数）
        hi = bars["high"].iloc[:-1].rolling(entry_n).max().iloc[-1]
        lo = bars["low"].iloc[:-1].rolling(entry_n).min().iloc[-1]
        price = float(close.iloc[-1])

        if price > hi:
            return Signal(direction="long", target_pos=1.0, price=price,
                          note=f"突破 {entry_n} 日高点 {hi:.2f}")
        if price < lo:
            return Signal(direction="short", target_pos=1.0, price=price,
                          note=f"跌破 {entry_n} 日低点 {lo:.2f}")
        return Signal(direction="hold", note="区间内")
