"""
策略基类 + Signal —— 整个系统的核心接口。

设计思想：策略是一个"数据进 → 信号出"的对象。
  回测和每日信号都调用 on_bar，只是喂的数据范围不同。
  这样一套策略代码同时服务 A（每日信号）和 B（回测），无需维护两份。

注意：这里不依赖 backtrader，保持框架无关。
      backtrader 适配在 adapter.py，让本基类的策略能跑在 backtrader 上。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Signal:
    """一个策略在某个时点产出的交易信号。"""
    direction: str           # "long" 做多 / "short" 做空 / "flat" 平仓 / "hold" 维持
    target_pos: float = 0.0  # 目标仓位比例 [0, 1]（相对总资金）
    price: Optional[float] = None   # 参考价（通常是当前 close）
    note: str = ""           # 触发原因，便于人读

    @property
    def is_actionable(self) -> bool:
        """是否需要实际调仓（hold/flat 持平时不算）。"""
        return self.direction in ("long", "short", "flat")


class StrategyBase:
    """
    策略基类。子类只需实现 on_bar。

    用法（每日信号场景）:
        strat = MyStrategy(params={"fast":5,"slow":20})
        bars  = df  # DataFrame[date,open,high,low,close,volume]
        sig   = strat.on_bar(bars)   # 看最新一根 bar 该干嘛

    用法（回测场景）由 adapter.py 驱动：
        backtrader 在每个 bar 调 strat.on_bar(到当前为止的bars)，拿到信号后下单。
    """
    # 默认参数，子类覆盖
    default_params: dict = {}

    def __init__(self, params: dict | None = None):
        self.params = {**self.default_params, **(params or {})}

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        """
        根据到当前为止的 K 线数据返回信号。

        bars: DataFrame，包含历史到当前 bar 的数据，列 date/open/high/low/close/volume
              （每日信号时 bars 是完整历史；回测时 bars 是到当前 bar 为止的子集）
        返回: Signal
        """
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
