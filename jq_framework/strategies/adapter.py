"""
backtrader 适配层。

作用：让 strategies/base.py 里框架无关的策略，能跑在 backtrader 回测引擎上。

思路：
- 自定义一个 bt.Strategy 子类 BacktraderAdapter
- 它内部持有一个我们的 StrategyBase 实例
- 用 self.data 的 line 对象直接取当前 bar 的值，自己累积一个 DataFrame
- 每个 bar 把累积的 DataFrame 喂给我们的策略，拿到 Signal，翻译成下单动作

这样策略逻辑全在我们的基类里，backtrader 只负责撮合和绩效统计。
"""
from __future__ import annotations

import backtrader as bt
import pandas as pd

from jq_framework.strategies.base import Signal, StrategyBase


class BacktraderAdapter(bt.Strategy):
    """
    把 StrategyBase 包装成 backtrader 策略。

    用法见 backtest/runner.py。
    """
    params = (
        ("strat", None),   # StrategyBase 实例
        ("market", "stock"),  # stock / cta，影响做空与 T+1 处理
    )

    def __init__(self):
        self.strat: StrategyBase = self.p.strat
        self._last_signal: Signal | None = None
        self._orders = []
        # 累积的 bar 数据，每个 next 追加当前行
        self._rows: list[dict] = []

    def next(self):
        """每个新 bar 调用一次。"""
        # 取当前 bar 的值（line 对象在 next() 中指向当前 bar）
        row = {
            "date": bt.num2date(self.data.datetime[0]).date(),
            "open": self.data.open[0],
            "high": self.data.high[0],
            "low": self.data.low[0],
            "close": self.data.close[0],
            "volume": self.data.volume[0],
        }
        self._rows.append(row)

        bars = pd.DataFrame(self._rows)
        if len(bars) < 2:
            return

        sig = self.strat.on_bar(bars)
        self._last_signal = sig
        self._execute_signal(sig)

    def _execute_signal(self, sig: Signal):
        """把 Signal 翻译成 backtrader 订单。"""
        # target_pos 封顶，避免满仓时因手续费/滑点导致现金不足被拒(Margin)
        pos = max(0.0, min(sig.target_pos, 0.98))

        if sig.direction == "long":
            self.order_target_percent(target=pos)
        elif sig.direction == "short":
            # A 股不能做空；期货可以
            if self.p.market == "cta":
                self.order_target_percent(target=-pos)
            # stock 模式下 short 被忽略
        elif sig.direction == "flat":
            self.order_target_percent(target=0.0)
        # hold: 什么都不做

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self._orders.append(order)
