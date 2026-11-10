"""示例因子：教学用，不含任何真实 alpha。

这两个因子的作用是**证明框架能跑通**，不是提供收益：
  Momentum20      20 日动量（最经典的时序因子，公开知识）
  Reversal20      20 日反转（动量的反面）

写自己的因子，照抄这里的结构即可：继承 FactorBase，实现 calc()。
"""
from __future__ import annotations

import pandas as pd

from jq_framework.factors.base import FactorBase


class Momentum20(FactorBase):
    """20 日动量：close / close[-21] - 1。

    方向约定：值越大 → 预期未来收益越高（所以直接返回收益率，不取负）。
    """

    name = "momentum20"

    def calc(self, df: pd.DataFrame) -> float | None:
        n = int(self.params.get("window", 20))
        if len(df) < n + 1:
            return None
        close = df["close"]
        return float(close.iloc[-1] / close.iloc[-n - 1] - 1.0)


class Reversal20(FactorBase):
    """20 日反转：动量的反面。

    注意这里的 `-`：因子基类要求"值越大越好"，而反转逻辑是"跌得多的要涨"，
    所以必须取负号翻转。这是写因子最容易搞错的地方。
    """

    name = "reversal20"

    def calc(self, df: pd.DataFrame) -> float | None:
        n = int(self.params.get("window", 20))
        if len(df) < n + 1:
            return None
        close = df["close"]
        return -float(close.iloc[-1] / close.iloc[-n - 1] - 1.0)
