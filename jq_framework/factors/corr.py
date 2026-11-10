"""
价量相关性因子：过去 N 日收盘价与 log(成交量) 的滚动时序相关。

来源：Qlib Alpha158 的 CORR 算子（[[qlib-alpha158-survey]] 调研真增量之一），
     2026-09-09 门1-3 认证为独立新维度（见 [[new-factor-process-checklist]]）：
  - 门1 IC: CORR10=-0.050 ICIR=-0.479 t=-4.17（全场 158 特征最高 ICIR）
            CORR5 =-0.031 ICIR=-0.314 t=-2.73
  - 门2 与现有 11 因子 max|corr|<=0.30（vs momentum 独立，非动量/反转/波动变体）
  - 门3 正交掉 momentum/reversal/low_vol 后残差 IC 仍显著（CORR10 t=-3.55）
逻辑：价格水平与成交量的时序相关（放量推高=正价量相关）。负 IC = 高正价量相关 → 下期收益低
     （放量追高后回归），符合 A 股微观结构。
方向：直接取 corr 值（因子值越大=价量正相关越强→预期收益越低），由 walk-forward IC 自动定符号。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from jq_framework.factors.base import FactorBase


class CorrFactor(FactorBase):
    name = "corr"
    process_method = "rank"

    def calc(self, df: pd.DataFrame) -> float | None:
        n = self.params.get("window", 10)
        if len(df) < n + 1:
            return None
        close = df["close"]
        logv = np.log(df["volume"] + 1)
        c = close.rolling(n, min_periods=n).corr(logv)
        v = c.iloc[-1]
        return None if (v is None or np.isnan(v)) else float(v)

    def calc_series(self, df: pd.DataFrame) -> pd.Series | None:
        """向量化：全历史一次算 rolling corr（无前视，全历史某日值==截断末值）。"""
        n = self.params.get("window", 10)
        if "date" in df.columns:
            df = df.set_index("date")
        close = df["close"]
        logv = np.log(df["volume"] + 1)
        return close.rolling(n, min_periods=n).corr(logv)
