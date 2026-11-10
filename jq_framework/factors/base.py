"""
因子基类与截面处理工具。

多因子系统的核心概念：
- 因子（Factor）：对每只股票，在每个时点算出一个标量值，用来预测未来收益
- 截面（cross-section）：同一时点所有股票的因子值集合
- 处理流程：原始因子值 → 去极值 → 标准化 → 打分

防未来函数原则：因子计算只用到当前时点及之前的数据。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import inspect

import numpy as np
import pandas as pd


# ============ 截面处理工具 ============

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    去极值：把超出 [lower, upper] 分位数的值截断到分位数边界。
    防止少数极端值主导排序。
    """
    if series.empty:
        return series
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore_standardize(series: pd.Series) -> pd.Series:
    """
    标准化：减均值除标准差，使因子值跨股票可比、跨因子可加。
    """
    if series.empty:
        return series
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


def rank_standardize(series: pd.Series) -> pd.Series:
    """
    秩标准化：转成 [0,1] 的秩，对异常值更鲁棒。多因子常用。
    """
    if series.empty:
        return series
    ranks = series.rank(method="average")
    return (ranks - ranks.min()) / (ranks.max() - ranks.min() if ranks.max() != ranks.min() else 1)


def process_factor(raw: pd.Series, method: str = "rank") -> pd.Series:
    """
    标准截面处理：去极值 → 标准化。
    method: "rank"(秩标准化，推荐) / "zscore"
    """
    w = winsorize(raw)
    if method == "zscore":
        return zscore_standardize(w)
    return rank_standardize(w)


def neutralize_industry(raw: pd.Series, industry_map: dict[str, str]) -> pd.Series:
    """
    行业中性化：对原始因子值回归行业哑变量，取残差。
    剔除行业系统性差异，使因子值在行业内可比。

    raw: 截面因子值（index=symbol）
    industry_map: {symbol: 行业名}（get_stock_industry_map）
    返回: 残差序列（index=symbol），行业缺失的股票保留原值
    """
    if raw.empty or not industry_map:
        return raw
    # 对齐到有行业信息的股票
    idx = raw.index.astype(str)
    ind = pd.Series([industry_map.get(s) for s in idx], index=raw.index)
    has_ind = ind.notna()
    if has_ind.sum() < 2 or ind[has_ind].nunique() < 2:
        # 行业信息不足，无法回归，直接返回原值
        return raw
    # 行业哑变量回归：因子值 = 行业均值 + 残差
    sub_raw = raw[has_ind]
    sub_ind = ind[has_ind]
    dummies = pd.get_dummies(sub_ind, drop_first=True).astype(float)
    # 最小二乘：residual = y - X @ beta，beta = (X'X)^-1 X'y
    X = np.column_stack([np.ones(len(sub_raw)), dummies.values])
    y = sub_raw.values.astype(float)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
    except Exception:
        return raw
    result = raw.copy().astype(float)
    result[has_ind] = resid
    return result


# ============ 因子基类 ============

class FactorBase(ABC):
    """
    因子基类。子类实现 calc，输入单只股票的日线，返回当前时点的因子值。

    方向约定：因子值越大 → 预期未来收益越高（即"越大越好"）。
    如果原始因子是"越大越差"（如波动率），子类需取负号翻转。
    """
    # 因子名，子类覆盖
    name: str = "base"
    # 截面处理方法（可由 params.process_method 覆盖）
    process_method: str = "rank"
    # 去极值分位（可由 params.winsorize_quantile 覆盖）。(0.01, 0.99) = 前后各截 1%
    winsorize_quantile: tuple[float, float] = (0.01, 0.99)
    # 是否做行业中性化（由配置 multifactor.industry_neutral 控制，默认关闭）
    industry_neutral: bool = False
    # 是否依赖基本面数据（子类按需开启，调用方据此决定是否拉取/注入 fund_data）
    is_fundamental: bool = False
    # 因子大类（由 load_factor 从 config 注入；空串=未设置，回退到 name）
    # 用于 twolevel 合成：族内正交 + 族间风险预算。flat 模式下被忽略
    family: str = ""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        # 行业映射，由回测主流程注入（set_industry_map）；中性化关闭时为 None
        self._industry_map: dict[str, str] | None = None
        # 基本面数据，由回测主流程注入（set_fundamental_data）；价量因子不用，为 None
        self._fund_data: dict[str, pd.DataFrame] | None = None
        # 一次性探测子类 calc 是否接受 symbol/date 参数，缓存避免每次反射
        sig = inspect.signature(self.calc)
        self._calc_accepts_symbol = "symbol" in sig.parameters
        self._calc_accepts_date = "date" in sig.parameters

    def set_industry_map(self, industry_map: dict[str, str] | None) -> None:
        """注入行业映射，开启后该因子在截面处理时会做行业中性化。"""
        self._industry_map = industry_map

    def set_fundamental_data(self, fund_data: dict[str, pd.DataFrame] | None) -> None:
        """注入基本面数据 {symbol: df}，基本面因子在 calc 里按 symbol 取用。价量因子忽略。"""
        self._fund_data = fund_data

    @abstractmethod
    def calc(self, df: pd.DataFrame) -> float | None:
        """
        计算单只股票当前时点的因子值。
        df: 该股票到当前时点为止的日线 [date,open,high,low,close,volume]
        返回: 因子值（float），数据不足返回 None

        基本面因子可扩展签名为 calc(self, df, symbol=None, date=None)，
        由 calc_value 自动适配传参；价量因子保持 calc(self, df) 即可。
        """
        ...

    def calc_series(self, df: pd.DataFrame) -> pd.Series | None:
        """向量化入口（可选覆盖）：一次算出该股票所有时点的因子值序列。

        df: 单股票完整历史日线（已 set_index(date) 或带 date 列均可，子类自处理）。
        返回: pd.Series(index=日期)，每行=该时点因子值；warmup 段为 NaN。
              未实现向量化时返回 None（基类默认）→ 调用方（如 calc_factor_ic）回退逐点 calc_value。

        为什么有这个钩子：calc_factor_ic 在每个采样日对每只股票截断 df 再调 calc_value，
        每次都从头算一遍整条 rolling 序列只取 .iloc[-1]，是 O(n²)。覆盖 calc_series 后，
        调用方每只股票只算一次全序列、按采样日索引，降到 O(n)，口径逐字一致
        （纯 rolling/前缀运算只回看，全序列在某位置的值 == 截断到该位置的末值）。
        """
        return None

    def calc_value(self, df: pd.DataFrame, symbol: str | None = None,
                   date=None) -> float | None:
        """统一入口：按子类 calc 签名自动适配，只传它接受的参数。价量因子零侵入。"""
        if self._calc_accepts_symbol or self._calc_accepts_date:
            kwargs = {}
            if self._calc_accepts_symbol:
                kwargs["symbol"] = symbol
            if self._calc_accepts_date:
                kwargs["date"] = date
            return self.calc(df, **kwargs)
        return self.calc(df)

    def calc_cross_section(self, price_dfs: dict[str, pd.DataFrame],
                           series_cache: dict[str, pd.Series] | None = None) -> pd.Series:
        """
        对所有股票计算因子值，做截面处理，返回处理后的因子序列。
        price_dfs: {symbol: df}，每个 df 已截到当前时点
        返回: pd.Series(index=symbol)，处理后的因子值

        处理顺序：原始值 → 去极值 → 行业中性化(可选) → 标准化

        series_cache: 可选，{symbol: 全历史因子序列(Series, index=date)}。提供时跳过逐股票
            calc_value 重算，直接按当前时点日期从序列取值——把"每调仓日×每股票从头算 rolling"
            降为"每窗口每股票算一次全序列、调仓日索引"。None 时走原逐点 calc_value 路径（逐字不变）。
            序列由调用方在窗口边界用 calc_series(完整 df) 预算一次；本方法只读、不改它。
        """
        raw = {}
        for sym, df in price_dfs.items():
            if df is None or df.empty:
                continue
            # 当前时点 = 价格 df 的最后一行日期（基本面因子据此取最近已公告值，防未来函数）
            # "date" 在列里（原 df）取列；不在列里但索引是 DatetimeIndex（set_index("date") 后）
            # 取索引——两种构造方式都支持 series_cache 按日期索引。
            if "date" in df.columns:
                date = df["date"].iloc[-1]
            elif isinstance(df.index, pd.DatetimeIndex):
                date = df.index[-1]
            else:
                date = None
            if series_cache is not None and sym in series_cache:
                s = series_cache[sym]
                fv = s.get(date) if date is not None else None
                if fv is not None and not isinstance(fv, float):
                    fv = float(fv) if not pd.isna(fv) else None
                if fv is not None and isinstance(fv, float) and np.isnan(fv):
                    fv = None
            else:
                try:
                    val = self.calc_value(df, symbol=sym, date=date)
                except Exception:
                    val = None
                fv = val
            if fv is not None and not (isinstance(fv, float) and np.isnan(fv)):
                raw[sym] = fv
        if not raw:
            return pd.Series(dtype=float)
        s = pd.Series(raw)
        lo_q, hi_q = self.params.get("winsorize_quantile", self.winsorize_quantile)
        s = winsorize(s, float(lo_q), float(hi_q))
        if self.industry_neutral and self._industry_map:
            s = neutralize_industry(s, self._industry_map)
        method = self.params.get("process_method", self.process_method)
        if method == "zscore":
            return zscore_standardize(s)
        return rank_standardize(s)
