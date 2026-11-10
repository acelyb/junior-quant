"""
大盘择时层：RSRS（阻力支撑相对强度）仓位乘数。

核心思想（光大金工研报《再探技术分析指数》）：用过去 N 日「最高价对最低价」做
OLS 回归 high = α + β·low，斜率 β 衡量阻力位与支撑位的相对强弱——β 大说明高点
比低点涨得快（支撑强→多头），β 小说明高点涨不动（阻力强→减仓）。

设计定位（重要）：这是**择时层**非选股层。与 factors/ 截面选股正交：
- 选股层（composite_score）定"买什么"——横截面排序，每只股票有相对得分
- 择时层（本模块）定"何时全仓/减仓"——时间序列，对整个组合施加统一仓位乘数
两者相乘：个股目标权重 = 选股等权(或波动率加权) × 大盘择时乘数。

不做的事（与博客/研报的差异，诚实标注）：
- 不做"右偏标准分"（β_norm × R² × β）——博客自承四个变体里只有斜率复现稳定，
  标准分/修正/右偏均与研报差异大且 2018 后长期空仓踏空。先用最朴素的"β z-score
  阈值减仓"验证有无样本外降回撤效果，朴素版不 work 则复杂版更不会 work。
- 不做个股 RSRS——维度错位（个股 β 无截面可比性）且与 momentum/low_vol 信息重叠。
- 不做空仓——保留底仓（min_exposure），避免长期空仓踏空结构性行情（2019-2021 茅指数、
  2024 小盘），这是博客 RSRS 的已知缺陷。减仓而非清仓。

防未来函数：z-score 用 shift(1) 的滚动均值/标准差（只用 t-1 及之前的数据算统计量）。
OLS 斜率 β 本身用 [t-N+1, t] 窗口，t 日 high/low 在 t 日收盘后已知，作为 t 日信号
驱动 t+1 仓位，无未来函数（与选股层 T 收盘信号→T+1 开盘执行同口径）。

四硬约束：
- Windows：纯 pandas/numpy，无 os/path/subprocess，无文件 I/O
- 正确性+线程安全：纯函数无全局可变状态，无 mutate 入参
- 资源：O(NT) 向量化滚动回归，N=18 轻量；只在每个交易日算一次指数（单序列非全池）
- 优化：numpy.polyfit 替代 statsmodels.OLS（轻量，项目已用 numpy 线性代数）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_beta(high: np.ndarray, low: np.ndarray, n: int) -> np.ndarray:
    """对每个窗口 [i-n+1, i] 做 OLS high=α+β·low，返回逐点斜率 β 数组。

    向量化滚动 OLS：β = cov(high,low)/var(low)，用滚动和计算（O(NT)，N=18 极轻量）。
    等价于 numpy.polyfit(x,y,1)[0] 但避免 per-window 函数调用开销。
    前 n-1 个点无完整窗口→NaN。low 方差为 0（窗口内 low 恒定，如停牌）→NaN（不强行拟合）。
    """
    m = len(high)
    beta = np.full(m, np.nan)
    if n < 2 or m < n:
        return beta
    # 滚动求和：用 cumsum 差分 O(N) 算每个窗口的 Σx,Σy,Σx²,Σxy
    h = pd.Series(high).rolling(n, min_periods=n).sum().to_numpy()
    l = pd.Series(low).rolling(n, min_periods=n).sum().to_numpy()
    hh = pd.Series(high * high).rolling(n, min_periods=n).sum().to_numpy()
    hl = pd.Series(high * low).rolling(n, min_periods=n).sum().to_numpy()
    ll = pd.Series(low * low).rolling(n, min_periods=n).sum().to_numpy()
    # β = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²)，分母=0 时（low 恒定）置 NaN
    denom = n * ll - l * l
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(denom != 0, (n * hl - l * h) / denom, np.nan)
    return beta


def _rolling_r2(high: np.ndarray, low: np.ndarray, beta: np.ndarray, n: int) -> np.ndarray:
    """滚动回归决定系数 R²（保留供后续修正标准分扩展用，当前仓位乘数未用）。

    R² = corr(high,low)² = β² × var(low)/var(high)。与 β 同窗口。方差为 0 → NaN。
    """
    m = len(high)
    r2 = np.full(m, np.nan)
    if n < 2 or m < n:
        return r2
    var_low = pd.Series(low).rolling(n, min_periods=n).var(ddof=0).to_numpy()
    var_high = pd.Series(high).rolling(n, min_periods=n).var(ddof=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(var_high != 0, beta * beta * var_low / var_high, np.nan)
    return r2


def calc_rsrs(index_df: pd.DataFrame, n: int = 18, m: int = 600) -> pd.DataFrame:
    """计算 RSRS 斜率 β 及其 z-score 标准分。

    index_df: 指数日线，需含列 date/high/low（升序）。date 为 datetime 或 str 均可。
    n: OLS 回归窗口（默认 18，研报值）
    m: z-score 标准化窗口（默认 600，研报值；不足 m 时用已有数据算，前期 z 不稳）

    返回: DataFrame[date, beta, rsrs_z]，升序。
        beta: 当日斜率（[t-n+1, t] 窗口 OLS）。NaN=前 n-1 天或窗口内 low 恒定。
        rsrs_z: beta 的 z-score，用 shift(1) 的滚动均值/标准差（防未来函数）。
            NaN=前 n+m-1 天或统计量未就绪。
    """
    df = index_df.sort_values("date").reset_index(drop=True).copy()
    if len(df) < n:
        return pd.DataFrame(columns=["date", "beta", "rsrs_z"])
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    beta = _rolling_beta(high, low, n)

    # z-score：均值/标准差用 shift(1)（只用 t-1 及之前），严格防未来函数
    beta_s = pd.Series(beta)
    roll_mean = beta_s.rolling(m, min_periods=max(2, n)).mean().shift(1)
    roll_std = beta_s.rolling(m, min_periods=max(2, n)).std(ddof=0).shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsrs_z = (beta_s - roll_mean) / roll_std
    rsrs_z = rsrs_z.where(roll_std > 0)  # std=0（beta 恒定）置 NaN

    df["beta"] = beta
    df["rsrs_z"] = rsrs_z.to_numpy()
    return df[["date", "beta", "rsrs_z"]]


def timing_multiplier(rsrs_z: float | None,
                      entry: float = 0.7, exit: float = -0.7,
                      full_exposure: float = 1.0,
                      min_exposure: float = 0.3) -> float:
    """RSRS z-score → 仓位乘数（单点）。

    分段线性：z >= entry → full_exposure（满仓）；z <= exit → min_exposure（底仓）；
    exit < z < entry → 线性插值。保留底仓 min_exposure（默认 0.3）避免长期空仓踏空。

    z 为 None/NaN（数据不足或停牌）→ full_exposure（不因数据缺失惩罚仓位，保守保留），
    与 limits.py is_tradeable 数据缺失返 True 同语义。

    entry/exit: z 的进/出仓位阈值（默认 ±0.7，研报值）。entry > exit 必须成立。
    full_exposure/min_exposure: 满仓/底仓乘数，应在 [0,1]，full >= min。
    """
    if rsrs_z is None or pd.isna(rsrs_z):
        return full_exposure
    if entry <= exit:
        return full_exposure  # 参数非法，保守满仓
    z = float(rsrs_z)
    if z >= entry:
        return float(full_exposure)
    if z <= exit:
        return float(min_exposure)
    # exit < z < entry：线性插值
    t = (z - exit) / (entry - exit)
    return float(min_exposure + t * (full_exposure - min_exposure))


def calc_timing_series(index_df: pd.DataFrame,
                       n: int = 18, m: int = 600,
                       entry: float = 0.7, exit: float = -0.7,
                       full_exposure: float = 1.0,
                       min_exposure: float = 0.3) -> pd.DataFrame:
    """计算 RSRS z-score 序列及对应仓位乘数序列（回测用，向量化）。

    返回: DataFrame[date, beta, rsrs_z, timing_mult]，升序。timing_mult 为每个交易日的
    仓位乘数，用于驱动当日（及之后）组合仓位。NaN rsrs_z → full_exposure（保守）。
    """
    rsrs = calc_rsrs(index_df, n=n, m=m)
    if rsrs.empty:
        rsrs["timing_mult"] = []
        return rsrs
    z = rsrs["rsrs_z"].to_numpy()
    mult = np.where(
        pd.isna(z), full_exposure,
        np.where(z >= entry, full_exposure,
                 np.where(z <= exit, min_exposure,
                          min_exposure + (z - exit) / (entry - exit) * (full_exposure - min_exposure))))
    rsrs["timing_mult"] = mult
    return rsrs


class MarketTimer:
    """大盘择时器：用真实指数日线预算 RSRS 仓位乘数字典，回测中按日期查表。

    设计：回测开始前一次性算好全部历史日期的乘数 → {date: mult}，回测主循环每日
    O(1) 查表，避免每个交易日重复做滚动回归（资源约束）。乘数对整个组合统一施加，
    是"总仓位层"，与个股波动率加权（"相对权重层"）串联。

    用法：
        timer = MarketTimer(cfg)        # cfg=None 或 market_timing.enabled=false → no-op（恒返回 1.0）
        timer.build(index_df)           # 预算乘数字典（用真实指数日线）
        mult = timer.get_multiplier(date)  # 回测中查表，date 为 str/Timestamp/datetime64

    门控：enabled=false 时 build/get_multiplier 都是 no-op（恒 1.0），不拉数据不算回归，
    保证默认配置零开销且不破回归门（与 intraday_filter.enabled 同范式）。
    """

    def __init__(self, cfg: dict | None = None):
        """cfg: multifactor.market_timing 字典（config.yaml）。None 或 enabled=false → no-op。"""
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        # RSRS 参数（与 calc_rsrs/calc_timing_series 一致）
        self.n = int(cfg.get("n", 18))
        self.m = int(cfg.get("m", 600))
        self.entry = float(cfg.get("entry", 0.7))
        self.exit = float(cfg.get("exit", -0.7))
        self.full_exposure = float(cfg.get("full_exposure", 1.0))
        self.min_exposure = float(cfg.get("min_exposure", 0.3))
        self._mult_by_date: dict[pd.Timestamp, float] = {}  # 预算乘数字典（查表用）

    def build(self, index_df: pd.DataFrame) -> "MarketTimer":
        """用真实指数日线预算乘数字典。index_df 需含 date/high/low（升序）。

        enabled=false 时跳过（保持空字典，get_multiplier 恒返 full_exposure）。
        空数据也跳过（保守不施加择时）。
        """
        if not self.enabled or index_df is None or index_df.empty:
            return self
        ts = calc_timing_series(
            index_df, n=self.n, m=self.m,
            entry=self.entry, exit=self.exit,
            full_exposure=self.full_exposure, min_exposure=self.min_exposure)
        # 用 Timestamp 作 key 便于跨 str/datetime64 查表
        self._mult_by_date = {
            pd.Timestamp(d): float(m)
            for d, m in zip(ts["date"], ts["timing_mult"])}
        return self

    def get_multiplier(self, date) -> float:
        """查表取某日仓位乘数。date 为 str/Timestamp/datetime64 均可。

        未启用 / 字典空 / 该日无记录 → full_exposure（保守满仓，不因数据缺失惩罚仓位，
        与 limits.py is_tradeable 数据缺失返 True 同语义）。
        """
        if not self.enabled or not self._mult_by_date:
            return self.full_exposure
        return self._mult_by_date.get(pd.Timestamp(date), self.full_exposure)

    def coverage(self) -> tuple[int, float, float, float]:
        """诊断：返回 (覆盖天数, 平均仓位, 最低仓位, 满仓占比)。未启用返 (0,1,1,1)。"""
        if not self._mult_by_date:
            return (0, self.full_exposure, self.full_exposure, 1.0)
        ms = list(self._mult_by_date.values())
        n = len(ms)
        return (n, sum(ms) / n, min(ms), sum(1 for x in ms if x >= self.full_exposure - 1e-9) / n)
