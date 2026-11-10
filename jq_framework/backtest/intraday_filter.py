"""
A 股日线 OHLCV 日内增强执行层：放量过滤 + 开盘强度过滤。

与 backtest/limits.py 同层、同性质——成交执行语义，非选股 alpha。在 select_portfolio
选出 top_n 之后、pending_trade 赋值之前调用，滤掉"近 N 日放量不达标"/"开盘下行"的票
（关注度/流动性不足、买点时点差）。日线 high/low/volume 是已被选股层丢弃的日内摘要，本模块
把它们重新用于执行层过滤（"低频选股 + 高频 overlay 两段式"的 overlay 侧）。

为什么不进 composite_score（选股层）：放量/开盘强度是关注度/流动性/时点信号，无方向性 alpha
（放量上涨与放量下跌方向相反）。base.py 的方向约定写死"因子值越大→预期收益越高"，硬塞进
打分会引入方向噪声；且 twolevel 族内正交会从 momentum 里扣掉伪相关线性部分，削弱真信号。
故放执行层独立过滤，训练段 IC 权重逐字不变，delta 干净归因到成交时点。

性质：无状态、纯 pandas → 线程安全（回测循环单线程）；无 path/os/文件 I/O →
Windows 安全；只在调仓日对 top_n 只票算几个比值，无新瓶颈。

链式：volume_surge 先 → open_strength 后，输入用前一因子输出的 _pending。两因子
计数器分开记录剔除力度，delta 是叠加效果（无法干净拆分单因子贡献，归因 hook A 仍记
选股层 new_syms 不被污染）。
"""
from __future__ import annotations

import pandas as pd


def _hard_filter_with_backfill(
    new_syms: list[str],
    signal_fn,
    threshold: float,
    min_keep: int,
    pass_if_none: bool = True,
) -> tuple[list[str], dict[str, int]]:
    """硬过滤 + 最小保留数兜底的通用骨架（filter_by_volume / filter_by_open_strength 共用）。

    signal_fn(sym) -> float | None：返回该标的的信号值（date 通过调用方闭包捕获）。
        None/NaN 按 pass_if_none 处理：True=保守保留（数据缺失不惩罚，与 is_tradeable 缺失返 True 同语义），
        False=剔除。
    threshold: signal >= threshold 则通过。
    min_keep: 过滤后若剩余 < min_keep，按原 new_syms 排名从被滤掉的里补齐到该数。
        （new_syms 已按综合打分排序，被滤掉的也保留原排名顺序）

    返回: (过滤后 symbols 列表, {"filtered": 净剔除数, "backfilled": 兜底补回数})
        口径：len(passed) = len(new_syms) - filtered（passed 已含兜底补回的）
    """
    if not new_syms:
        return [], {"filtered": 0, "backfilled": 0}

    passed: list[str] = []
    filtered_syms: list[str] = []  # 被滤掉的，按原排名顺序保留供兜底补齐

    for sym in new_syms:
        try:
            sig = signal_fn(sym)
        except Exception:
            sig = None
        if sig is None or (isinstance(sig, float) and pd.isna(sig)):
            if pass_if_none:
                passed.append(sym)
            else:
                filtered_syms.append(sym)
            continue
        if sig >= threshold:
            passed.append(sym)
        else:
            filtered_syms.append(sym)

    # 兜底：过滤后不足 min_keep，按原排名从被滤掉的里补齐
    backfilled = 0
    if len(passed) < min_keep:
        need = min_keep - len(passed)
        backfill = filtered_syms[:need]
        passed = passed + backfill
        backfilled = len(backfill)

    stats = {"filtered": len(new_syms) - len(passed), "backfilled": backfilled}
    return passed, stats


def filter_by_volume(
    new_syms: list[str],
    date,
    valid: dict[str, pd.DataFrame],
    top_n: int,
    threshold: float = 1.0,
    window: int = 20,
    min_keep_ratio: float = 0.8,
) -> tuple[list[str], dict[str, int]]:
    """放量硬过滤 + 最小保留数兜底（日线 volume 信号）。

    new_syms: select_portfolio 选出的候选篮子（已按综合打分排序，前 top_n 只）
    date: 当前调仓日（T 日收盘，取到该日及之前的 volume 序列）
    valid: {symbol: df} 全池日线，df 含 [date,open,high,low,close,volume]
    top_n: 目标持仓数（用于计算 min_keep）
    threshold: 放量阈值。当日 volume / 过去 window 日 volume 均值 >= threshold 则通过。
        1.0 = 当日量不低于近 window 日均值（默认最宽松）。
    window: 均量回看窗口（默认 20 日，≥20 防短期脉冲噪声）
    min_keep_ratio: 最小保留比例。过滤后若剩余 < top_n × min_keep_ratio，按原 new_syms
        排名从被滤掉的里补齐到该数。0.8 = top_n=100 时至少保留 80 只，防全市场缩量日
        仓位不足放大回撤。

    返回: (过滤后 symbols 列表, 统计字典)
    """
    min_keep = max(1, int(top_n * min_keep_ratio))

    def _vol_signal(sym: str) -> float | None:
        df = valid.get(sym)
        if df is None:
            return None  # 数据缺失保守保留（pass_if_none=True）
        # 取到 date 当日及之前的 volume 序列（防未来函数，与 low_vol.py 同范式）
        hist = df[df["date"] <= date]
        if len(hist) < window + 1:
            return None  # 历史不足无法算均量，保守保留
        vol_today = hist["volume"].iloc[-1]
        vol_mean = hist["volume"].iloc[-(window + 1):-1].mean()
        if pd.isna(vol_today) or pd.isna(vol_mean) or vol_mean <= 0:
            return None
        return vol_today / vol_mean

    return _hard_filter_with_backfill(new_syms, _vol_signal, threshold, min_keep,
                                      pass_if_none=True)


def filter_by_open_strength(
    new_syms: list[str],
    date,
    valid_min: dict[str, pd.DataFrame],
    top_n: int,
    threshold: float = 0.0,
    window: int = 48,
    min_keep_ratio: float = 0.8,
) -> tuple[list[str], dict[str, int]]:
    """开盘强度硬过滤 + 最小保留数兜底（分钟级 close/open 信号，第二日内因子）。

    用分钟 valid_min（非日线 valid）：取到 date 当日及之前的 bar，window 根 bar 的
    close[-1]/open[0]-1 >= threshold 则通过。0.0=开盘不跌（过滤开盘即下行的烂时点）。
    window=48 bar ≈ 1 交易日（5min 下）。

    语义/参数/返回口径与 filter_by_volume 完全同构，只是信号函数换成分钟开盘强度。
    性质同：纯 pandas 无状态无 os/path（Windows 安全、线程安全），只在调仓日对 top_n 只算。
    """
    min_keep = max(1, int(top_n * min_keep_ratio))

    def _os_signal(sym: str) -> float | None:
        df = valid_min.get(sym)
        if df is None:
            return None  # 分钟数据缺失保守保留
        # 防未来函数：date 是调仓日天精度 Timestamp，匹配当日及之前所有 bar
        hist = df[df["date"] <= date]
        if len(hist) < window:
            return None  # 历史不足一个 window，保守保留
        day = hist.iloc[-window:]
        o = day["open"].iloc[0]
        c = day["close"].iloc[-1]
        if pd.isna(o) or pd.isna(c) or o <= 0:
            return None
        return c / o - 1

    return _hard_filter_with_backfill(new_syms, _os_signal, threshold, min_keep,
                                      pass_if_none=True)


def filter_by_amplitude(
    new_syms: list[str],
    date,
    valid_min: dict[str, pd.DataFrame],
    top_n: int,
    threshold: float = 0.0,
    window: int = 48,
    min_keep_ratio: float = 0.8,
) -> tuple[list[str], dict[str, int]]:
    """振幅硬过滤 + 最小保留数兜底（分钟级 high/low/close 信号，第三日内因子）。

    用分钟 valid_min：取到 date 当日及之前的 bar，window 根 bar 的
    (max_high - min_low)/close[-1] = 振幅。IC 实测为负（IC=-0.108 t=-2.02，振幅大→未来
    收益差，"日内剧烈波动=出货/分歧"），故信号返回 **-振幅**，复用 >=threshold 的"通过下限"语义：
        -amp >= threshold  ⇔  amp <= -threshold   （保低振幅、滤高振幅）
    threshold=0.0 = 振幅非负（最宽松，默认）；配负 threshold 收紧（如 -0.05=振幅≤5%）。
    direction 不可写反：振幅恒 ≥0，若直接用 +振幅 配正 threshold 会全通过、配负 threshold 亦全通过，
    取负后下限语义才正确生效。与因子 docstring"负 IC 配负 threshold"意图一致（此处以取负实现，等价）。

    语义/参数/返回口径与 filter_by_open_strength 完全同构，只信号函数换为 -振幅。
    性质同：纯 pandas 无状态无 os/path（Windows 安全、线程安全），只在调仓日对 top_n 只算。
    """
    min_keep = max(1, int(top_n * min_keep_ratio))

    def _amp_signal(sym: str) -> float | None:
        df = valid_min.get(sym)
        if df is None:
            return None  # 分钟数据缺失保守保留
        # 防未来函数：date 是调仓日天精度 Timestamp，匹配当日及之前所有 bar
        hist = df[df["date"] <= date]
        if len(hist) < window:
            return None  # 历史不足一个 window，保守保留
        day = hist.iloc[-window:]
        hi = day["high"].max()
        lo = day["low"].min()
        c = day["close"].iloc[-1]
        if pd.isna(hi) or pd.isna(lo) or pd.isna(c) or c <= 0:
            return None
        return -((hi - lo) / c)  # 取负：低振幅信号大→通过下限保留

    return _hard_filter_with_backfill(new_syms, _amp_signal, threshold, min_keep,
                                      pass_if_none=True)
