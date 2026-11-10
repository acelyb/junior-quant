"""
IC（信息系数）计算 —— 因子有效性的核心检验指标。

核心概念：
- IC：某日因子值与未来 N 日收益的截面相关性。
  Rank IC 用秩相关（Spearman），比 Pearson 鲁棒，是业界标准。
- IC > 0：因子值大的股票未来收益高（因子有效，正向）
- IC < 0：因子值大的股票未来收益低（因子反向有效）
- |IC均值| > 0.03 算有弱预测力，> 0.05 不错，> 0.1 很强

关键指标：
- IC 均值：因子平均预测力
- ICIR（IR）：IC均值 / IC标准差，衡量预测力的稳定性，> 0.5 算好
- IC 胜率：IC > 0 的天数占比（正向因子应 > 50%）
- IC t 值：IC均值 / (IC标准差/sqrt(n))，|t|>2 统计显著

防未来函数：t 日的因子值与 [t+1, t+N] 的收益配对，用未来收益检验，本身不引入未来信息（这是检验，不是策略）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jq_framework.factors.base import FactorBase
from jq_framework.factors.composite import composite_score


def _future_return(dates, closes, idx: int, n: int) -> float | None:
    """计算 idx 位置未来 n 日收益。无足够数据返回 None。"""
    if idx + n >= len(dates):
        return None
    c0 = closes.iloc[idx]
    c1 = closes.iloc[idx + n]
    if c0 <= 0:
        return None
    return c1 / c0 - 1


def calc_factor_ic(price_dfs: dict[str, pd.DataFrame], factor: FactorBase,
                   horizon: int = 10, freq: int = 5) -> pd.DataFrame:
    """
    计算单因子在多个时点的 IC 序列。

    price_dfs: {symbol: 完整历史 df}
    factor: 因子实例
    horizon: 未来收益持有期（天）
    freq: 采样频率（每隔几天算一次IC，减少计算量且去自相关）

    返回: DataFrame[date, ic]，每日（采样点）的 Rank IC
    """
    # 对齐所有股票的日期轴
    all_dates = sorted(set().union(*[set(df["date"]) for df in price_dfs.values()]))
    # 每只股票的 close 序列，按日期对齐（用 dict 加速）
    sym_data = {}
    for sym, df in price_dfs.items():
        if df.empty:
            continue
        d = df.set_index("date")
        sym_data[sym] = d

    # 向量化捷径：因子若实现了 calc_series，每只股票只算一次全序列、按采样日索引，
    # 替代"每个采样日截断 df 再从头算 rolling"的 O(n²) 退化。口径与逐点 calc_value
    # 逐字一致（纯 rolling/前缀运算只回看过去）。未实现则 series_map 为空、回退原路径。
    series_map: dict[str, pd.Series] = {}
    use_series = False
    try:
        use_series = factor.calc_series is not FactorBase.calc_series
    except AttributeError:
        use_series = False
    if use_series:
        for sym, d in sym_data.items():
            try:
                s = factor.calc_series(d)
            except Exception:
                s = None
            if s is not None:
                series_map[sym] = s

    records = []
    # 采样时点：保证有 horizon 天的未来数据
    for i in range(0, len(all_dates) - horizon, freq):
        date = all_dates[i]
        factor_vals = {}
        future_rets = {}
        for sym, d in sym_data.items():
            # 索引单调递增（fetcher 已 sort_values + drop_duplicates）→ searchsorted O(log n)
            # 比 d.loc[:date].reset_index() 快一个量级：避免每采样日×每股票复制整张截断 df。
            # 不 reset_index：calc_value 只读 df["close"]（价量因子）或用 date kwarg 取 _fund_data
            # （基本面因子），均不读 df["date"] 列；而 calc_cross_section 才读 date 列——这里不喂它。
            # np.datetime64(date)：numpy>=2.0 的 searchsorted 对 datetime64 数组+Timestamp 值会
            # 误判类型（'<' between int and Timestamp），转 np.datetime64 双向兼容（日级/分钟级）。
            pos = int(np.searchsorted(d.index.values, np.datetime64(date)))
            if pos >= len(d) or d.index[pos] != date:
                continue  # date 不在该股票的日期轴上（与原 `date not in d.index` 等价）

            if sym in series_map:
                fv = series_map[sym].get(date)
                if fv is not None and (not isinstance(fv, float)) and hasattr(fv, "__float__"):
                    fv = float(fv)
                if fv is not None and isinstance(fv, float) and np.isnan(fv):
                    fv = None
            else:
                # 截到 date 的数据算因子（位置切片，含 date 行；与 d.loc[:date] 等价）
                sub = d.iloc[:pos + 1]
                try:
                    fv = factor.calc_value(sub, symbol=sym, date=date)
                except Exception:
                    fv = None
            if fv is None or (isinstance(fv, float) and np.isnan(fv)):
                continue
            # 未来 horizon 日收益：复用上面已算的 pos，省掉原来的 closes.index.get_loc(date)
            closes = d["close"]
            fr = _future_return(closes.index, closes, pos, horizon)
            if fr is None:
                continue
            factor_vals[sym] = fv
            future_rets[sym] = fr

        if len(factor_vals) < 10:  # 样本太少不算
            continue

        f_series = pd.Series(factor_vals)
        r_series = pd.Series(future_rets)
        # 对齐
        common = f_series.index.intersection(r_series.index)
        if len(common) < 10:
            continue
        ic, _ = spearmanr(f_series.loc[common], r_series.loc[common])
        if not np.isnan(ic):
            records.append({"date": date, "ic": ic})

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame(columns=["ic"])


def calc_composite_ic(price_dfs: dict[str, pd.DataFrame],
                      factors: list[tuple[FactorBase, float]],
                      horizon: int = 10, freq: int = 5,
                      combine_cfg: dict | None = None) -> pd.DataFrame:
    """
    计算多因子合成打分的 IC（检验组合是否比单因子更好）。

    combine_cfg: 透传给 composite_score（None=flat，与回测一致）。twolevel 合成 IC
                 反映"分层正交+风险预算"后组合打分的预测力，与 flat 可比。
    """
    all_dates = sorted(set().union(*[set(df["date"]) for df in price_dfs.values()]))
    sym_data = {sym: df.set_index("date") for sym, df in price_dfs.items() if not df.empty}

    records = []
    for i in range(0, len(all_dates) - horizon, freq):
        date = all_dates[i]
        current = {sym: d.loc[:date].reset_index() for sym, d in sym_data.items()
                   if date in d.index}
        score = composite_score(current, factors, combine_cfg=combine_cfg)
        # 未来收益
        fr = {}
        for sym in score.index:
            d = sym_data[sym]
            pos = d["close"].index.get_loc(date)
            r = _future_return(d["close"].index, d["close"], pos, horizon)
            if r is not None:
                fr[sym] = r
        common = score.index.intersection(pd.Series(fr).index)
        if len(common) < 10:
            continue
        ic, _ = spearmanr(score.loc[common], pd.Series(fr).loc[common])
        if not np.isnan(ic):
            records.append({"date": date, "ic": ic})
    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame(columns=["ic"])


def ic_summary(ic_series: pd.Series) -> dict:
    """
    由 IC 序列计算汇总指标。
    """
    if ic_series.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "win_rate": np.nan, "t_stat": np.nan, "n": 0}
    n = len(ic_series)
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    win_rate = (ic_series > 0).mean()
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else np.nan
    return {"ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
            "win_rate": win_rate, "t_stat": t_stat, "n": n}


def interpret(summary: dict) -> str:
    """给出因子是否有效的文字判断。"""
    ic = summary.get("ic_mean", np.nan)
    icir = summary.get("icir", np.nan)
    t = summary.get("t_stat", np.nan)
    if np.isnan(ic):
        return "样本不足，无法判断"
    strength = "弱" if abs(ic) < 0.03 else ("中" if abs(ic) < 0.05 else "强")
    stable = "稳定" if (not np.isnan(icir) and abs(icir) > 0.5) else "不稳定"
    sig = "显著" if (not np.isnan(t) and abs(t) > 2) else "不显著"
    direction = "正向" if ic > 0 else "反向"
    return f"{direction}、预测力{strength}、{stable}、统计{sig}"
