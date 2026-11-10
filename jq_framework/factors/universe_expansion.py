"""扩充可投资域 U(K, t)：point-in-time 构造逐日可投资池。

口径来源：《验证强化与实验防重复执行计划》第八节 M7 预登记（2026-11-02，写于读数之前）。

    U(K, t) = csi1000 在册成员(t)  ∪  过门槛的市值最小 K 只非成员(t)

三条硬约束，与预登记逐条对应：
1. **并集语义**：成分股侧不加任何门槛、不做市值截断。基线 csi1000 回测的硬门默认关闭，
   成员从不过滤；这里若对成分股施加上市/流动性门槛，U 与基线的差异会混入"剔除"这一
   未登记因素。**门槛只作用于新增的非成员**。
2. **point-in-time**：每个交易日只用该日及之前已可观测的市值。市值取 <= t 的最近一条，
   超过 max_stale_days 个自然日视为当日不可观测（停牌/序列中断）。成交量同理只用
   t 日及之前的窗口。**绝不使用任何 t 之后的信息**。
3. **确定性**：所有排序用 (市值, symbol) 双键。同市值并列时顺序不得随哈希种子变化
   （已知口径偏差 §10 的教训：set 迭代序曾造成 ML 训练跨 run 摆动 17pp）。

与 factors/dynamic_pool.py 的关系：那个模块实现的是 M6 的"市值最小 K 只"训练池
（语义是**替换**训练域）。本模块实现的是**并集扩充**（语义是**新增**），命题不同，
故独立成文件而不改动 M6 的既有实现——M6 已收口，其行为须逐字保留。

实现方式：向量化面板。逐日 × 逐股调用选择器在 2000 日 × 5000 股规模上是千万次查询，
不可接受；改为构造 dates × symbols 的市值/成交额面板后按行做排名，内存约 80MB 量级。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parse_expansion_cfg(cfg: dict | None) -> dict | None:
    """解析扩充池配置；关闭/非法返回 None（调用方走现行路径，零改动）。"""
    if not cfg or not cfg.get("enabled"):
        return None
    top_n = int(cfg.get("top_n", 0) or 0)
    if top_n <= 0:
        return None
    return {
        "top_n": top_n,
        "min_amount": float(cfg.get("min_amount", 3e7)),
        "min_listed_days": int(cfg.get("min_listed_days", 120)),
        "amt_window": int(cfg.get("amt_window", 20)),
        "max_stale_days": int(cfg.get("max_stale_days", 15)),
    }


def _mc_panel(market_caps: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
              symbols: list[str], max_stale_days: int) -> pd.DataFrame:
    """构造 dates × symbols 的市值面板（point-in-time，含陈旧度截断）。

    对每个标的，把其市值序列按日期 reindex 到交易日，只做**向前**填充，且填充跨度
    不得超过 max_stale_days 个自然日——超出的位置置 NaN，表示"当日不可观测"。
    绝不做向后填充（那会引入未来信息）。
    """
    out = pd.DataFrame(np.nan, index=dates, columns=symbols, dtype="float64")
    if not symbols:
        return out
    for sym in symbols:
        df = market_caps.get(sym)
        if df is None or df.empty or "market_cap" not in df.columns:
            continue
        d = df[["date", "market_cap"]].dropna()
        if d.empty:
            continue
        s = pd.Series(d["market_cap"].astype(float).values,
                      index=pd.to_datetime(d["date"].values)).sort_index()
        s = s[np.isfinite(s.values) & (s.values > 0)]
        if s.empty:
            continue
        # asof 到交易日：每行取 <= 该交易日的最近一条
        idx = pd.Index(dates)
        pos = s.index.searchsorted(idx, side="right") - 1
        vals = np.where(pos >= 0, s.values[np.clip(pos, 0, len(s) - 1)], np.nan)
        # 陈旧度：交易日与其对应观测日相差 > max_stale_days 自然日 → NaN
        obs_dates = np.where(pos >= 0, s.index.values[np.clip(pos, 0, len(s) - 1)],
                             np.datetime64("NaT"))
        gap_days = (dates.values - obs_dates) / np.timedelta64(1, "D")
        stale = ~np.isfinite(gap_days) | (gap_days > max_stale_days)
        vals[stale] = np.nan
        out[sym] = vals
    return out


def _gated_panel(price_dfs: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                 symbols: list[str], min_amount: float, min_listed_days: int,
                 amt_window: int) -> pd.DataFrame:
    """构造 dates × symbols 的"过流动性/上市时长门槛"布尔面板。

    流动性：当日往前 amt_window 个交易日的日均成交额（volume × close）≥ min_amount。
    上市时长：该标的当日往前已有 ≥ min_listed_days 个交易日的历史（用可用交易日数作代理）。
    两者都只用 t 日及之前的数据。
    """
    out = pd.DataFrame(False, index=dates, columns=symbols, dtype=bool)
    if not symbols:
        return out
    for sym in symbols:
        df = price_dfs.get(sym)
        if df is None or df.empty:
            continue
        if not {"date", "close", "volume"} <= set(df.columns):
            continue
        d = df[["date", "close", "volume"]].dropna()
        if d.empty:
            continue
        dd = pd.to_datetime(d["date"].values)
        amt = pd.Series((d["volume"].astype(float).values * d["close"].astype(float).values),
                        index=dd).sort_index()
        listed = pd.Series(np.arange(1, len(amt) + 1), index=amt.index)
        # 用当日往前 w 日均值：rolling 只看历史，天然无未来
        roll = amt.rolling(amt_window, min_periods=max(1, amt_window // 2)).mean()
        ok_amt = (roll >= min_amount)
        ok_list = (listed >= min_listed_days)
        ok = (ok_amt & ok_list)
        # 对齐到交易日：取 <= 该交易日的最近一条判定结果
        pos = ok.index.searchsorted(dates, side="right") - 1
        vals = np.where(pos >= 0, ok.values[np.clip(pos, 0, len(ok) - 1)], False)
        out[sym] = vals
    return out


def build_universe_panel(market_caps: dict[str, pd.DataFrame],
                         price_dfs: dict[str, pd.DataFrame],
                         members: set[str],
                         dates: list,
                         cfg: dict) -> dict[pd.Timestamp, set[str]]:
    """构造逐日 U(K, t)，返回 {date: set[symbol]}。

    market_caps: {symbol: DataFrame[date, market_cap]}——来自 get_valuation。
    price_dfs:   {symbol: DataFrame[date, close, volume]}——日线（须覆盖训练起点）。
    members:     csi1000 在册成员集合（静态名单，与基线同源）。
    dates:       需要构造的交易日序列。
    cfg:         扩充池配置块（字段见 parse_expansion_cfg）。

    返回的集合在每个日期都包含当日**有行情**的全部成分股，加上过门槛的市值最小 K 只非成员。
    """
    p = parse_expansion_cfg(cfg)
    if p is None:
        return {}

    dts = pd.DatetimeIndex(sorted(pd.to_datetime(dates)))
    if len(dts) == 0:
        return {}
    members = set(members)
    # 候选全集：取市值与日线两边都有的标的（成分股即使无市值也不能丢——见并集语义）
    cand = sorted(set(price_dfs) | (set(market_caps) & set(price_dfs)))
    cand = [s for s in cand if price_dfs.get(s) is not None and not price_dfs[s].empty]

    mc = _mc_panel(market_caps, dts, cand, int(p["max_stale_days"]))
    gate = _gated_panel(price_dfs, dts, cand, float(p["min_amount"]),
                        int(p["min_listed_days"]), int(p["amt_window"]))

    # 当日有行情的标的（收盘价有限）：用于新增非成员的入池判定——买入需要当日可交易。
    traded = pd.DataFrame(False, index=dts, columns=cand, dtype=bool)
    for s in cand:
        dd = pd.to_datetime(price_dfs[s]["date"].values)
        traded.loc[dts.isin(dd), s] = True

    # 截至当日已有历史数据的标的：用于成分股入池判定。
    # **必须与基线的选股域判据逐字一致**——基线是
    #     current = {s: df[df["date"] <= date] for s, df in valid.items()
    #                if (df["date"] <= date).any()}
    # 即"截至当日有任意历史数据"，**不要求当日有行情**。若这里改用 traded（当日有行情），
    # 停牌中的成分股会被剔出 U，U 就不再是 csi1000 的超集，违反预登记§二"U 与基线的差
    # 纯粹来自新增、绝不来自剔除"的保证——那会让 M7 的读数混入"剔除停牌股"这一未登记因素。
    has_hist = pd.DataFrame(False, index=dts, columns=cand, dtype=bool)
    for s in cand:
        dd = pd.to_datetime(price_dfs[s]["date"].values)
        if len(dd):
            has_hist.loc[dts >= dd.min(), s] = True

    member_cols = [s for s in cand if s in members]
    nonmember_cols = [s for s in cand if s not in members]
    K = int(p["top_n"])

    out: dict[pd.Timestamp, set[str]] = {}
    mc_vals = mc[nonmember_cols].values if nonmember_cols else np.empty((len(dts), 0))
    gate_vals = gate[nonmember_cols].values if nonmember_cols else np.empty((len(dts), 0), bool)
    hh_mem = has_hist[member_cols].values if member_cols else np.empty((len(dts), 0), bool)
    tr_non = traded[nonmember_cols].values if nonmember_cols else np.empty((len(dts), 0), bool)

    for i, ts in enumerate(dts):
        # 成分股：截至当日有历史数据即入池（不施加任何门槛，与基线口径逐字一致）
        picked = {member_cols[j] for j in np.flatnonzero(hh_mem[i])} if member_cols else set()
        if nonmember_cols:
            ok = gate_vals[i] & tr_non[i] & np.isfinite(mc_vals[i]) & (mc_vals[i] > 0)
            idx = np.flatnonzero(ok)
            if len(idx) > K:
                # 市值升序；同市值按 symbol 定序（确定性）
                order = np.lexsort((np.array(nonmember_cols, dtype=object)[idx],
                                    mc_vals[i][idx]))
                idx = idx[order[:K]]
            picked |= {nonmember_cols[j] for j in idx}
        out[ts] = picked
    return out


def panel_to_returns(universe_panel: dict[pd.Timestamp, set[str]],
                     price_dfs: dict[str, pd.DataFrame],
                     start) -> pd.Series:
    """把逐日 U(t) 转成等权基准日收益序列（成员逐日变化）。

    与 fetcher.get_index_daily 同口径（等权、截面均值收益），区别只在**成员按日变化**：
    后者接受静态符号表，本函数每天用当日 U(t) 的成员算截面均值。

    注意：不得用"全部历史出现过的 U 成员的并集"——那会让早期基准混入后期才入池的标的，
    等价于给基准注入它本不该有的成分。
    """
    if not universe_panel:
        return pd.Series(dtype=float)
    # 预计算每只股票的日收益（pct_change 只用前一日，无未来）
    rets: dict[str, pd.Series] = {}
    for sym, df in price_dfs.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
        rets[sym] = s.pct_change()
    if not rets:
        return pd.Series(dtype=float)
    ret_df = pd.DataFrame(rets)

    rows = []
    for ts, syms in universe_panel.items():
        if ts not in ret_df.index:
            continue
        have = [s for s in syms if s in ret_df.columns]
        if not have:
            rows.append((ts, 0.0))
            continue
        v = ret_df.loc[ts, have].values.astype(float)
        v = v[np.isfinite(v)]
        rows.append((ts, float(v.mean()) if len(v) else 0.0))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({d: r for d, r in rows}).sort_index()
    s = s[s.index >= pd.Timestamp(start)]
    return s.fillna(0.0)
