"""逐日动态训练池：按每个采样日的市值/流动性/上市时长筛选 ML 训练的样本域。

背景（2026-10-30，待办 #64 / 计划§八 M6）：现有 ML 训练池 = 选股池 = csi1000 的静态
成分股名单。M6 的机制假设是"训练域下沉到小市值区间"，要求训练样本**逐日**由当日可观测
的市值与流动性决定——用固定名单会让 2025 年的窗口仍在训练 2019 年那批如今已是中大盘的
股票，命题自动失效。

设计要点：
- **point-in-time**：每个采样日只用该日及之前的已公告数据。市值取当日或之前最近一日的
  `market_cap`，且该条不得超过 `max_stale_days` 个自然日（超期视为当日无市值）；成交额取
  当日往前 `amt_window` 个交易日的 `volume × close` 均值；上市时长取该标的在该日的可用
  交易日数。
- **数据边界**：市值数据源最早 2018-01-02。早于该日的采样日，全市场无任何市值可用，
  此时**不施加池过滤**（返回 None，调用方按"全池"处理）。这是数据可得性所迫的退化，
  已在预登记第八节写死，不是设计选择。
- **确定性**：所有排序用 (市值, symbol) 双键，避免同市值并列时顺序随哈希变化（§10 教训）。

纯函数式：不写任何缓存，不拉网络（数据由调用方以 dict 传入）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_pool_selector(market_caps: dict[str, pd.DataFrame],
                        price_dfs: dict[str, pd.DataFrame],
                        cfg: dict, fallback_symbols=None):
    """构造逐日池选择器。

    market_caps: {symbol: DataFrame[date, market_cap, ...]}——来自 get_valuation，
        已按公告日对齐（日频，无未来函数）。
    price_dfs: {symbol: DataFrame[date, close, volume, ...]}——训练段日线。
    cfg: 池配置块，字段见 parse_pool_cfg。
    fallback_symbols: 该日无任何可用市值时退回的符号集合（= **基线训练域**，即选股池
        universe 的成分股）。不传则退回当日全部候选——那会让早于市值数据起点的窗口
        训练域膨胀成全市场，与"退化=不施加过滤"的字面语义不符，详见下方注释。

    返回 callable(date, candidates) -> set[str] | None
        candidates 为当日有行情的候选符号集合；返回该日入池的符号集合，
        fallback 为 None 时返回 None 表示"不施加过滤"（调用方按全部候选训练）。
    """
    p = parse_pool_cfg(cfg)
    if p is None:
        return lambda date, candidates: None
    # 退化集统一为 fallback_symbols（缺省 None = 不施加过滤，调用方自行决定）
    fallback = None if fallback_symbols is None else set(fallback_symbols)

    # 预索引：{symbol: Series(index=date, value=market_cap)}，只保留有限正值
    mc_series: dict[str, pd.Series] = {}
    for sym, df in market_caps.items():
        if df is None or df.empty or "market_cap" not in df.columns:
            continue
        d = df[["date", "market_cap"]].dropna()
        if d.empty:
            continue
        m = d["market_cap"].astype(float)
        keep = np.isfinite(m) & (m > 0)
        d, m = d[keep], m[keep]
        if d.empty:
            continue
        mc_series[sym] = pd.Series(m.values, index=pd.to_datetime(d["date"].values)).sort_index()
    if not mc_series:
        return lambda date, candidates: None

    # 预索引：{symbol: Series(index=date, value=成交额)} + 累计交易日序号（上市时长代理）
    # 成交额 = volume × close（后复权无关：qfq 的 close 与 volume 同比例反向缩放时乘积
    # 不变；本仓库 qfq 只调价不调量，乘积即真实成交额量级）。
    amt_series: dict[str, pd.Series] = {}
    for sym, df in price_dfs.items():
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        if not {"date", "close", "volume"} <= cols:
            continue
        d = df[["date", "close", "volume"]].dropna()
        if d.empty:
            continue
        amt = pd.Series((d["volume"].astype(float) * d["close"].astype(float)).values,
                        index=pd.to_datetime(d["date"].values)).sort_index()
        amt_series[sym] = amt

    w = int(p["amt_window"])
    mc_gate = float(p["min_market_cap"]) if p["min_market_cap"] is not None else 0.0
    # 市值陈旧度上限（自然日）。searchsorted 取 <= ts 的最近一条，若该条已很久没有更新
    # （标的停牌/序列提前结束），继续用它筛池等于用过期市值定位——不是未来函数，但也不是
    # "当日可观测"。超过上限视为该标的当日无市值。15 天覆盖春节/国庆长假（最长连休约 9 天）。
    max_stale_days = int(p["max_stale_days"])

    def _select(date, candidates) -> set[str] | None:
        ts = pd.Timestamp(date)
        rows = []          # (market_cap, symbol)
        any_mc = False     # 该日全市场是否有任何**新鲜**的可用市值
        for sym in candidates:
            s = mc_series.get(sym)
            if s is None:
                continue
            # 取 <= ts 的最近一条（point-in-time，绝不看未来）
            pos = int(s.index.searchsorted(ts, side="right")) - 1
            if pos < 0:
                continue
            if (ts - s.index[pos]).days > max_stale_days:
                continue          # 市值过期，视为当日不可观测
            any_mc = True
            mc = float(s.iloc[pos])
            if not np.isfinite(mc) or mc <= 0 or mc < mc_gate:
                continue
            # 流动性：当日往前 w 个交易日均成交额
            a = amt_series.get(sym)
            if a is not None:
                apos = int(a.index.searchsorted(ts, side="right"))
                if apos < int(p["min_listed_days"]):
                    continue          # 上市不足（用可用交易日数作代理）
                seg = a.iloc[max(0, apos - w):apos]
                if seg.empty:
                    continue
                mean_amt = float(np.nanmean(seg.values))
                if not np.isfinite(mean_amt) or mean_amt < float(p["min_amount"]):
                    continue
            rows.append((mc, sym))
        if not any_mc:
            # 该日全市场无任何新鲜市值（早于数据起点的窗口）。退回 fallback：
            # 「不施加池过滤」在**本实验**里必须解释为"按基线训练域训练"——基线训练域是
            # 选股池 universe 的成分股，不是当日全部候选。若退回全候选，前几窗训练域会从
            # 1000 只膨胀到 5000+ 只，那既不是基线、也不是小市值池，三场景与基线的差异
            # 将混入"样本量被动扩大"这一未登记因素。预登记第八节(1) 已按此语义写死。
            if fallback is not None:
                return {s for s in candidates if s in fallback}
            return None
        # 市值升序；同市值按 symbol 定序（§10 确定性）
        rows.sort(key=lambda x: (x[0], x[1]))
        return {sym for _, sym in rows[: int(p["top_n"])]}

    return _select


def parse_pool_cfg(cfg: dict | None) -> dict | None:
    """解析并校验池配置；关闭/非法返回 None（调用方按现行路径处理，零改动）。"""
    if not cfg or not cfg.get("enabled"):
        return None
    top_n = int(cfg.get("top_n", 0) or 0)
    if top_n <= 0:
        return None
    return {
        "top_n": top_n,
        "min_amount": float(cfg.get("min_amount", 3e7)),
        "min_listed_days": int(cfg.get("min_listed_days", 120)),
        "min_market_cap": cfg.get("min_market_cap"),
        "amt_window": int(cfg.get("amt_window", 20)),
        "max_stale_days": int(cfg.get("max_stale_days", 15)),
    }
