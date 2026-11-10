"""
组合回测（多因子选股）。

逻辑：
- 在每个调仓日，对所有成分股算因子打分、选股、调仓
- 非调仓日保持持仓（每日可能调仓，但换手控制约束实际换手）
- 按等权持仓，跟踪组合净值
- 与基准（成分股等权指数）对比，输出超额收益

这是"截面选股"回测，和 backtest/runner.py 的"单标的择时"回测不同。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jq_framework.data.fetchers.fetcher import (get_index_constituents, get_batch_daily, get_index_daily,
                          get_stock_industry_map, get_index_real_daily)
from jq_framework.data.fetchers.fetcher_fundamental import build_fundamental_data
from jq_framework.factors.composite import load_factor, composite_score, select_portfolio, compute_turnover
from jq_framework.risk.timing import MarketTimer
from jq_framework.backtest.limits import is_tradeable
from jq_framework.backtest.intraday_filter import filter_by_volume, filter_by_open_strength, filter_by_amplitude


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def run_portfolio_backtest(cfg: dict, universe: str | None = None,
                           verbose: bool = True,
                           valid_min: dict[str, pd.DataFrame] | None = None) -> dict:
    """
    运行多因子组合回测。

    cfg: 配置
    universe: 股票池指数名（覆盖配置）
    valid_min: 分钟级数据 {symbol: df}，供执行层 open_strength 过滤用。None=不启用第二因子过滤。
    返回: 回测结果字典
    """
    mf = cfg["multifactor"]
    universe = universe or mf["universe"]
    bt = mf["backtest"]

    # 1. 获取成分股 + 基准成分股
    if verbose:
        print(f"获取 {universe} 成分股 ...")
    syms = get_index_constituents(universe)
    bench_name = mf.get("benchmark", "csi300")
    bench_syms = get_index_constituents(bench_name)

    # 2. 批量拉日线
    if verbose:
        print(f"拉取 {len(syms)} 只股票日线 ...")
    # 键序确定性（2026-10-13，docs/quant_system/已知口径偏差.md §10）：set 迭代序依赖
    # PYTHONHASHSEED 逐进程随机，sorted 切断该随机源（static 引擎实测对符号序不敏感，
    # 见 /tmp 验证对 A==B；此修为防御性统一，与 fetcher.get_batch_daily 同步）。
    all_syms = sorted(set(syms + bench_syms))
    price_dfs = get_batch_daily(all_syms, start=bt["start"], progress=verbose)

    # 3. 对齐日期轴：取所有股票共有的交易日
    valid = {s: df for s, df in price_dfs.items() if not df.empty}
    if verbose:
        print(f"有效股票: {len(valid)}/{len(all_syms)}")
    all_dates = sorted(set().union(*[set(df["date"]) for df in valid.values()]))
    dates = [d for d in all_dates if pd.Timestamp(bt["start"]) <= d <= pd.Timestamp(bt["end"])]
    if verbose:
        print(f"回测区间: {dates[0].date()} -> {dates[-1].date()}, 共 {len(dates)} 个交易日")

    # 4. 加载因子
    factors = [(load_factor(spec), spec.get("weight", 1.0))
               for spec in mf["factors"].values()]

    # 行业中性化（可选）：拉申万一级行业映射，注入各因子
    industry_neutral = mf.get("industry_neutral", False)
    industry_map = None
    if industry_neutral:
        if verbose:
            print("拉取申万一级行业映射（用于行业中性化）...")
        try:
            industry_map = get_stock_industry_map()
            for factor, _ in factors:
                factor.industry_neutral = True
                factor.set_industry_map(industry_map)
            if verbose:
                print(f"  行业映射: {len(industry_map)} 只股票")
        except Exception as e:
            if verbose:
                print(f"  行业映射拉取失败，跳过中性化: {e}")
            industry_neutral = False

    # 基本面因子数据注入（若有任何 is_fundamental 因子）：拉估值+财务并注入
    has_fund = any(getattr(f, "is_fundamental", False) for f, _ in factors)
    if has_fund:
        if verbose:
            print("拉取基本面数据（估值+财务）注入基本面因子 ...")
        try:
            fund_data = build_fundamental_data(all_syms, progress=verbose)
            for factor, _ in factors:
                factor.set_fundamental_data(fund_data)
            if verbose:
                ok = sum(1 for v in fund_data.values() if v is not None and not v.empty)
                print(f"  基本面数据: {ok}/{len(all_syms)} 只股票就绪")
        except Exception as e:
            if verbose:
                print(f"  基本面数据拉取失败，基本面因子将返回 None: {e}")

    # 5. 逐日回测（次日开盘成交 + 风控）
    top_n = mf["select"]["top_n"]
    buffer = mf["select"].get("turnover_buffer", 0)
    # 两层合成配置（None/flat=逐字复现旧 flat 逻辑；twolevel=族内正交+族间风险预算）
    combine_cfg = {k: mf.get(k) for k in ("combine", "families", "orthogonalize")}
    # 交易成本（单边费率）
    cost_cfg = bt.get("cost", {})
    cost_buy = cost_cfg.get("commission", 0.00025) + cost_cfg.get("slippage", 0.001)
    cost_sell = (cost_cfg.get("commission", 0.00025) + cost_cfg.get("slippage", 0.001)
                 + cost_cfg.get("stamp_duty", 0.0005))

    # 风控管理器
    from jq_framework.risk.manager import RiskManager
    risk_cfg = mf.get("risk", {})
    risk = RiskManager(
        stop_loss=risk_cfg.get("stop_loss", 0.10),
        take_profit=risk_cfg.get("take_profit", 0.30),
        target_vol=risk_cfg.get("target_vol", 0.15),
        lookback=risk_cfg.get("lookback", 20),
        max_leverage=risk_cfg.get("max_leverage", 1.5),
    )

    # 大盘择时（RSRS 仓位乘数，默认关闭→no-op 恒 1.0 不破回归门）
    market_timer = MarketTimer(mf.get("market_timing", {}))

    # 持仓状态：{symbol: {"weight":权重, "entry_price":买入价, "entry_date":买入日}}
    holdings: dict[str, dict] = {}
    pending_trade: list[str] | None = None  # T日产生、T+1开盘执行的信号
    days_since_rebal = 0  # 距上次重算信号的天数（rebalance_days 控制换仓频率）
    rebalance_days = max(1, int(mf["select"].get("rebalance_days", 1)))
    nav = 1.0
    bench_nav = 1.0
    nav_series = []
    bench_series = []
    turnovers = []
    total_cost = 0.0
    stop_count = {"stop_loss": 0, "take_profit": 0}
    vol_filtered = 0     # 放量执行层过滤净剔除数（诊断）
    vol_backfilled = 0   # 放量过滤兜底补回数（诊断）
    os_filtered = 0      # 开盘强度执行层过滤净剔除数（诊断，第二日内因子）
    os_backfilled = 0    # 开盘强度过滤兜底补回数（诊断）
    amp_filtered = 0     # 振幅执行层过滤净剔除数（诊断，第三日内因子，IC=-0.108 显著为负）
    amp_backfilled = 0   # 振幅过滤兜底补回数（诊断）

    # 预构建基准日收益（等权）
    bench_df = get_index_daily(bench_syms, {s: valid.get(s, pd.DataFrame()) for s in bench_syms},
                               start=bt["start"])
    bench_ret = bench_df.set_index("date")["nav"].pct_change() if not bench_df.empty else pd.Series(dtype=float)

    # 大盘择时乘数预算：用真实指数日线（需 high/low，合成基准不可用）一次性算好全部日期
    # 的 RSRS 仓位乘数 → 查表字典。默认关闭时 no-op 不拉数据。start 早于回测起点以保证
    # z-score 标准化窗口 M=600 有足够历史预热。起点取 max(回测起点前推 M×2 日历日, 2009-01-01)：
    # M≈600交易日≈2.4年，前推留足 RSRS 的 N+M 滚动回归预热余量；2009 为 csi300 真实指数可拉最早。
    if market_timer.enabled and dates:
        warmup_days = market_timer.m * 2
        idx_start = (dates[0] - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        idx_start = max(idx_start, "2009-01-01")
        idx_df = get_index_real_daily(bench_name, start=idx_start,
                                      end=dates[-1].strftime("%Y-%m-%d"))
        market_timer.build(idx_df)
        if verbose:
            cov = market_timer.coverage()
            print(f"  RSRS 大盘择时: 覆盖{cov[0]}天 均仓{cov[1]:.2f} 满仓{cov[3]*100:.0f}%")

    def get_open(sym, date):
        df = valid.get(sym)
        if df is None:
            return None
        row = df[df["date"] == date]
        return row["open"].iloc[0] if not row.empty else None

    for i, date in enumerate(dates):
        day_cost = 0.0

        # --- (a) 开盘前先检查止盈止损（基于昨日持仓的历史买入价） ---
        # 必须在调仓前检查，否则保留持仓的 entry_price 会被重置成今天开盘价，导致永不触发
        if holdings:
            to_sell = risk.filter_stops(holdings, date, valid)
            to_sell = [s for s in to_sell if is_tradeable(s, date, valid, "sell")]  # 跌停开盘锁死：持有次日重试
            if to_sell:
                for s in to_sell:
                    reason = risk.check_stop(get_open(s, date), holdings[s]["entry_price"])
                    stop_count[reason] = stop_count.get(reason, 0) + 1
                sell_w = sum(holdings[s]["weight"] for s in to_sell)
                day_cost += sell_w * cost_sell
                total_cost += sell_w * cost_sell
                holdings = {s: info for s, info in holdings.items() if s not in to_sell}
                # 释放的权重按比例放大到剩余持仓（保持满仓）
                if holdings:
                    scale = 1.0 / sum(info["weight"] for info in holdings.values())
                    for s in holdings:
                        holdings[s]["weight"] *= scale

        # --- (b) 执行昨日产生的待成交信号：次日开盘价成交 ---
        if pending_trade is not None:
            new_syms = pending_trade
            target_w = 1.0 / len(new_syms) if new_syms else 0
            new_weights = {s: target_w for s in new_syms}
            # 波动率加权：高波动个股降权，低波动个股加权，总仓位不变
            if risk.target_vol is not None and new_syms:
                mults = {s: risk.vol_weight_multiplier(s, date, valid) for s in new_syms}
                total_mult = sum(mults.values()) or 1.0
                avg_mult = total_mult / len(new_syms)
                new_weights = {s: target_w * mults[s] / avg_mult for s in new_syms}
            # 大盘择时仓位乘数（RSRS）：对整个组合统一缩放总仓位（满仓→底仓）。
            # 在个股波动率加权（相对权重层）之后、交易成本计算之前施加（总仓位层），
            # 串联非并联。默认关闭时 get_multiplier 恒 1.0 无开销。
            if market_timer.enabled and new_syms:
                market_mult = market_timer.get_multiplier(date)
                new_weights = {s: w * market_mult for s, w in new_weights.items()}
            # 交易成本：买入新增 + 卖出退出 + 再平衡
            old_w = {s: info["weight"] for s, info in holdings.items()}
            buy_amt = sum(new_weights[s] for s in new_syms if s not in old_w)
            sell_amt = sum(old_w[s] for s in old_w if s not in new_weights)
            rebal = sum(abs(new_weights.get(s, 0) - old_w.get(s, 0))
                        for s in set(new_weights) | set(old_w))
            trade_cost = (buy_amt * cost_buy + sell_amt * cost_sell
                          + rebal * (cost_buy + cost_sell) / 2) / 2
            day_cost += trade_cost
            total_cost += trade_cost
            # 建立新持仓：保留旧持仓的 entry_price/entry_date，新增用今天开盘价
            new_holdings = {}
            for s in new_syms:
                op = get_open(s, date)
                if op is None:
                    continue
                if not is_tradeable(s, date, valid, "buy"):  # 涨停开盘锁死：跳过
                    continue
                if s in holdings:
                    # 保留持仓：买入价和买入日不变
                    new_holdings[s] = {"weight": new_weights[s],
                                       "entry_price": holdings[s]["entry_price"],
                                       "entry_date": holdings[s]["entry_date"]}
                else:
                    new_holdings[s] = {"weight": new_weights[s],
                                       "entry_price": op, "entry_date": date}
            holdings = new_holdings
            pending_trade = None

        # --- (c) 计算当日组合收益（按持仓权重，各股当日收益） ---
        port_ret = 0.0
        if holdings:
            for s, info in holdings.items():
                df = valid.get(s)
                if df is None:
                    continue
                row = df[df["date"] == date]
                if not row.empty and len(df[df["date"] <= date]) >= 2:
                    prev_close = df[df["date"] < date]["close"].iloc[-1]
                    port_ret += info["weight"] * (row["close"].iloc[0] / prev_close - 1)

        nav *= (1 + port_ret) * (1 - day_cost)
        br = bench_ret.get(date, 0.0) if not bench_ret.empty else 0.0
        if br is None or (isinstance(br, float) and np.isnan(br)):
            br = 0.0
        bench_nav *= (1 + br)
        nav_series.append({"date": date, "nav": nav})
        bench_series.append({"date": date, "nav": bench_nav})

        # --- (d) 当日收盘产生信号（次日开盘执行，rebalance_days 控制换仓频率）---
        if days_since_rebal >= rebalance_days or not holdings:
            current = {s: df[df["date"] <= date] for s, df in valid.items()
                       if (df["date"] <= date).any()}
            # 打分域口径（2026-09-24 审计#11，与 walk_forward 同修）：valid 含 benchmark
            # 成分股（为算基准收益），但选股打分域恒为 universe——不过滤时 universe 与
            # benchmark 不相交的池（csi500/csi1000 + csi300 基准）会选出池外大盘股。
            current = {s: d for s, d in current.items() if s in set(syms)}
            score = composite_score(current, factors, combine_cfg=combine_cfg)
            new_syms = select_portfolio(score, top_n, list(holdings.keys()), buffer)
            # 执行层过滤：链式 volume_surge → open_strength（T 日收盘信号 → T+1 开盘成交）。
            # 无归因 hook，直接改 new_syms；turnovers 在过滤后算，反映过滤后实际换手。
            ifilt = mf.get("intraday_filter", {})
            vs_cfg = ifilt.get("volume_surge", {}) if ifilt.get("enabled") else {}
            if vs_cfg.get("enabled", True) and vs_cfg:
                new_syms, vstats = filter_by_volume(
                    new_syms, date, valid, top_n,
                    threshold=vs_cfg.get("threshold", 1.0),
                    window=vs_cfg.get("window", 20),
                    min_keep_ratio=vs_cfg.get("min_keep_ratio", 0.8),
                )
                vol_filtered += vstats["filtered"]
                vol_backfilled += vstats["backfilled"]
            # 第二因子（链式）：open_strength 用分钟 valid_min
            os_cfg = ifilt.get("open_strength", {}) if ifilt.get("enabled") else {}
            if os_cfg.get("enabled", False) and os_cfg and valid_min is not None:
                new_syms, ostats = filter_by_open_strength(
                    new_syms, date, valid_min, top_n,
                    threshold=os_cfg.get("threshold", 0.0),
                    window=os_cfg.get("window", 48),
                    min_keep_ratio=os_cfg.get("min_keep_ratio", 0.8),
                )
                os_filtered += ostats["filtered"]
                os_backfilled += ostats["backfilled"]
            # 第三因子（链式）：amplitude 用分钟 valid_min
            amp_cfg = ifilt.get("amplitude", {}) if ifilt.get("enabled") else {}
            if amp_cfg.get("enabled", False) and amp_cfg and valid_min is not None:
                new_syms, astats = filter_by_amplitude(
                    new_syms, date, valid_min, top_n,
                    threshold=amp_cfg.get("threshold", 0.0),
                    window=amp_cfg.get("window", 48),
                    min_keep_ratio=amp_cfg.get("min_keep_ratio", 0.8),
                )
                amp_filtered += astats["filtered"]
                amp_backfilled += astats["backfilled"]
            if i > 0:
                turnovers.append(compute_turnover(new_syms, list(holdings.keys())))
            pending_trade = new_syms
            days_since_rebal = 0
        else:
            pending_trade = None  # 非调仓日不换仓，保持现持仓
            days_since_rebal += 1

    # 6. 绩效统计
    result = _stats(nav_series, bench_series, turnovers, verbose)
    result["final_holdings"] = list(holdings.keys())
    result["total_cost"] = total_cost
    result["stop_count"] = stop_count
    result["vol_filtered"] = vol_filtered
    result["vol_backfilled"] = vol_backfilled
    result["os_filtered"] = os_filtered
    result["os_backfilled"] = os_backfilled
    result["amp_filtered"] = amp_filtered
    result["amp_backfilled"] = amp_backfilled
    result["industry_neutral"] = industry_neutral
    if verbose:
        print(f"  累计交易成本 : {total_cost:.2%} (占初始资金)")
        print(f"  止盈止损触发 : 止损{stop_count['stop_loss']}次 止盈{stop_count['take_profit']}次")
        if vol_filtered or vol_backfilled:
            print(f"  放量过滤     : 净剔除{vol_filtered}次 兜底补回{vol_backfilled}次")
        if os_filtered or os_backfilled:
            print(f"  开盘强度过滤 : 净剔除{os_filtered}次 兜底补回{os_backfilled}次")
        if amp_filtered or amp_backfilled:
            print(f"  振幅过滤     : 净剔除{amp_filtered}次 兜底补回{amp_backfilled}次")
        print(f"  行业中性化   : {'开启' if industry_neutral else '关闭'}")
    return result


def _stats(nav_series, bench_series, turnovers, verbose):
    nav_df = pd.DataFrame(nav_series).set_index("date")
    bench_df = pd.DataFrame(bench_series).set_index("date")

    daily_ret = nav_df["nav"].pct_change().dropna()
    bench_ret = bench_df["nav"].pct_change().dropna()

    n_years = len(nav_df) / 252
    total_ret = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    bench_total = bench_df["nav"].iloc[-1] / bench_df["nav"].iloc[0] - 1
    bench_ann = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0

    # 最大回撤
    peak = nav_df["nav"].cummax()
    dd = (nav_df["nav"] / peak - 1)
    max_dd = dd.min()

    # 夏普（无风险2%）
    sharpe = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    bench_sharpe = (bench_ret.mean() * 252 - 0.02) / (bench_ret.std() * np.sqrt(252)) if bench_ret.std() > 0 else 0

    # 超额
    excess = ann_ret - bench_ann
    avg_turnover = np.mean(turnovers) if turnovers else 0

    result = {
        "total_return": total_ret, "ann_return": ann_ret,
        "bench_total": bench_total, "bench_ann": bench_ann,
        "excess_annual": excess, "max_drawdown": max_dd,
        "sharpe": sharpe, "bench_sharpe": bench_sharpe,
        "avg_turnover": avg_turnover, "n_days": len(nav_df),
    }

    if verbose:
        print("\n" + "=" * 50)
        print("  多因子组合回测绩效")
        print("=" * 50)
        print(f"  回测天数     : {result['n_days']}")
        print(f"  累计收益     : 组合 {total_ret:+.2%}  | 基准 {bench_total:+.2%}")
        print(f"  年化收益     : 组合 {ann_ret:+.2%}  | 基准 {bench_ann:+.2%}")
        print(f"  超额年化     : {excess:+.2%}")
        print(f"  最大回撤     : {max_dd:.2%}")
        print(f"  夏普比率     : 组合 {sharpe:.2f}  | 基准 {bench_sharpe:.2f}")
        print(f"  平均换手率   : {avg_turnover:.2%} (单边/日)")
        print("=" * 50)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--universe", default=None, help="股票池: csi300/csi500/csi1000")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_portfolio_backtest(cfg, universe=args.universe)


if __name__ == "__main__":
    main()
