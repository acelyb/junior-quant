"""
每日选股信号（多因子 A 路径）。

作用：跑最新因子打分，输出今日目标持仓清单（新增/剔除/保留）。
与组合回测共用 factors/composite.py 的因子和选股逻辑，保证一致。

合成方式（--weighting 覆盖 / config walk_forward.weighting）：
  static/ic: composite_score 线性合成（原路径）。
  ml:        加载持久化 LightGBM artifact（signals/ml_model_store，路线 b）做非线性合成。
             fail-loud：无 artifact / 因子装配契约不符 / 陈旧超线（ml_serving.max_age_days）
             直接退出进程——旧模型静默打分比没有信号更危险。
  两分支的因子装配统一走 build_production_factors（镜像 walk_forward L666-722：
  行业中性化 + 基本面注入按 config 生效，修复审计#5 信号侧装配断层）。

用法：
  python -m signals.daily_portfolio --config config.yaml               # 按 config weighting
  python -m signals.daily_portfolio --weighting ml                     # 强制 ML 分支
  python -m signals.daily_portfolio --holdings 000009,000021
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from jq_framework.data.fetchers.fetcher import get_index_constituents, get_batch_daily
from jq_framework.factors.composite import composite_score, select_portfolio
from jq_framework.factors.composite_ml import composite_score_ml
from jq_framework.signals.ml_model_store import (
    resolve_serving_universe, build_production_factors, load_model_artifact,
    serving_train_start,
)


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def run_daily_portfolio(cfg: dict, prev_holdings: list[str] | None = None,
                        verbose: bool = True, weighting: str | None = None,
                        read_only: bool = False) -> dict:
    """
    计算今日目标持仓。

    prev_holdings: 当前持仓（用于换手控制和"新增/剔除"对比）。无则视为新建仓。
    weighting: 合成方式覆盖（None=用 config walk_forward.weighting）。
    read_only: True=数据获取走 cache-only（缺数据直接失败，不拉网不写盘）——
        烟测/复现用；日常运行缺省 False（增量更新缓存）。
    返回: {target, score, new, removed, kept, ref_prices, weighting, universe,
           model_meta?, stale_days?, limit_up}
        ref_prices: {symbol: {date, close, open}} 最新交易日参考价。close 作"建议买入
            参考价"（实盘入场价=T+1 开盘价，信号产生时 T 日收盘尚无次日 open，故以 T 日
            收盘近似）；open 一并带出供后续"已持仓止盈止损基准"扩展用，当前报告未展示。
        limit_up: T 日已涨停的 symbol 集合（次日开盘大概率买不进，报告标注）。
    """
    mf = cfg["multifactor"]
    universe = resolve_serving_universe(cfg)  # ml_serving.universe 显式优先（防 config 漂移换池）
    weighting = weighting or mf.get("walk_forward", {}).get("weighting", "static")

    if verbose:
        print(f"拉取 {universe} 成分股日线 ...")
    syms = get_index_constituents(universe, read_only=read_only)
    # 起点必须与训练器逐字相同（serving_train_start）：缓存键含起点，起点不同即读到
    # 另一份刷新节奏独立的文件，信号会在无告警的情况下陈旧（2026-10-29 修：原实现
    # 走默认 2020-01-01，读到停在 10-09 的旧键，而生产键已到 10-28）。
    price_dfs = get_batch_daily(syms, start=serving_train_start(cfg),
                                progress=verbose, read_only=read_only)
    price_dfs = {s: d for s, d in price_dfs.items() if d is not None and not d.empty}

    # 因子装配：与回测同源（load → aug[ml] → 行业中性化 → 基本面注入），两分支共用
    factors = build_production_factors(cfg, syms=list(price_dfs.keys()),
                                       verbose=verbose, read_only=read_only)

    result = {"weighting": weighting, "universe": universe}
    if weighting == "ml":
        # ML 分支：加载持久化 artifact（契约校验 + 新鲜度红线均 fail-loud，见 ml_model_store）
        dates = sorted(set().union(*[set(df["date"]) for df in price_dfs.values()]))
        model, feature_names, meta = load_model_artifact(cfg, available_dates=dates)
        score = composite_score_ml(price_dfs, factors, model, feature_names)
        result["model_meta"] = meta
        result["stale_days"] = sum(1 for d in dates
                                   if pd.Timestamp(d) > pd.Timestamp(meta["data_end"]))
    else:
        # 两层合成配置（None/flat=逐字复现旧 flat 逻辑；twolevel=族内正交+族间风险预算）
        combine_cfg = {k: mf.get(k) for k in ("combine", "families", "orthogonalize")}
        score = composite_score(price_dfs, factors, combine_cfg=combine_cfg)

    # 选股：top_n + 换手缓冲。top_n 优先取 ml_serving.top_n（生产信号口径，与 config
    # 全局 select.top_n 的回测基线值解耦）；buffer 分池解析（镜像回测 buffer_by_universe
    # 优先级，修复审计#5 信号侧分池参数断层——csi1000 生产口径 buf16 而非全局 buf4）
    ms = mf.get("ml_serving", {}) or {}
    sel = mf.get("select", {})
    top_n = ms.get("top_n") or sel.get("top_n", 20)
    buf_by = sel.get("buffer_by_universe") or {}
    buffer = buf_by.get(universe, sel.get("turnover_buffer", 0))

    target = select_portfolio(score, top_n, prev_holdings, buffer)
    prev_set = set(prev_holdings or [])
    new = [s for s in target if s not in prev_set]
    removed = [s for s in (prev_holdings or []) if s not in set(target)]
    kept = [s for s in target if s in prev_set]

    # 最新交易日参考价 + T 日涨停检测（次日开盘买不进的概率高，报告标注）
    # + vol 止损模式下的逐票止损参考价：k×σ×√min_days（σ 与 risk/manager._rolling_vol
    #   同口径——T+1 开盘执行侧用 dates<T+1 即 ≤T 的 lookback 收益，此处用截至 T 的
    #   同一段，估计值与回测首日阈值逐字一致）
    risk_cfg = mf.get("risk", {}) or {}
    vol_mode = risk_cfg.get("stop_loss_mode") == "vol"
    k_vol = float(risk_cfg.get("stop_vol_k", 1.0) or 1.0)
    min_days = int(risk_cfg.get("stop_vol_min_days", 10) or 10)
    lookback = int(risk_cfg.get("lookback", 20) or 20)
    ref_prices: dict[str, dict] = {}
    limit_up: set[str] = set()
    for s, df in price_dfs.items():
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        close = float(last["close"])
        ref_prices[s] = {"date": last["date"], "close": close,
                         "open": float(last["open"])}
        if vol_mode and len(df) >= lookback + 1:
            # 与 _rolling_vol(df, T+1) 同窗：dates < T+1 = 截至 T 的 lookback 段收益（含 T 当段）
            rets = df["close"].pct_change().iloc[-lookback:]
            sigma = float(rets.std())
            if sigma > 0 and pd.notna(sigma):
                ref_prices[s]["est_stop"] = close * (1 - k_vol * sigma * (min_days ** 0.5))
        if len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            if prev_close > 0:
                chg = close / prev_close - 1
                # 主板 10%；创业板(300)/科创板(688) 20%（ST 5% 无名称数据源，忽略）
                thr = 0.195 if s.startswith(("300", "688")) else 0.095
                if chg >= thr:
                    limit_up.add(s)

    result.update({"target": target, "score": score, "new": new, "removed": removed,
                   "kept": kept, "ref_prices": ref_prices, "limit_up": limit_up})
    return result


def format_report(result: dict, universe: str,
                  risk: dict | None = None) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append(f"  多因子选股信号  {datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"  股票池: {universe}")
    if result.get("weighting") == "ml" and result.get("model_meta"):
        m = result["model_meta"]
        lines.append(f"  合成: ML 持久化模型 (ml_h{m['horizon']}_{str(m['data_end']).replace('-', '')}, "
                     f"特征 {m.get('n_features', '?')} 个, 距训练末 {result.get('stale_days', '?')} 个交易日)")
    else:
        lines.append(f"  合成: {result.get('weighting', 'static')} 线性")
    lines.append("=" * 55)

    ref = result.get("ref_prices", {})
    # 风控口径展示。审计#20（2026-09-24 修）：stop_loss:null 合法（None 走默认）。
    def _risk_num(key: str, default: float) -> float | None:
        v = (risk or {}).get(key)
        return float(v) if v is not None else default

    stop_loss = _risk_num("stop_loss", 0.10)
    take_profit = _risk_num("take_profit", 0.30)
    vol_mode = (risk or {}).get("stop_loss_mode", "fixed") == "vol"
    if vol_mode:
        k = _risk_num("stop_vol_k", 1.0)
        min_days = int(_risk_num("stop_vol_min_days", 10) or 10)
        stop_note = f"止损=波动自适应 {k:.1f}×σ×√t（t 下限 {min_days} 日，逐票参考价见下）"
    else:
        stop_note = f"止损=入场价×{1 - stop_loss:.2f}" if stop_loss else "止损=无"

    lines.append(f"\n目标持仓 {len(result['target'])} 只:")
    score = result["score"]
    # 按打分排序展示
    ranked = score.reindex(result["target"]).sort_values(ascending=False)
    for i, sym in enumerate(ranked.index, 1):
        s = ranked[sym]
        lines.append(f"  {i:3d}. {sym}  得分 {s:+.3f}")

    lines.append(f"\n新增买入 ({len(result['new'])} 只):")
    # 价格标注：T 日收盘产生信号，实盘 T+1 开盘成交。信号时刻无次日开盘价，
    # 故以 T 日收盘作"建议买入参考价"近似基准。
    if ref and result["new"]:
        any_date = next(
            (pd.Timestamp(ref[s]["date"]).strftime("%Y-%m-%d")
             for s in result["new"] if s in ref), "—")
        lines.append(
            f"  （截至 {any_date} 收盘；实盘入场价=T+1 开盘；{stop_note}，"
            f"止盈=入场价×{1 + take_profit:.2f}）")
    limit_up = result.get("limit_up", set())
    for s in result["new"]:
        p = ref.get(s)
        flag = "  ⚠ T日涨停，次日开盘大概率买不进" if s in limit_up else ""
        if p is not None and p["close"] > 0:
            close = p["close"]
            if vol_mode and p.get("est_stop"):
                stop_txt = f"止损(估) {p['est_stop']:.2f}"
            else:
                stop_txt = f"止损 {close * (1 - stop_loss):.2f}"
            lines.append(
                f"  + {s}  得分 {score.get(s, 0):+.3f}  "
                f"参考价 {close:.2f}  {stop_txt}  "
                f"止盈 {close * (1 + take_profit):.2f}{flag}")
        else:
            lines.append(f"  + {s}  得分 {score.get(s, 0):+.3f}  (无最新价){flag}")

    lines.append(f"\n建议卖出 ({len(result['removed'])} 只):")
    for s in result["removed"]:
        lines.append(f"  - {s}")
    if result["new"] and limit_up & set(result["new"]):
        lines.append(f"  ⚠ 新增中 {len(limit_up & set(result['new']))} 只 T 日涨停（回测口径"
                     f"开盘涨停拒买），次日集合竞价按开盘价可成交性自行确认")

    lines.append(f"\n继续持有 ({len(result['kept'])} 只)")
    lines.append("=" * 55)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--holdings", default=None,
                        help="当前持仓，逗号分隔，如 000009,000021。无则视为新建仓")
    parser.add_argument("--weighting", default=None, choices=["static", "ic", "ml"],
                        help="合成方式覆盖（缺省用 config walk_forward.weighting）")
    parser.add_argument("--read-only", action="store_true",
                        help="cache-only（缺数据直接失败，不拉网；烟测/复现用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 审计低危（2026-09-24 修）：strip 空格——"000009, 000021" 里第二个带前导空格，
    # 原实现永不匹配真实持仓（被当新增 + 真实持仓被误列"建议卖出"）
    prev = [s.strip() for s in args.holdings.split(",")] if args.holdings else None
    result = run_daily_portfolio(cfg, prev_holdings=prev, weighting=args.weighting,
                                 read_only=args.read_only)
    report = format_report(result, result["universe"],
                           cfg["multifactor"].get("risk"))

    out = Path(cfg.get("daily_signal", {}).get("output", "reports/daily_signals.txt"))
    out = out.parent / "daily_portfolio.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n已保存到: {out}")


if __name__ == "__main__":
    main()
