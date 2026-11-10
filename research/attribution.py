"""
多因子收益归因（截面回归法）。

回答："样本外超额收益里，每个因子各贡献多少？" —— 特别要看 size 因子是否一家独大。

算法（Brinson 式截面回归）：
1. 从 walk_forward 的 attribution_log 取每个调仓日各 directed 因子对持仓股的截面打分
   + 每日各持仓股收益
2. 以调仓周期为单元：持仓股在"建仓日→下一调仓日"的累计收益，对各因子在建仓日的打分做截面回归
   r_i = α + Σ β_f · score_{f,i} + ε   （i 遍历该周期持仓股）
3. 因子 f 该周期贡献 = β_f × mean(score_f over holdings)
4. 跨周期累加得全程各因子贡献，再算占比

为什么截面回归而非分组：因子间相关（size 与 value 在小盘池正相关），回归能控制相关性给独立贡献；
分组会把相关因子的贡献重复计算。

用法：python attribution.py --universe csi1000 [--weighting ic] [--rebalance-days 1]
默认复用 config.yaml 的 multifactor 设置（生产口径 top_n=20/buffer=4/rebalance=1，见 [[topn-20-vs-50-ab]]），
可命令行覆盖关键旋钮做敏感性测试。
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import yaml

from jq_framework.backtest.walk_forward import run_walk_forward


def _base_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_periods(attrib: dict) -> list[dict]:
    """
    把 attribution_log 整理成"调仓周期"单元。

    每个周期 = 一个调仓日(建仓日) + 到下一调仓日前的所有交易日。
    返回 [{rebal_date, window, factor_scores: {factor: {sym: score}}, holdings: [sym],
           daily: [(date, {sym: ret})], period_ret: {sym: cumret}}, ...]
    """
    sel = pd.DataFrame(attrib["selection_scores"])
    ret = pd.DataFrame(attrib["daily_returns"])
    if sel.empty or ret.empty:
        return []

    # 调仓日（去重，同一天多因子记录只取日期）
    rebal_dates = sorted(sel["date"].unique())
    # 每个调仓日的因子打分表：{factor: {sym: score}}
    factor_scores_by_date = {}
    holdings_by_date = {}
    for d in rebal_dates:
        sub = sel[sel["date"] == d]
        fs = {}
        held = None
        for _, row in sub.iterrows():
            fs[row["factor"]] = row["selected_scores"]
            held = list(row["selected_scores"].keys())
        factor_scores_by_date[d] = fs
        holdings_by_date[d] = held or []

    # 按窗口归属切周期：每个调仓日到下一调仓日(不含)为一周期的持有期
    periods = []
    for i, d in enumerate(rebal_dates):
        d_next = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
        # 该周期内的交易日收益：date > d 且 (d_next is None or date < d_next)
        mask = ret["date"] > d
        if d_next is not None:
            mask &= ret["date"] < d_next
        period_ret = ret[mask]
        if period_ret.empty:
            continue
        # 每只股票周期内累计收益 = ∏(1+r_t) - 1
        cum = (period_ret.assign(g=1 + period_ret["ret"])
               .groupby("symbol")["g"].prod() - 1).to_dict()
        win = period_ret["window"].iloc[0]
        periods.append({
            "rebal_date": d, "window": win,
            "factor_scores": factor_scores_by_date[d],
            "holdings": holdings_by_date[d],
            "period_ret": cum,
        })
    return periods


def _regress_period(period: dict) -> dict:
    """
    单周期截面回归：r_i = α + Σ β_f · score_{f,i} + ε
    返回 {factor: β_f × mean(score_f)} 即各因子该周期收益贡献。
    """
    holdings = period["holdings"]
    ret = period["period_ret"]
    # 取所有股票都有收益和打分的交集
    valid_syms = [s for s in holdings if s in ret]
    if len(valid_syms) < 5:  # 样本太少不回归
        return {}

    factor_names = list(period["factor_scores"].keys())
    # 构建设计矩阵 X[n_sym, n_factor] + 截距，y[n_sym]
    rows = []
    y = []
    for s in valid_syms:
        row = []
        ok = True
        for fn in factor_names:
            sc = period["factor_scores"][fn].get(s)
            if sc is None or (isinstance(sc, float) and np.isnan(sc)):
                ok = False
                break
            row.append(sc)
        if ok and s in ret:
            rows.append(row)
            y.append(ret[s])
    if len(rows) < max(5, len(factor_names) + 2):
        return {}

    X = np.array(rows, dtype=float)
    y = np.array(y, dtype=float)
    # 标准化打分（截面 zscore）使 β 可比，并加截距
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    Xa = np.column_stack([np.ones(len(Xs)), Xs])
    try:
        beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    # 因子贡献 = β_f × mean(标准化打分) = β_f × 0（zscore 均值为0）→ 改用原始打分均值
    # 正确口径：贡献 = β_f × mean(原始 score_f over holdings)，β 来自 zscore 回归
    contribs = {}
    for j, fn in enumerate(factor_names):
        orig = np.array([period["factor_scores"][fn].get(s, np.nan) for s in valid_syms], dtype=float)
        orig = orig[~np.isnan(orig)]
        if len(orig) == 0:
            continue
        contribs[fn] = beta[j + 1] * orig.mean()
    return contribs


def run_attribution(cfg: dict, universe: str, read_only: bool = False, **wf_kwargs) -> dict:
    """跑 walk-forward（开 attribution）并做截面回归归因。

    基准自动跟随 universe（csi500/csi_all→csi300 宽基对比、csi1000→csi1000 同池对比），
    与 compare_strategies.py 的场景定义保持一致，避免基准错配导致超额收益失真。
    csi_all（全市场）对 csi300 宽基对比，不与选股池同构（否则超额恒 0）。

    read_only: True = walk-forward 取数走 cache-only（场景级进程并行 worker 用，预热后杜绝并发写）。
        默认 False。见 plans 点3。
    """
    c = copy.deepcopy(cfg)
    c["multifactor"]["universe"] = universe
    # 基准跟随：csi1000 同池对比（隔离小盘 beta），csi_all/其余对宽基 csi300
    c["multifactor"]["benchmark"] = universe if universe == "csi1000" else c["multifactor"].get("benchmark", "csi300")
    print(f"跑 walk-forward ({universe}, 基准 {c['multifactor']['benchmark']}) 收集归因原料 ...", flush=True)
    res = run_walk_forward(c, universe=universe, attribution=True, read_only=read_only, **wf_kwargs)
    attrib = res.get("attribution_log")
    if not attrib or not attrib["selection_scores"]:
        print("归因原料为空（可能 attribution 未启用或无调仓）")
        return {"result": res}

    periods = _build_periods(attrib)
    if not periods:
        print("无法构建调仓周期")
        return {"result": res}

    # 逐周期回归
    per_period = []
    for p in periods:
        c_ = _regress_period(p)
        if c_:
            per_period.append({"rebal_date": p["rebal_date"], "window": p["window"], "contrib": c_})

    if not per_period:
        print("无周期通过回归（样本不足）")
        return {"result": res}

    # 聚合：各因子全程平均贡献 + 周期数 + 正贡献占比
    factor_names = sorted({fn for pp in per_period for fn in pp["contrib"]})
    summary = {}
    for fn in factor_names:
        vals = [pp["contrib"][fn] for pp in per_period if fn in pp["contrib"]]
        summary[fn] = {
            "mean_contrib": float(np.mean(vals)),
            "sum_contrib": float(np.sum(vals)),
            "n_periods": len(vals),
            "pos_rate": float(np.mean([1 if v > 0 else 0 for v in vals])),
            "std": float(np.std(vals)),
        }

    # 打印
    print("\n" + "=" * 70, flush=True)
    print(f"  收益归因（截面回归，{universe}，{len(per_period)} 个调仓周期）", flush=True)
    print("=" * 70, flush=True)
    print(f"  策略超额年化: {res['excess_annual']:+.2%}  夏普: {res['sharpe']:.2f}", flush=True)
    print(f"  {'因子':<14}{'平均贡献':>10}{'累计贡献':>10}{'周期数':>7}{'正贡献占比':>11}{'贡献std':>9}", flush=True)
    print("  " + "-" * 66, flush=True)
    # 按平均贡献绝对值排序
    for fn in sorted(summary, key=lambda x: -abs(summary[x]["mean_contrib"])):
        s = summary[fn]
        print(f"  {fn:<14}{s['mean_contrib']:>+10.4f}{s['sum_contrib']:>+10.4f}"
              f"{s['n_periods']:>7}{s['pos_rate']:>10.0%}{s['std']:>9.4f}", flush=True)
    # 占比（按平均贡献绝对值归一）
    tot = sum(abs(s["mean_contrib"]) for s in summary.values()) or 1.0
    print("  " + "-" * 66, flush=True)
    print("  贡献占比（按 |平均贡献| 归一）:", flush=True)
    for fn in sorted(summary, key=lambda x: -abs(summary[x]["mean_contrib"])):
        print(f"    {fn:<12} {abs(summary[fn]['mean_contrib'])/tot:>6.1%}", flush=True)
    print("=" * 70, flush=True)
    return {"result": res, "summary": summary, "n_periods": len(per_period)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--universe", default="csi500")
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--weighting", default=None, help="覆盖 walk_forward.weighting")
    parser.add_argument("--rebalance-days", type=int, default=None, help="覆盖 select.rebalance_days")
    parser.add_argument("--clamp-cap", type=float, default=None, help="覆盖 ic_clamp_cap（敏感性测试）")
    parser.add_argument("--top-n", type=int, default=None, help="覆盖 select.top_n（生产默认 20）")
    parser.add_argument("--buffer", type=int, default=None, help="覆盖 select.turnover_buffer（生产默认 4）")
    args = parser.parse_args()

    cfg = _base_cfg(args.config)
    if args.weighting or args.rebalance_days or args.clamp_cap is not None or args.top_n or args.buffer:
        wf = cfg["multifactor"].setdefault("walk_forward", {})
        sel = cfg["multifactor"].setdefault("select", {})
        if args.weighting:
            wf["weighting"] = args.weighting
        if args.rebalance_days is not None:
            sel["rebalance_days"] = args.rebalance_days
        if args.clamp_cap is not None:
            wf["ic_clamp_cap"] = args.clamp_cap
        if args.top_n is not None:
            sel["top_n"] = args.top_n
        if args.buffer is not None:
            sel["turnover_buffer"] = args.buffer

    run_attribution(cfg, args.universe,
                    train_days=args.train_days, test_days=args.test_days,
                    step_days=args.step_days, horizon=args.horizon)


if __name__ == "__main__":
    main()
