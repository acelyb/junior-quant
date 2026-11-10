"""
因子检验报告：对配置中的所有因子跑 IC 分析，输出有效性与可视化。

用法：
  python -m factors.ic_report --config config.yaml
  python -m factors.ic_report --config config.yaml --horizon 10 --universe csi300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面环境
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from jq_framework.data.fetchers.fetcher import get_index_constituents, get_batch_daily
from jq_framework.data.fetchers.fetcher_fundamental import build_fundamental_data
from jq_framework.factors.composite import load_factor
from jq_framework.factors.ic import calc_factor_ic, calc_composite_ic, ic_summary, interpret


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def run_ic_analysis(cfg: dict, universe: str | None = None,
                    horizon: int = 10, freq: int = 5,
                    verbose: bool = True) -> dict:
    """
    对所有因子 + 合成打分跑 IC 分析。
    返回 {name: {summary, ic_series}}
    """
    mf = cfg["multifactor"]
    universe = universe or mf["universe"]
    syms = get_index_constituents(universe)

    if verbose:
        print(f"拉取 {len(syms)} 只股票日线 ...")
    price_dfs = get_batch_daily(syms, progress=verbose)

    # 基本面因子数据：若有 is_fundamental 因子则构建一次，单因子与合成共用
    specs = list(mf["factors"].items())
    has_fund = any(getattr(load_factor(spec), "is_fundamental", False)
                   for _, spec in specs)
    fund_data = None
    if has_fund:
        if verbose:
            print("拉取基本面数据（估值+财务）注入基本面因子 ...")
        try:
            fund_data = build_fundamental_data(syms, progress=verbose)
        except Exception as e:
            if verbose:
                print(f"  基本面数据拉取失败: {e}")

    results = {}
    # 单因子
    for name, spec in mf["factors"].items():
        if verbose:
            print(f"\n计算因子 [{name}] IC ...")
        factor = load_factor(spec)
        if getattr(factor, "is_fundamental", False) and fund_data is not None:
            factor.set_fundamental_data(fund_data)
        ic = calc_factor_ic(price_dfs, factor, horizon=horizon, freq=freq)
        summary = ic_summary(ic["ic"] if not ic.empty else pd.Series(dtype=float))
        results[name] = {"summary": summary, "ic": ic, "interpret": interpret(summary)}
        if verbose:
            print(f"  IC均值={summary['ic_mean']:+.4f}  ICIR={summary['icir']:+.3f}  "
                  f"胜率={summary['win_rate']:.1%}  t={summary['t_stat']:+.2f}  "
                  f"n={summary['n']}")

    # 合成打分
    if verbose:
        print(f"\n计算 [合成打分] IC ...")
    factors = [(load_factor(spec), spec.get("weight", 1.0))
               for spec in mf["factors"].values()]
    if fund_data is not None:
        for factor, _ in factors:
            if getattr(factor, "is_fundamental", False):
                factor.set_fundamental_data(fund_data)
    # 合成 IC 与回测用同一 combine 路径（None=flat 默认；twolevel 时反映分层合成打分）
    combine_cfg = {k: mf.get(k) for k in ("combine", "families", "orthogonalize")}
    ic = calc_composite_ic(price_dfs, factors, horizon=horizon, freq=freq,
                           combine_cfg=combine_cfg)
    summary = ic_summary(ic["ic"] if not ic.empty else pd.Series(dtype=float))
    results["composite"] = {"summary": summary, "ic": ic, "interpret": interpret(summary)}
    if verbose:
        print(f"  IC均值={summary['ic_mean']:+.4f}  ICIR={summary['icir']:+.3f}  "
              f"胜率={summary['win_rate']:.1%}  t={summary['t_stat']:+.2f}")

    return results


def print_report(results: dict, universe: str, horizon: int):
    print("\n" + "=" * 72)
    print(f"  因子IC检验报告  股票池={universe}  预测期={horizon}日")
    print("=" * 72)
    print(f"{'因子':<12}{'IC均值':>10}{'ICIR':>9}{'胜率':>8}{'t值':>8}{'样本':>7}  判断")
    print("-" * 72)
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<12}{s['ic_mean']:>+10.4f}{s['icir']:>+9.3f}"
              f"{s['win_rate']:>8.1%}{s['t_stat']:>+8.2f}{s['n']:>7}  {r['interpret']}")
    print("=" * 72)
    print("判定标准: |IC均值|>0.03弱 / >0.05中 / >0.1强; |ICIR|>0.5稳定; |t|>2显著")
    print("=" * 72)


def plot_ic(results: dict, out_path: str):
    """画各因子累计IC曲线。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in results.items():
        ic = r["ic"]
        if ic.empty:
            continue
        cum_ic = ic["ic"].cumsum()
        ax.plot(cum_ic.index, cum_ic.values, label=name, linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative IC")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative IC")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"\n累计IC曲线已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--horizon", type=int, default=10, help="未来收益持有期(天)")
    parser.add_argument("--freq", type=int, default=5, help="采样频率(天)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_ic_analysis(cfg, universe=args.universe,
                              horizon=args.horizon, freq=args.freq)
    universe = args.universe or cfg["multifactor"]["universe"]
    print_report(results, universe, args.horizon)
    plot_ic(results, "reports/ic_curve.png")


if __name__ == "__main__":
    main()
