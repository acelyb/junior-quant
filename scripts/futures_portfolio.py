"""futures_portfolio.py — 期货线一键跑批（配置驱动，对标 A 股 compare_*.py）。

主流期货流程「多策略 × 多权重 × 风控 × 成本/换手/归因」的一次性入口：跑一批
`futures_walk_forward` 组合回测并汇总对比报告 + 落盘 NAV/换手/成本/保证金 CSV。
回测引擎复用 backtest.futures_walk_forward.run_futures_portfolio（单源），本脚本
只做「跑批 + 汇总」编排，不做策略实现。

默认跑 config 里注册的所有 cta 策略 × 默认权重；可用 --strategy/--weight-method 收窄。
输出：reports/futures_portfolio_<ts>.csv（汇总）+ 各策略 NAV/换手/成本/保证金 CSV。

用法:
  python scripts/futures_portfolio.py --config config.yaml
  python scripts/futures_portfolio.py --strategy tsmom_cta,tsmom_carry_cta --weight-method vol_parity
  python scripts/futures_portfolio.py --contract built   # 自建换月拼接对照
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime

import pandas as pd
import yaml

from jq_framework.backtest.futures_walk_forward import run_futures_portfolio
from jq_framework.data.fetchers.fetcher_futures import resolve_contract

# 期货时序线策略（cta 类型）——跑批默认全集；A 股策略不在期货线跑批内
_CTA_STRATS = ("tsmom_cta", "carry_cta", "tsmom_carry_cta")


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def _all_cta_strats(cfg: dict) -> list[str]:
    return [name for name, sc in cfg.get("strategies", {}).items()
            if sc.get("type") == "cta"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--strategy", default=None,
                        help="逗号分隔策略名；缺省=跑全部 cta 策略")
    parser.add_argument("--weight-method", default="vol_parity",
                        choices=["vol_parity", "equal"])
    parser.add_argument("--contract", default=None, choices=["X0", "built"],
                        help="主连口径；默认读 config futures.contract（缺省 X0）")
    parser.add_argument("--rebalance-days", type=int, default=None)
    parser.add_argument("--attribution", action="store_true")
    parser.add_argument("--train-days", type=int, default=None,
                        help="样本外分窗：训练段长度（天）。与 --test-days 同给才启用分窗")
    parser.add_argument("--test-days", type=int, default=None,
                        help="样本外分窗：测试段长度（天）")
    parser.add_argument("--step-days", type=int, default=None,
                        help="样本外分窗：步进（默认=test-days，即无重叠滑窗）")
    parser.add_argument("--out", default=None, help="汇总 CSV 输出路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    strats = (args.strategy.split(",") if args.strategy
              else _all_cta_strats(cfg))
    strats = [s for s in strats if s in cfg.get("strategies", {})]
    # 第零步契约激活：--contract 缺省读 config futures.contract（打印/汇总列用有效值）
    contract = resolve_contract(cfg, args.contract)

    rows = []
    for strat in strats:
        print(f"\n===== {strat}  [{args.weight_method}] contract={contract} =====")
        res = run_futures_portfolio(
            cfg, strat, contract=contract, weight_method=args.weight_method,
            rebalance_days=args.rebalance_days, attribution=args.attribution,
            train_days=args.train_days, test_days=args.test_days,
            step_days=args.step_days, verbose=False)
        st = res["stats"]
        rows.append({
            "strategy": strat,
            "weight": args.weight_method,
            "contract": contract,
            "total_return": st.get("total_return"),
            "ann_return": st.get("ann_return"),
            "excess_annual": st.get("excess_annual"),
            "sharpe": st.get("sharpe"),
            "max_drawdown": st.get("max_drawdown"),
            "bench_total": st.get("bench_total"),
        })
        # 落盘各策略 NAV 时序（含换手/成本/保证金）
        if not res["nav_df"].empty:
            tag = f"{strat}_{args.weight_method}"
            res["nav_df"].to_csv(Path("reports") / f"futures_portfolio_nav_{tag}.csv",
                                 index=False)
            for k in ("turnover_series", "cost_series", "margin_series"):
                dfx = res.get(k)
                if dfx is not None and not dfx.empty:
                    dfx.to_csv(Path("reports") / f"futures_portfolio_{k}_{tag}.csv",
                               index=False)

    if not rows:
        print("无可跑策略")
        return
    df = pd.DataFrame(rows)
    out = args.out or f"reports/futures_portfolio_{datetime.now():%Y%m%d_%H%M}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # stdout 汇总对比
    print("\n" + "=" * 78)
    print(f"  期货组合跑批汇总   {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 78)
    print(f"  {'策略':<18}{'累计':>12}{'年化':>10}{'超额/年':>10}{'夏普':>8}{'回撤':>10}")
    for _, r in df.iterrows():
        print(f"  {r['strategy']:<18}{r['total_return']:>11.1%}{r['ann_return']:>9.1%}"
              f"{r['excess_annual']:>9.1%}{r['sharpe']:>8.2f}{r['max_drawdown']:>9.1%}")
    print(f"\n汇总已保存: {out}")


if __name__ == "__main__":
    main()
