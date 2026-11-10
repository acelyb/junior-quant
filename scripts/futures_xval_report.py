"""futures_xval_report.py — 期货数据基建独立交叉验证报告（只读、离线、不拉网）。

定位（plan ①b / docs §3.2）：期货线的数据基建（换月拼接/复权）从未被独立验证。
本脚本对每个配置品种跑**两路独立数据源**的对账：
  - X0   ：新浪主连（外部厂商已复权口径）
  - built：自建主力连续（build_contract_series，真换月拼接 + 前复权）
两者互相印证数据正确性：高相关(≥0.99)/高重合(≥95%) → 数据基建正常；
I/TA 类 0.97/0.98 属已知对照差异（X0 主连个别品种复权基准与自建有系统差），记入报告非断言。

产物：reports/futures_xval_report.csv（每品种一行），stdout 打印判据达标情况。

用法:
  python scripts/futures_xval_report.py --config config.yaml
  python scripts/futures_xval_report.py --config config.yaml --out reports/futures_xval_report.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime

import pandas as pd
import yaml

from jq_framework.config.loader import load_config
from jq_framework.data.fetchers.fetcher_futures import (
    build_contract_series,
    get_futures_main_continuous,
    _x0_vs_built_stats,
)


def _variety(sym: str) -> str:
    """新浪主连 'RB0' → 品种 'RB'（去尾部 0）。"""
    return sym[:-1] if sym.endswith("0") else sym


def xval_one(variety: str, read_only: bool = True) -> dict:
    """单品种两路对账：返回报告行（corr/overlap/换月点一致性），失败品种指标留空。"""
    row = {"variety": variety, "corr": None, "overlap": None,
           "n": 0, "roll_days": 0, "roll_consistency": None, "status": "ok"}
    try:
        x0 = get_futures_main_continuous(variety, contract="X0", read_only=read_only)
        built = build_contract_series(variety, read_only=read_only)
    except Exception as e:
        row["status"] = f"拉取/构建失败: {str(e)[:60]}"
        return row
    if x0 is None or built is None or built.empty:
        row["status"] = "数据为空（X0 或 built 缺）"
        return row
    st = _x0_vs_built_stats(built, x0)
    row.update({
        "corr": st["corr"], "overlap": st["overlap"], "n": st["n"],
        "roll_days": st["roll_days"], "roll_consistency": st["roll_day_ret_consistency"],
    })
    row["pass"] = _pass(st["corr"], st["overlap"])
    return row


def _pass(corr, overlap):
    if corr is None or overlap is None:
        return False
    return corr >= 0.99 and overlap >= 0.95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--symbols", default=None,
                        help="逗号分隔品种（默认取 config futures.symbols）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fut = cfg.get("futures", {})
    symbols = args.symbols.split(",") if args.symbols \
        else fut.get("symbols", ["RB0", "AU0"])
    varieties = [_variety(s) for s in symbols]

    rows = [xval_one(v) for v in varieties]
    df = pd.DataFrame(rows).replace({None: ""})
    out = args.out or "reports/futures_xval_report.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_pass = int(df["pass"].sum()) if "pass" in df.columns else 0
    print(f"\n{'='*72}\n  期货数据基建交叉验证（X0 vs built）"
          f"{datetime.now():%Y-%m-%d %H:%M}\n{'='*72}")
    print(f"  {'品种':<6}{'corr':>9}{'overlap':>9}{'对账天数':>9}{'换月日':>7}"
          f"{'换月一致':>9}  判据")
    for _, r in df.iterrows():
        ok = "PASS" if r.get("pass") else "FAIL/NA"
        cons = f"{r['roll_consistency']:.3f}" if isinstance(r.get("roll_consistency"), (int, float)) else "-"
        print(f"  {r['variety']:<6}{r['corr']:>9.4f}{r['overlap']:>8.2%}"
              f"{r['n']:>9}{r['roll_days']:>7}{cons:>9}  {ok}")
    print(f"\n判据：corr≥0.99 且 overlap≥95%（I/TA 等 0.97/0.98 属已知系统差，见报告 non-assert）")
    print(f"\n通过 {n_pass}/{len(df)} 品种。报告已保存: {out}")


if __name__ == "__main__":
    main()
