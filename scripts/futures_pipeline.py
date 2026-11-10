"""futures_pipeline.py — 期货线统一可复现编排（拉数→回测→健康卡→对账报告）。

定位（plan §2）：此前一条可复现 OOS 链路要手动串 3-5 条命令（docs §14），无统一入口。
本脚本一条命令链起：**预热/拉数缓存 → 多策略 OOS 回测 → 因子健康卡 → 数据基建对账**。
各 stage **子进程复用现有各脚本 CLI**（不复制逻辑、不动现有脚本），结果各自落 --out-dir。

stage（--stages comma 子集，默认全跑）：
  fetch   —— 预热数据：X0 主连 + self-built 换月 + carry（read_only=False 拉+写 parquet），
              给每品种起点/行数摘要（数据缺失 M4 回填后在此暴露，不在回测里暗炸）。
  backtest—— backtest.futures_walk_forward 各白名单策略 OOS 回测（tsmom_carry 默认）。
             写 NAV/turnover/cost/margin CSV。
  health  —— scripts.futures_factor_health.py → health 卡（research/futures_factor_health.csv）。
  xval    —— scripts.futures_xval_report.py（read_only=True 离线对账 X0 vs built）
              → --out-dir/futures_xval_report.csv。

幂等/可复现：--run-tag（默认 YYYYMMDD_HHMM）建时间戳子目录，避免覆盖上次产物；
末尾打印各 stage 产物路径 + health 卡决策表。

用法：
  python scripts/futures_pipeline.py --config config.yaml
  python scripts/futures_pipeline.py --stages fetch,backtest --contract built
  python scripts/futures_pipeline.py --stages health --train-days 504 --test-days 126 --step-days 126
"""
from __future__ import annotations

from jq_framework.config.loader import load_config

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# stage → (帮助文本, 是否吃网络)
STAGES = {
    "fetch": "预热数据（X0 主连 + built 换月 + carry），写 parquet 缓存",
    "backtest": "OOS 回测（futures_walk_forward，白名单策略）",
    "health": "因子健康卡（futures_factor_health）",
    "xval": "数据基建对账（futures_xval_report，离线只读）",
}
_DEFAULT_STAGES = ",".join(STAGES.keys())
# 回测白名单策略（治理后主线 + 单因子载体）
_BACKTEST_STRATEGIES = ("tsmom_carry_cta", "tsmom_cta", "carry_cta")


def _varieties(cfg: dict) -> list[str]:
    syms = cfg.get("futures", {}).get("symbols", [])
    return [s[:-1] if s.endswith("0") else s for s in syms]


def _run(cmd: list[str], cwd: Path) -> None:
    """子进程跑同解释器命令（conda 下用同 python），失败即抛。"""
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def stage_fetch(cfg: dict, args) -> None:
    """预热数据缓存：X0/built 主连 + carry，逐品种给起点/行数摘要。"""
    from jq_framework.data.fetchers.fetcher_futures import (
        build_contract_series, get_futures_main_continuous,
        get_roll_yield_series,
    )
    varieties = _varieties(cfg)
    print(f"\n[futures_pipeline] stage=fetch 预热 {len(varieties)} 品种（read_only=False）...")
    for v in varieties:
        x0 = get_futures_main_continuous(v, contract="X0", read_only=False)
        x0_days = len(x0) if x0 is not None and not x0.empty else 0
        x0_0 = x0["date"].iloc[0] if x0_days else "-"
        built = build_contract_series(v, read_only=False)
        b_days = len(built) if built is not None and not built.empty else 0
        b_0 = built["date"].iloc[0] if b_days else "-"
        carry = get_roll_yield_series(v)  # 默认 config futures.start（M4）
        c_days = len(carry); c_0 = carry["date"].iloc[0] if c_days else "-"
        print(f"  {v:<4} X0:{x0_days}天@{x0_0} | built:{b_days}天@{b_0} | carry:{c_days}天@{c_0}")


def stage_backtest(cfg: dict, args) -> None:
    for strat in _BACKTEST_STRATEGIES:
        cmd = [sys.executable, "-m", "jq_framework.backtest.futures_walk_forward",
               "--config", str(args.config), "--strategy", strat,
               "--weight-method", args.weight_method,
               # 审计 #18：透传 run-tag 子目录（引擎产物文件名带策略名），
               # 三策略互不覆盖，跑完可横向对比
               "--out-dir", str(args.out_dir)]
        if args.contract is not None:
            cmd += ["--contract", args.contract]
        if args.attribution:
            cmd.append("--attribution")
        for flag, val in (("--train-days", args.train_days),
                          ("--test-days", args.test_days),
                          ("--step-days", args.step_days)):
            if val is not None:
                cmd += [flag, str(val)]
        _run(cmd, ROOT)


def stage_health(cfg: dict, args) -> None:
    cmd = [sys.executable, "scripts/futures_factor_health.py",
           "--config", str(args.config), "--weight-method", args.weight_method]
    if args.contract is not None:
        cmd += ["--contract", args.contract]
    for flag, val in (("--train-days", args.train_days),
                      ("--test-days", args.test_days),
                      ("--step-days", args.step_days)):
        if val is not None:
            cmd += [flag, str(val)]
    cmd += ["--out", str(args.out_dir / f"futures_factor_health_{args.run_tag}.csv")]
    _run(cmd, ROOT)


def stage_xval(cfg: dict, args) -> None:
    cmd = [sys.executable, "scripts/futures_xval_report.py",
           "--config", str(args.config),
           "--out", str(args.out_dir / f"futures_xval_report_{args.run_tag}.csv")]
    _run(cmd, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stages", default=_DEFAULT_STAGES,
                        help=f"逗号子集 ({_DEFAULT_STAGES})；默认全跑")
    parser.add_argument("--contract", default=None, choices=["X0", "built"],
                        help="主连口径；缺省读 config futures.contract（子进程各自解析）")
    parser.add_argument("--weight-method", default="vol_parity",
                        choices=["vol_parity", "equal"])
    parser.add_argument("--attribution", action="store_true", help="回测开逐日归因")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--run-tag", default=None,
                        help="产物子目录名（默认 YYYYMMDD_HHMM）")
    parser.add_argument("--out-dir", default=None,
                        help="报告落盘根目录（默认 reports/；各 stage 产物放 --out-dir/<run-tag>/）")
    args = parser.parse_args()

    args.config = Path(args.config)
    if not args.config.exists():
        print(f"[futures_pipeline] 配置不存在: {args.config}")
        sys.exit(2)
    import yaml
    cfg = load_config(args.config)

    now = datetime.now()
    args.run_tag = args.run_tag or now.strftime("%Y%m%d_%H%M%S")
    args.out_dir = Path(args.out_dir or "reports") / args.run_tag
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[futures_pipeline] run_tag={args.run_tag} out_dir={args.out_dir}")

    steps = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in steps:
        if s not in STAGES:
            print(f"[futures_pipeline] 未知 stage: {s}（可选 {_DEFAULT_STAGES}）")
            sys.exit(2)

    t0 = time.time()
    for s in steps:
        print(f"\n{'='*72}\n  stage={s}: {STAGES[s]}\n{'='*72}")
        if s == "fetch":
            stage_fetch(cfg, args)
        elif s == "backtest":
            stage_backtest(cfg, args)
        elif s == "health":
            stage_health(cfg, args)
        elif s == "xval":
            stage_xval(cfg, args)
    print(f"\n[futures_pipeline] 完成 {steps}，耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"[futures_pipeline] 产物: {args.out_dir}")


if __name__ == "__main__":
    main()
