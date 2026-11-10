"""futures_combination_signal.py — 期货组合级每日信号（可执行手数清单）。

对标主流期货流程「信号 → 组合权重 → 风控 → 实际头寸」的**当日出口**（A 股有
daily_portfolio.py 可执行清单，期货线补这一等价物）。与回测共用单源：
  - 数据/方向：复用 signals.daily_runner.run_daily（各策略逐品种方向 + 合成因子分）
  - 权重：波动率平价 w_i ∝ 1/σ_i（与 futures_walk_forward 同口径，可 --weight-method 切换）
  - 风控：FuturesRiskManager.apply（杠杆/单品种保证金/保证金预算，与回测一致）
  - 头寸：pos_size 换手数/名义/保证金（portfolio/futures_sizing，回测与信号共用）

**出口关系（强化计划第三步写清，防双出口混淆）**：
  - `signals/daily_runner.py` = 每日信号**正式出口**（逐品种方向 + 口径标注 +
    carry 覆盖 fail-loud，股票与期货共用）；
  - 本脚本 = **组合级执行清单出口**，纯下游——方向/标注全部取自 run_daily 记录，
    只叠加组合层（权重/风控/手数/保证金预算），不做任何信号计算。
    本报告头部的 carry 覆盖与口径汇总同样来自 run_daily 记录，与信号报告同源。

输出「今日组合：各品种开几手/占多少保证金/需多少资金/自有余量」，回答实盘执行问题。
默认 100 万账户净值，--nav 覆盖，--capital-utilization 覆盖保证金预算。

用法:
  python scripts/futures_combination_signal.py --config config.yaml
  python scripts/futures_combination_signal.py --weight-method vol_parity --capital-utilization 0.7
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import pandas as pd
import yaml

from jq_framework.signals.daily_runner import run_daily, _carry_coverage_line
from jq_framework.backtest.futures_walk_forward import effective_bets, sector_shares
from jq_framework.data.fetchers.fetcher_futures import (
    build_contract_series, get_futures_main_continuous, get_futures_rules,
    normalize_built_frame, resolve_contract,
)
from jq_framework.risk.futures_risk import FuturesRiskManager
from jq_framework.portfolio.futures_sizing import pos_size, margin_used


def _concentration_lines(w_scaled: dict[str, float]) -> list[str]:
    """板块集中度软形态（第六步验收）：只提示不强制。纯函数，供单测。

    输出类别份额 + 有效独立下注（1/HHI，futures_walk_forward 单一定义）；
    单一板块份额 > 25%（volparity_capped 硬上限候选水平）时加告警行。
    空仓（总 |w| ≤ 0）返回空列表。
    """
    sh = sector_shares(w_scaled)
    if not sh:
        return []
    lines = ["  板块集中度（软提示，不强制）: "
             + " | ".join(f"{c} {s:.0%}" for c, s in
                         sorted(sh.items(), key=lambda kv: -kv[1])),
             f"  有效独立下注: {effective_bets(sh):.1f} / {len(sh)} 类"]
    top = max(sh.values())
    if top > 0.25:
        lines.append(f"  ⚠️ 单一板块份额 {top:.0%} > 25%"
                     "（volparity_capped 硬上限候选水平，启用与否见 结论.md §2.9）")
    return lines


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def _futures_variety(sym: str) -> str:
    return sym[:-1] if sym.endswith("0") else sym


def _load_daily(var: str, contract: str) -> pd.DataFrame | None:
    """口径一致的品种日线（与信号侧/回测同口径）：built → 归一化复权帧。

    built 必须经 normalize_built_frame——原始拼接帧的 settle 含换月跳空，
    直接取波动率会把跳空当真实波动（S4 同款问题在信号侧的镜像）。
    """
    if contract == "built":
        return normalize_built_frame(build_contract_series(var, read_only=True))
    return get_futures_main_continuous(var, contract="X0", read_only=True)


def _factor_weights(records: list[dict], strat_name: str) -> dict[str, float]:
    """从 run_daily 记录取期货各品种的因子/方向 → 带符号权重 w_i。

    run_daily 记录带 factor（TSMOM+Carry 合成方向分，∈[-1,1] 连续），
    以 sign(factor) 定向、|factor| 作该品种目标仓位（target_pos 语义）。
    逐品种 → 组合前权重（未做风控缩放）。
    """
    w: dict[str, float] = {}
    for r in records:
        # 记录 market = 策略 type（"cta" 表示期货线），strategy 名对齐组合策略
        if not _is_futures_rec(r) or r["direction"] not in ("long", "short"):
            continue
        # 只用与组合策略同名的记录（默认 tsmom_cta——2026-10-15 裁定 C 后生产白名单）
        if r["strategy"] != strat_name:
            continue
        var = _futures_variety(r["symbol"])
        w[var] = {"long": 1.0, "short": -1.0}[r["direction"]] * float(r["target_pos"])
    return w


def _is_futures_rec(r: dict) -> bool:
    """run_daily 期货记录：market==\"cta\"（cta=期货时序线，stock=A股）。"""
    return r.get("market") == "cta"


def _realized_vols(cfg: dict, window: int = 60) -> dict[str, float]:
    """各品种近期已实现波动率（年化），供波动率平价权重。取各品种主连日线算。
    数据缺失品种给默认 0.2（年化），避免绝权重给零。
    口径与信号侧一致（_load_daily）：built → 归一化复权帧（settle 重建无跳空）。
    """
    fut = cfg.get("futures", {})
    symbols = fut.get("symbols", ["RB0", "AU0"])
    contract = resolve_contract(cfg)  # 第零步：口径一个来源（config futures.contract）
    vols: dict[str, float] = {}
    for sym in symbols:
        var = _futures_variety(sym)
        try:
            df = _load_daily(var, contract)
            if df is None or df.empty or "settle" not in df.columns:
                continue
            s = pd.to_numeric(df["settle"], errors="coerce")
            r = s.pct_change().dropna().tail(window)
            vols[var] = float(np.std(r, ddof=1)) * np.sqrt(252) if len(r) > 5 else 0.2
        except Exception:
            pass
        vols.setdefault(var, 0.2)
    return vols


def make_weights(dirs: dict[str, float], vols: dict[str, float],
                 budget: float, method: str = "vol_parity") -> dict[str, float]:
    """组合权重：vol_parity（w∝1/σ，与回测同口径）或 equal（±budget/N）。"""
    active = {k: v for k, v in dirs.items() if v != 0}
    if not active:
        return {}
    if method == "vol_parity":
        inv = {k: 1.0 / max(vols.get(k, 0.05), 0.05) for k in active}
        s = sum(inv.values())
        return {k: budget * (inv[k] / s) * (1 if dirs[k] > 0 else -1) for k in active}
    n = len(active)
    return {k: (budget / n) * (1 if dirs[k] > 0 else -1) for k in active}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--strategy", default="tsmom_cta")
    parser.add_argument("--weight-method", default="vol_parity",
                        choices=["vol_parity", "equal"])
    parser.add_argument("--nav", type=float, default=None,
                        help="账户净值（默认 config futures.nav 或 1e6）")
    parser.add_argument("--capital-utilization", type=float, default=None,
                        help="保证金占用预算上限（覆盖 config futures.risk.capital_utilization）")
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fut = cfg.get("futures", {})
    nav = args.nav or float(fut.get("nav", 1_000_000.0))

    # 1) 方向（run_daily 已做 carry 注入 + 因子合成；单次调用，记录同时供
    #    方向提取与下方 carry 覆盖/口径汇总——与信号报告同源）
    records = run_daily(cfg)
    dirs = _factor_weights(records, args.strategy)
    if not dirs:
        print("今日无可操作期货方向（全部观望/平仓）")
        return

    # 2) 波动率平价权重（预算 = futures.risk.max_leverage 或 1.0）
    risk_cfg = fut.get("risk", {}) or {}
    budget = float(risk_cfg.get("max_leverage", 1.0) or 1.0)
    vols = _realized_vols(cfg, window=args.window)
    weights = make_weights(dirs, vols, budget, args.weight_method)

    # 3) 风控（与回测同一 FuturesRiskManager，含保证金预算）
    rm_cfg = dict(risk_cfg)
    if args.capital_utilization is not None:
        rm_cfg["capital_utilization"] = args.capital_utilization
    risk = FuturesRiskManager(rm_cfg)
    rules = get_futures_rules(read_only=True)
    margin_ratios = {}
    mult = {}
    for var in weights:
        r = rules.get(var, {})
        mr = r.get("margin_ratio")
        margin_ratios[var] = (mr / 100.0) if mr else float(fut.get("margin", 0.1))
        mult[var] = r.get("multiplier")
    w_scaled, margin_total = risk.apply(weights, margin_ratios, nav_t=None)

    # 4) 头寸换算（开几手/名义/保证金）——口径与方向/权重同源（config futures.contract）
    contract = resolve_contract(cfg)
    prices = {}
    for var in weights:
        try:
            df = _load_daily(var, contract)
            prices[var] = float(df["close"].iloc[-1]) if df is not None and not df.empty else None
        except Exception:
            prices[var] = None
    default_mult = fut.get("default_multiplier")
    positions = pos_size(w_scaled, nav, prices, mult, margin_ratios,
                         cap_util=0.0, default_multiplier=default_mult)

    # 5) 报告
    print("=" * 66)
    print(f"  期货组合今日信号   {args.strategy}  权重={args.weight_method}")
    print("=" * 66)
    # carry 覆盖 + 口径兜底汇总（来自同一批 run_daily 记录，与信号报告同源）
    cov = _carry_coverage_line(records)
    if cov:
        print(f"  {cov}")
        n_active = sum(1 for r in records if "carry" in r and r["carry"]["active"])
        if n_active < 8:
            print(f"  ⚠️ carry 活跃品种仅 {n_active}——无源/过期品种已退化为 0.5×TSMOM 半信号")
    fallback = sorted({r["symbol"] for r in records if r.get("contract") == "X0-fallback"})
    if fallback:
        print(f"  ⚠️ {len(fallback)} 个品种自建换月无数据、已退回 X0 口径（含换月跳空）: "
              f"{','.join(fallback)}")
    print(f"  账户净值: {nav:,.0f}   保证金预算占用: {margin_total:.1%}")
    print(f"  {'品种':<6}{'方向':<6}{'目标仓位':>9}{'开仓手数':>9}{'名义':>14}{'保证金':>12}")
    for var, w in w_scaled.items() if w_scaled else []:
        p = positions.get(var, {})
        direction = "多" if w > 0 else ("空" if w < 0 else "平")
        if w == 0:
            continue
        print(f"  {var:<6}{direction:<6}{w:>+9.2%}{p.get('lots',0):>9}{p.get('notional',0):>14,.0f}"
              f"{p.get('margin',0):>12,.0f}")
    # 自有余量：保证金预算 - 当前占用（若预算未设则显示无需约束）
    cap = rm_cfg.get("capital_utilization")
    if cap:
        used_now = margin_used(w_scaled, margin_ratios)
        print(f"\n  保证金占用: {used_now:.2%} / 预算 {cap:.0%}   "
              f"自余: {max(cap - used_now, 0.0):.2%}")
    # 板块集中度软形态（第六步）：只提示不强制
    for line in _concentration_lines(w_scaled):
        print(line)
    print("=" * 66)


if __name__ == "__main__":
    main()
