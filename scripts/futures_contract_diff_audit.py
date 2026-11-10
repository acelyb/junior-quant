"""futures_contract_diff_audit.py — X0 vs built 信号差异审计（期货CTA_强化计划 第二步）。

切换生产口径（config futures.contract: X0 → built）前回答一个问题：
**原始主连 X0 上的信号被换月跳空翻转过多少**。逐品种用生产策略
tsmom_carry_cta 的两个独立实例分别吃 X0 / built(自建换月复权) 序列，
carry 同源注入（差异只来自价格口径——TSMOM 腿的 settle/250 日累计收益），
按引擎同款喂数语义（信号用 dt 之前 bars、carry 用 ≤ dt）在公共交易日轴上
逐日比对方向。

口径细节：
  - built 判据 = 归一化后帧有无 adj_close（与引擎/每日信号 fail-loud 同根）；
    无 adj_close 的品种（build 兜底返 X0）两口径同数据 → 无差异可比，标跳过。
  - 预热期（bars < lookback+1）双方都是 hold，不算可比日——避免把"信息不足
    阶段的一致"混进一致率。carry 同源 → carry 腿逐日同分，差异全归 TSMOM 腿。
  - built 覆盖起点随日历滚动（逐月枚举近 years_back=5 年合约，最早合约挂牌
    ≈ 2020-01），公共轴起点由 built 决定，审计结论限于该样本段。

产物（read_only 只读缓存、单进程串行、不拉网）：
  reports/futures_contract_diff.csv        逐品种汇总（可比日/不一致数/率/换月日数）
  reports/futures_contract_diff_days.csv   不一致日明细（品种/日期/两口径方向）

用法:
  python scripts/futures_contract_diff_audit.py --config config.yaml
  python scripts/futures_contract_diff_audit.py --config config.yaml \
      --out reports/futures_contract_diff.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None

import argparse

import pandas as pd
import yaml

from jq_framework.data.fetchers.fetcher_futures import (
    build_contract_series,
    get_futures_main_continuous,
    get_roll_yield_series,
    normalize_built_frame,
)


def _strategy_instances(strat_cfg: dict) -> tuple:
    """两个全新策略实例（X0 侧 / built 侧），参数来自 config 生产注册。"""
    from jq_framework.signals.daily_runner import build_strategy
    s1, _ = build_strategy(strat_cfg)
    s2, _ = build_strategy(strat_cfg)
    return s1, s2


def _lookback(strat_cfg: dict) -> int:
    """TSMOM 预热门槛：lookback+1 根 bar 才有方向（tsmom.on_bar 同判据）。"""
    p = (strat_cfg.get("params") or {}).get("tsmom") or {}
    return int(p.get("lookback", 250)) + 1


def audit_variety(var: str, strat_cfg: dict) -> dict:
    """单品种 X0 vs built 逐日方向比对。返回汇总 dict（含 mismatch 明细行）。"""
    x0 = get_futures_main_continuous(var, contract="X0", read_only=True)
    built = normalize_built_frame(build_contract_series(var, read_only=True))
    rec: dict = {"variety": var}
    if built is None or built.empty or "adj_close" not in built.columns:
        rec.update(status="无自建数据（X0兜底）", n_days=0, n_mismatch=0,
                   mismatch_rate=None, n_roll=None, rows=[])
        return rec
    if x0 is None or x0.empty:
        rec.update(status="X0 无数据", n_days=0, n_mismatch=0,
                   mismatch_rate=None, n_roll=None, rows=[])
        return rec

    xd = pd.to_datetime(x0["date"])
    bd = pd.to_datetime(built["date"])
    common = sorted(set(xd) & set(bd))
    # 引擎同款喂数：信号用 < dt 的 bars；carry 用 <= dt（同源注入两实例）
    carry = get_roll_yield_series(var, read_only=True)
    if carry is not None and not carry.empty:
        carry = carry.sort_values("date")
        cd = pd.to_datetime(carry["date"])
    else:
        carry, cd = None, None
    min_bars = _lookback(strat_cfg)
    strat_x0, strat_b = _strategy_instances(strat_cfg)

    n_days = n_mismatch = 0
    rows: list[dict] = []
    n_roll = int(built["is_roll"].fillna(False).astype(bool).sum()) \
        if "is_roll" in built.columns else 0
    for t in range(1, len(common)):
        dt = common[t]
        bars_x0 = x0[xd < dt]
        bars_b = built[bd < dt]
        if len(bars_x0) < min_bars or len(bars_b) < min_bars:
            continue  # 预热期：双方均 hold，非可比日
        if carry is not None:
            dfc = carry[cd <= dt]
            strat_x0.inject_carry(dfc)
            strat_b.inject_carry(dfc)
        try:
            d_x0 = strat_x0.on_bar(bars_x0).direction
            d_b = strat_b.on_bar(bars_b).direction
        except Exception:
            continue
        n_days += 1
        if d_x0 != d_b:
            n_mismatch += 1
            rows.append({"variety": var, "date": str(dt.date()),
                         "x0_dir": d_x0, "built_dir": d_b})
    rec.update(status="ok", n_days=n_days, n_mismatch=n_mismatch,
               mismatch_rate=(n_mismatch / n_days) if n_days else None,
               n_roll=n_roll, rows=rows)
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--strategy", default="tsmom_carry_cta",
                        help="比对用的生产策略（默认白名单主线）")
    parser.add_argument("--out", default="reports/futures_contract_diff.csv")
    parser.add_argument("--out-days",
                        default="reports/futures_contract_diff_days.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    strat_cfg = cfg["strategies"][args.strategy]
    symbols = cfg["futures"]["symbols"]
    varieties = [s[:-1] if s.endswith("0") else s for s in symbols]

    print(f"X0 vs built 信号差异审计（策略 {args.strategy}，read_only 只读缓存）")
    print(f"喂数语义与引擎一致：信号 < dt、carry <= dt；预热（<{_lookback(strat_cfg)} bars）跳过\n")
    results = [audit_variety(v, strat_cfg) for v in varieties]

    total_days = sum(r["n_days"] for r in results)
    total_mm = sum(r["n_mismatch"] for r in results)
    print(f"{'品种':<6}{'可比日':>8}{'不一致':>8}{'不一致率':>10}{'换月日':>8}  状态")
    for r in results:
        rate = f"{r['mismatch_rate']:.2%}" if r["mismatch_rate"] is not None else "-"
        print(f"{r['variety']:<6}{r['n_days']:>8}{r['n_mismatch']:>8}{rate:>10}"
              f"{r['n_roll'] if r['n_roll'] is not None else '-':>8}  {r['status']}")
    if total_days:
        print(f"\n合计: 可比 {total_days} 日，不一致 {total_mm} 日"
              f"（{total_mm / total_days:.2%}）")
    else:
        print("\n合计: 无可比日")

    # 不一致日明细（品种内升序；品种间按 config symbols 序）
    day_rows = [row for r in results for row in r["rows"]]
    if day_rows:
        print(f"不一致日明细 {len(day_rows)} 条（前 20 条，全量见 {args.out_days}）:")
        for row in day_rows[:20]:
            print(f"  {row['variety']:<4} {row['date']}  X0={row['x0_dir']:<5}"
                  f" built={row['built_dir']}")
    else:
        print("不一致日明细: 无")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variety", "status", "n_days", "n_mismatch", "mismatch_rate", "n_roll"]
    pd.DataFrame([{c: r.get(c) for c in cols} for r in results]).to_csv(
        out, index=False)
    out_days = Path(args.out_days)
    out_days.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(day_rows, columns=["variety", "date", "x0_dir", "built_dir"]).to_csv(
        out_days, index=False)
    print(f"\n已保存: {out}\n已保存: {out_days}")


if __name__ == "__main__":
    main()
