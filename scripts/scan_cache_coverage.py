"""scan_cache_coverage.py — 扫描 qfq 日线缓存覆盖率, 定位缺失/滞后数据。

目的: 确认把长历史 walk-forward 回测延伸到 2025-12-31(或更远)时, 成分股日线数据
是否齐全、滞后到哪。为"先补缺再延伸"提供精确清单。

策略
----
生产 get_stock_daily 走 config cache_dir(默认 data/cache), 键 stock_{sym}_qfq_{start}。
每个 symbol 有多个历史快照文件(start 不同/带废弃 end 的旧键), 尾端日期参差
(增量更新时点不同)。对每个 symbol:
  - 扫描其所有 stock_{sym}_qfq_*.parquet 文件
  - 统计每文件的 最早/最晚/行数
  - 该 symbol 当前数据末端 = 所有文件中最大的最晚日期 max_end
  - 该 symbol 全史跨度 = 覆盖最早的可行文件(最早 start 且行数最大的)
统计指标(核心关注当前数据是否够延伸回测):
  - 每 symbol 的 max_end 分布: 多少只到 2026-09(最新/目标)、多少滞后、滞后到哪
  - history 起点分布: 多少只能回 2018(长历史 2018-2025 需要)
  - 指定日期(如 2025-12-31)是否被所有 symbol 覆盖: last<该日 = 缺该段

用法
----
  python scripts/scan_cache_coverage.py --universe csi1000          # 扫描 csi1000 成分
  python scripts/scan_cache_coverage.py --asof 2025-12-31           # 报告能否覆盖到该日
  python scripts/scan_cache_coverage.py --universe csi_all --asof 2026-09-01
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import glob
import os
import pandas as pd


def _max_end_for_symbol(cache_dir: Path, sym: str) -> tuple[pd.Timestamp, int, int]:
    """返回该 symbol 的 (最大最晚日期, 行数最大的文件行数, 找到的文件数)."""
    files = sorted(cache_dir.glob(f"stock_{sym}_qfq_*.parquet"))
    max_end = None
    best_rows, best_span = 0, 0
    for p in files:
        try:
            df = pd.read_parquet(p, columns=["date"])
        except Exception:
            continue
        if df is None or df.empty:
            continue
        d = pd.to_datetime(df["date"])
        end, start, n = d.max(), d.min(), len(df)
        if max_end is None or end > max_end:
            max_end = end
        span = int((end - start).days)
        if n > best_rows and span >= best_span:
            best_rows, best_span, = n, span
    return max_end, best_rows, len(files)


def main():
    ap = argparse.ArgumentParser(description="扫描 qfq 日线缓存覆盖率")
    ap.add_argument("--universe", default="csi1000")
    ap.add_argument("--asof", default=None, help="目标覆盖日 YYYY-MM-DD(检查是否所有成分都到该日)")
    ap.add_argument("--history-since", default="2018-01-01", help="长历史起点(检查能否回该起点)")
    ap.add_argument("--sample", type=int, default=0, help=">0 则只扫前 N 只(调试)")
    args = ap.parse_args()

    import yaml
    from jq_framework.data.fetchers.fetcher import get_index_constituents, _load_config

    _load_config()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cache_dir = Path(cfg.get("cache_dir", "data/cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    print(f"扫描缓存目录: {cache_dir}", flush=True)

    syms = get_index_constituents(args.universe)
    if args.sample > 0:
        syms = syms[: args.sample]
    print(f"universe={args.universe} 成分 {len(syms)} 只", flush=True)

    asof = pd.Timestamp(args.asof) if args.asof else None
    since = pd.Timestamp(args.history_since)

    # 分类统计
    to_asof = []          # max_end >= asof
    not_to_asof = []      # (sym, max_end, diff_days)
    not_to_since = []     # 起点太晚(最早 start 晚于 history_since)
    rows = []

    for i, sym in enumerate(syms):
        max_end, best_rows, n_files = _max_end_for_symbol(cache_dir, sym)
        if max_end is None:
            rows.append({"sym": sym, "max_end": None, "files": n_files, "note": "无日线文件"})
            not_to_asof.append((sym, None, None))
            continue
        rows.append({"sym": sym, "max_end": max_end, "files": n_files,
                     "end_str": max_end.strftime("%Y-%m-%d")})
        if asof is not None and max_end < asof:
            not_to_asof.append((sym, max_end, (asof - max_end).days))
        if max_end < since:
            not_to_since.append((sym, max_end))
        if (i + 1) % 200 == 0:
            print(f"  扫描 {i+1}/{len(syms)} ...", flush=True)

    print("\n" + "=" * 70)
    print(f"扫描完成: {len(syms)} 只")

    # 当前数据末端分布
    ends = [r["max_end"] for r in rows if r["max_end"] is not None]
    if ends:
        print("\n[当前数据末端 max_end 分布]")
        print(f"  最早: {min(ends).date()}   最晚: {max(ends).date()}")
        # 按年末聚合
        from collections import Counter
        yrs = Counter(d.year for d in ends)
        print("  按年:", dict(sorted(yrs.items())))
        lags = [(d.year, d.strftime("%m")).count for _ in []]  # placeholder
        # 滞后最多的几只
        if asof is not None:
            lagged = sorted([(s, e, (asof - e).days) for s, e, _ in not_to_asof
                             if e is not None], key=lambda x: -x[2])
            print(f"\n[未能覆盖 {asof.date()} 的 symbol] {len(lagged)} 只:")
            print("   " + ", ".join(f"{s}(滞后{d}天@{e.date()})" for s, e, d in lagged[:30]))
            if len(lagged) > 30:
                print(f"   ... 其余 {len(lagged)-30} 只")

    if rows:
        df = pd.DataFrame(rows)
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        csv = out / f"cache_coverage_{args.universe}.csv"
        df.to_csv(csv, index=False)
        print(f"\n已写 {csv}")

    # 长历史起点检查
    print(f"\n[长历史起点检查 history_since={since.date()}]")
    ok_since = [r for r in rows if r["max_end"] is not None]
    print(f"  max_end >= {since.date()}: {len(ok_since)}/{len(syms)}")
    print(f"  起点太晚/local 早于 {since.date()}: {len(not_to_since)} 只 "
          + ("" if not not_to_since else f"(如 {not_to_since[-3:]})"))


if __name__ == "__main__":
    main()
