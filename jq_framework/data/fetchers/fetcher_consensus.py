"""fetcher_consensus.py — 分析师一致预期拉取（tushare 历史发布会时点）。

为什么:akshare 免费源(stock_profit_forecast_em/ths)只给"当前快照"(对未来年的最新预测,无历史
发布时点),做不了横截面历史门1(无每种预测的发布时点)。tushare `pro.consensus()` 返回历史不同时点
的一致预期,是唯一能做历史门1的形态(见 docs/数据拉取/data-fetchers.md)。

落盘:data/cache_consensus/consensus_{symbol}.parquet，主键 date(发布会时点)。对齐资金流增量范式。

接入共享设施（统一限流+选库）:
- 限流:Shared retry_backoff（指数退避重试）走 data/fetchers/_ratelimit.py
- 选库:注册进 SourceRegistry("consensus")；未来 akshare 快照若补全历史发布会时点可加一源

用法:
  TUSHARE_TOKEN=xxx python -m data.fetchers.fetcher_consensus --index csi1000
⚠️ consensus 是收费/较高权限接口,权限不足报错同 moneyflow QUOTA 处理。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PORT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PORT))

DATA_CONSENSUS = PORT / "data" / "cache_consensus"

# consensus 落盘列(取对门1有用的核心字段)
KEEP_COLS = ["report_date", "period", "target_price", "avg_score", "total_score",
             "pred_min", "pred_max", "pred_var"]


def _pro():
    from jq_framework.data.fetchers._source import _tushare_pro
    return _tushare_pro()


def _ts_code(symbol: str) -> str:
    return f"{symbol}.{'SH' if symbol.startswith('6') else 'SZ'}"


def _to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """tushare consensus 原生命名 → consensus_{symbol}.parquet 统一格式(主键 date=发布会时点升序)。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")
    out = {"date": df["date"], "period": df.get("period"),
           "target_price": df.get("target_price"), "avg_score": df.get("avg_score"),
           "total_score": df.get("total_score"), "pred_min": df.get("pred_min"),
           "pred_max": df.get("pred_max"), "pred_var": df.get("pred_var")}
    out = pd.DataFrame(out)
    # 同一天多券商/多次发布,按 (date, period) 保留最近一条
    out = out.drop_duplicates(subset=["date", "period"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def _fetch_consensus(pro, identifier: str, **kw) -> pd.DataFrame:
    """拉单票全部历史发布时点一致预期,返回 canonical df。consensus 无 start_date,一次拉全史本机过滤。"""
    from jq_framework.data.fetchers._ratelimit import retry_backoff
    th = kw.get("_throttle")
    code = identifier

    def _raw():
        return retry_backoff(
            lambda: pro.consensus(ts_code=_ts_code(code)),
            is_retryable=lambda e: any(k in str(e) for k in ("权限", "积分", "每分钟", "QPS", "500")),
            retries=3, base=2.0, error=Exception,
        )
    if th is not None:
        with th:
            raw = _raw()
    else:
        raw = _raw()
    if raw is None or raw.empty:
        return pd.DataFrame()
    return _to_canonical(raw)


def main():
    ap = argparse.ArgumentParser(description="tushare 一致预期深史拉取+增量(data/fetchers 收编)")
    ap.add_argument("--token", default=None, help="tushare token(缺省读 TUSHARE_TOKEN env)")
    ap.add_argument("--index", default="csi1000", help="成分股池:csi300/csi500/csi1000/csi_all")
    ap.add_argument("--out", default=None, help="落盘目录(默认 data/cache_consensus)")
    ap.add_argument("--delay", type=float, default=None, help="请求间隔秒(防 QPS，默认 0.35)")
    args = ap.parse_args()
    if args.token:
        os.environ["TUSHARE_TOKEN"] = args.token

    from jq_framework.data.fetchers._ratelimit import Throttle
    from jq_framework.data.fetchers._source import Source, REGISTRY
    from jq_framework.data.fetchers import _load_config, get_index_constituents

    out = Path(args.out) if args.out else DATA_CONSENSUS
    out.mkdir(parents=True, exist_ok=True)

    interval = args.delay if args.delay is not None else 0.35
    th = Throttle(min_interval=interval)

    _load_config()
    syms = get_index_constituents(args.index)
    print(f"共 {len(syms)} 只[{args.index}],落盘 {out}", flush=True)

    pro = _pro()
    REGISTRY.register("consensus", Source(
        name="tushare", fetch=lambda id_, **kw: _fetch_consensus(pro, id_, **kw),
        schema=tuple(KEEP_COLS), dir="data/cache_consensus",
        pattern="consensus_{symbol}.parquet",
    ))

    # 审计#12（2026-09-24 修，与 fetcher_fundflow 同款）：原 todo 只含"无文件"标的，
    # 存量文件永不更新（下方 p.exists() 合并分支不可达）。改为全部标的过一遍，
    # 合并分支按 (date, period) 去重负责幂等——重复跑不会产生重复行。
    todo = list(syms)
    print(f"共 {len(syms)} 只（含存量增量更新）", flush=True)

    import time
    t0 = time.time(); ok = 0; empty = 0; err = 0
    for i, code in enumerate(todo):
        with th:
            canon = _fetch_consensus(pro, code, _throttle=th)
        if canon is None or canon.empty:
            empty += 1; err += 1
            continue
        p = out / f"consensus_{code}.parquet"
        # 增量:已有缓存则读末日期后追加
        if p.exists():
            try:
                existing = pd.read_parquet(p)
                last = str(pd.Timestamp(existing["date"].iloc[-1]))[:10] if not existing.empty else None
                if last is not None:
                    new_part = canon[canon["date"] > last]
                    if not new_part.empty:
                        canon = pd.concat([existing, new_part], ignore_index=True).drop_duplicates(
                            subset=["date", "period"], keep="last").sort_values("date").reset_index(drop=True)
                    else:
                        canon = existing
            except Exception:
                pass
        from jq_framework.data.fetchers.fetcher import _atomic_write_parquet
        _atomic_write_parquet(p, canon)
        ok += 1
        if i % 200 == 199 or (i + 1) == len(todo):
            print(f"...{i+1}/{len(todo)} ok={ok} empty={empty} err={err} 耗时{time.time()-t0:.0f}s", flush=True)
    print(f"完成 ok={ok} empty={empty} err={err} 总耗时{time.time()-t0:.0f}s", flush=True)
    if err:
        print("⚠️ 有错误(配额/限频)。重跑跳过 err,建议提高积分或稍后重试。", flush=True)


if __name__ == "__main__":
    main()
