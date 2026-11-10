"""fetcher_fundflow.py — 主力资金流统一拉取，按 source 选 efinance(浅史) / tushare(深史)。

同一 canonical 落盘（对齐 fundflow_schema.CANONICAL_COLS）：fundflow_offline/data_fundflow/ff_{symbol}.parquet
efinance 免费浅史(最近~120日) 与 tushare 深史(2018+,需 token/积分) 写**同目录同名文件**，用
merge_fundflow 按 date 覆盖互相续 → 无缝合成 2018-2026 全史 → fundflow_factors.py 直接消费。

设计（选库+存储兼容示范）：
- 选库：SourceRegistry 注册 fundflow 的 efinance/tushare 两个 Source；--source 选择，缺省 tushare 优先
- 限流：Shared retry_backoff（指数退避重试）+ Throttle（批量最小间隔）统一走 _ratelimit
- 增量：读已有 parquet 末日期 → 拉 last_date+1 之后 → merge_fundflow(existing, new)
- 断点续：跳过已有 ff_*.parquet；连续失败(封禁)暂停等待
- 防呆：增量模式若静默续浅史而最早日期 < 深史起点，提示需 --full

用法：
  # 全量深史（首次，2018 起，需 TUSHARE_TOKEN）：
  TUSHARE_TOKEN=xxx python -m data.fetchers.fetcher_fundflow --source tushare --index csi1000 --full
  # 日增（eFinance 浅史兜底，免费）：
  python -m data.fetchers.fetcher_fundflow --source efinance --index csi1000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

PORT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PORT))

# 存储契约（唯一真源）：fundflow_schema 已写好 canonical 列 + tushare 列映射 + merge。
# 本项目路径在根:fundflow_offline/fundflow_schema.py
FUNDFLOW_ROOT = PORT / "fundflow_offline"
DATA_FUNDFLOW = FUNDFLOW_ROOT / "data_fundflow"

def _import_schema():
    sys.path.insert(0, str(FUNDFLOW_ROOT))
    from fundflow_schema import CANONICAL_COLS, TUSHARE_ALIAS, merge_fundflow
    return CANONICAL_COLS, merge_fundflow

# 防封/限流参数（对齐原 fetch_fundflow_efinance.py）
BASE_DELAY = 0.08        # efinance 防封基础间隔
MAX_CONSEC_ERR = 15      # 连续错误达阈值视为封禁，暂停
PAUSE_SECONDS = 60       # 封禁暂停时长
# tushare 深史起点
DEEP_START = "20180101"
THROTTLE = 0.35          # tushare QPS 防抖


# ============ 实现：efinance 浅史 / tushare 深史 ============

_EFINANCE_COL_MAP = {
    "日期": "date", "收盘价": "close", "涨跌幅": "pct_chg",
    "主力净流入": "main_amt", "主力净流入占比": "main_pct",
    "超大单净流入占比": "xlarge_pct", "大单流入净占比": "large_pct",
    "中单流入净占比": "mid_pct", "小单流入净占比": "small_pct",
}


def _fetch_efinance(identifier: str, **kw) -> pd.DataFrame:
    """efinance 拉单票最近~120日浅史资金流，返回 canonical df。"""
    import efinance as ef
    code = identifier
    df = ef.stock.get_history_bill(code)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_EFINANCE_COL_MAP)
    from fundflow_offline.fundflow_schema import CANONICAL_COLS
    for c in CANONICAL_COLS:
        if c not in df.columns:
            df[c] = float("nan")
    df = df[CANONICAL_COLS].copy()
    for c in CANONICAL_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").reset_index(drop=True)


def _ts_code(symbol: str) -> str:
    """6位代码 → tushare ts_code（沪 6 开头，含 688 科创板沪市；余为深）。"""
    return f"{symbol}.{'SH' if symbol.startswith('6') else 'SZ'}"


def _fetch_tushare_one(pro, ts_code: str, start: str | None) -> pd.DataFrame:
    """tushare moneyflow 拉单票 [start, today]（None=全史），返回原生 df（列映射留给 merge）。"""
    from jq_framework.data.fetchers._ratelimit import retry_backoff
    return retry_backoff(
        lambda: pro.moneyflow(ts_code=ts_code, **({"start_date": start} if start else {})),
        is_retryable=lambda e: any(k in str(e) for k in ("权限", "积分", "每分钟", "QPS", "500")),
        retries=3, base=2.0,
        error=Exception,
    )


def _fetch_tushare(pro, identifier: str, start: str | None, **kw) -> pd.DataFrame:
    """tushare 拉单票资金流 → canonical 列名 + % 值（net_mf_pct 已 /100，对齐 schema）。
    返回 canonical df，两源同构，供 merge 直接使用（merge alias=None）。
    """
    from jq_framework.data.fetchers._ratelimit import Throttle
    from fundflow_offline.fundflow_schema import CANONICAL_COLS, TUSHARE_ALIAS
    th = kw.get("_throttle")
    code = identifier
    if th is not None:
        with th:
            raw = _fetch_tushare_one(pro, _ts_code(code), start)
    else:
        raw = _fetch_tushare_one(pro, _ts_code(code), start)
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    # tushare moneyflow 返回 buy/sell 各档金额(万元，如 buy_lg_amount) + 主力净额 net_mf_amount。
    # 不直接给 main_pct 占比列(东财给，见 _EFINANCE_COL_MAP)，故从金额自算占比，与东财同口径：
    #   main_pct = 主力净额 / 单边总成交额 × 100（东财"净流入/成交额%"）
    # 单位统一为万元即可(占比无量纲)。
    amount_cols = ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"]
    if all(c in raw.columns for c in amount_cols):
        single_side_total = raw[amount_cols].sum(axis=1)
        raw["net_mf_amount"] = pd.to_numeric(raw["net_mf_amount"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            raw["main_pct"] = (raw["net_mf_amount"] / single_side_total * 100.0).replace([np.inf, -np.inf], np.nan)
        # 各档净流入占比: (buy档 - sell档) / 单边总额 × 100
        for pct, (bc, sc) in {"xlarge_pct": ("buy_elg_amount", "sell_elg_amount"),
                              "large_pct": ("buy_lg_amount", "sell_lg_amount"),
                              "mid_pct": ("buy_md_amount", "sell_md_amount"),
                              "small_pct": ("buy_sm_amount", "sell_sm_amount")}.items():
            buy = pd.to_numeric(raw.get(bc), errors="coerce")
            sell = pd.to_numeric(raw.get(sc), errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                raw[pct] = ((buy - sell) / single_side_total * 100.0).replace([np.inf, -np.inf], np.nan)
    if "close" in raw.columns:
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    # tushare 原生列名 → canonical（用 schema 的 TUSHARE_ALIAS；展开成 canonical df 供 merge alias=None）
    raw = raw.rename(columns=TUSHARE_ALIAS)
    for c in CANONICAL_COLS:
        if c not in raw.columns:
            raw[c] = float("nan")
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    return raw[CANONICAL_COLS].sort_values("date").reset_index(drop=True)


def _last_date_of(p: Path) -> str | None:
    """读已有资金流缓存末日期(YYYY-MM-DD)，无/空返回 None。增量起点。"""
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if df.empty or "date" not in df.columns:
            return None
        return str(pd.Timestamp(df["date"].iloc[-1]))[:10]
    except Exception:
        return None


# ============ 注册选库 + 主流程 ============

def _register_sources(pro=None):
    """注册 fundflow 的两个源。efinance 不需外部客户端；tushare 需 pro（惰性构造）。"""
    from jq_framework.data.fetchers._source import Source, REGISTRY
    CANONICAL, _ = _import_schema()

    def efin_latest(code: str) -> pd.DataFrame:
        return _fetch_efinance(code)

    def tushare_from(pro_):
        def f(identifier: str, start: str | None = None, **kw) -> pd.DataFrame:
            return _fetch_tushare(pro_, identifier, start, **kw)
        return f

    REGISTRY.register("fundflow", Source(
        name="efinance", fetch=efin_latest,
        schema=tuple(CANONICAL), dir="fundflow_offline/data_fundflow",
        pattern="ff_{symbol}.parquet",
    ))
    if pro is not None:
        REGISTRY.register("fundflow", Source(
            name="tushare", fetch=tushare_from(pro),
            schema=tuple(CANONICAL), dir="fundflow_offline/data_fundflow",
            pattern="ff_{symbol}.parquet",
        ))


def _throttle_for(src_name: str, override_delay: float | None = None):
    from jq_framework.data.fetchers._ratelimit import Throttle
    base = THROTTLE if src_name == "tushare" else BASE_DELAY
    return Throttle(min_interval=override_delay if override_delay is not None else base)


def main():
    ap = argparse.ArgumentParser(description="主力资金流统一拉取（efinance浅史/tushare深史，按source选）")
    ap.add_argument("--source", default="tushare", choices=["tushare", "efinance"],
                    help="数据源：tushare深史(需token)/efinance浅史(免费)")
    ap.add_argument("--token", default=None, help="tushare token(缺省读 TUSHARE_TOKEN env)")
    ap.add_argument("--index", default="csi_all", help="指数成分池：csi300/csi500/csi1000/csi_all")
    ap.add_argument("--all", action="store_true", help="全A(新浪实时)而非指数池——扫全A 5219只")
    ap.add_argument("--out", default=None, help="落盘目录(默认 fundflow_offline/data_fundflow)")
    ap.add_argument("--full", action="store_true", help="全量从 DEEP_START 拉(默认=增量:只拉每只末日期之后)")
    ap.add_argument("--start", default=DEEP_START, help="深史起点(YYYYMMDD)，仅 tushare 全量使用")
    ap.add_argument("--delay", type=float, default=None, help="请求间隔秒(覆盖默认)")
    args = ap.parse_args()
    if args.token:
        os.environ["TUSHARE_TOKEN"] = args.token

    # tushare 需 pro 客户端（token/网关走环境变量，见 _source._tushare_pro）
    pro = None
    if args.source == "tushare":
        from jq_framework.data.fetchers._source import _tushare_pro
        try:
            pro = _tushare_pro()
        except ImportError:
            raise SystemExit("未安装 tushare，请 pip install tushare")

    _register_sources(pro)
    from jq_framework.data.fetchers._source import REGISTRY
    src = REGISTRY.select("fundflow", args.source)

    out = Path(args.out) if args.out else PORT / src.dir
    out.mkdir(parents=True, exist_ok=True)

    from jq_framework.data.fetchers import get_index_constituents, _load_config
    _load_config()
    if args.all:
        from scripts.backfill_all_a_daily import get_all_a_symbols
        syms = get_all_a_symbols()
    else:
        syms = get_index_constituents(args.index)
    print(f"共 {len(syms)} 只[{'all' if args.all else args.index}] 源={args.source} 落盘={out} full={args.full}", flush=True)

    # 断点续/增量（审计#12 修复，2026-09-24）：原实现增量模式 todo 只含"无文件"标的，
    # 存量文件永不追加（last 合并分支不可达，与 docstring 宣称相反）。现改为全部标的过一遍：
    # 文件新鲜（末日期距今 ≤4 日历日，容周末/短假）跳过，否则从末日期+1 增量拉取并合并。
    # 首次回填的断点续跑语义保持——已拉且新鲜的文件同样跳过。
    if args.full:
        todo = list(syms)
    else:
        fresh_cut = pd.Timestamp.today().normalize() - pd.Timedelta(days=4)
        todo = []
        for s in syms:
            fp = out / f"ff_{s}.parquet"
            if fp.exists():
                last_d = _last_date_of(fp)
                if last_d is not None and pd.Timestamp(last_d) >= fresh_cut:
                    continue  # 已新鲜，跳过
            todo.append(s)
    print(f"共 {len(syms)} 只, 新鲜跳过 {len(syms) - len(todo)} 只, 本次处理 {len(todo)} 只", flush=True)

    th = _throttle_for(args.source, args.delay)
    t0 = time.time(); ok = 0; empty = 0; err = 0; consec_err = 0
    for i, code in enumerate(todo):
        p = out / f"ff_{code}.parquet"
        last = None if args.full or not p.exists() else _last_date_of(p)
        start = None if args.source != "tushare" or last is None else \
            (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        try:
            if args.source == "tushare":
                with th:
                    new = src.fetch(code, start=start) if start else src.fetch(code)
            else:
                with th:
                    new = src.fetch(code)
            if new is None or (isinstance(new, pd.DataFrame) and new.empty):
                empty += 1; consec_err += 1
            else:
                from fundflow_offline.fundflow_schema import merge_fundflow
                if last is not None and p.exists():
                    existing = pd.read_parquet(p)
                    new = merge_fundflow(existing, new, alias=None)  # 两源已 canonical，无需再映射
                from jq_framework.data.fetchers.fetcher import _atomic_write_parquet
                _atomic_write_parquet(p, new)
                ok += 1; consec_err = 0
        except Exception as e:
            msg = str(e)
            empty += 1; err += 1; consec_err += 1
            if "权限" in msg or "积分" in msg or "QPS" in msg:
                print(f"  [{i+1}/{len(todo)}] {code} 配额/权限不足: {msg[:60]}", flush=True)
        # 封禁暂停(efinance 连续失败)/tushare 限流退避
        if args.source == "efinance" and consec_err >= MAX_CONSEC_ERR:
            print(f"...连续{consec_err}失败,疑似封禁,暂停{PAUSE_SECONDS}s", flush=True); time.sleep(PAUSE_SECONDS); consec_err = 0
        if i % 200 == 199 or (i + 1) == len(todo):
            print(f"...{i+1}/{len(todo)} ok={ok} empty={empty} err={err} 耗时{time.time()-t0:.0f}s", flush=True)
    # 防呆：增量模式最早日期远晚于深史起点 → 提示 --full
    if not args.full:
        try:
            shots = [p for p in out.glob("ff_*.parquet") if p.stat().st_size > 0][:50]
            if shots:
                first = min(pd.read_parquet(p)["date"].min() for p in shots)
                if str(first) > args.start[:4] + "-" + args.start[4:6]:
                    print(f"⚠️ 增量最早={first}，未到深史起点({args.start})。首次拉取应改用 --full", flush=True)
        except Exception:
            pass
    print(f"完成 ok={ok} empty={empty} err={err} 总耗时{time.time()-t0:.0f}s", flush=True)
    if err:
        print("⚠️ 有错误(配额/限频/封禁)。重跑跳过 err，建议稍后重试或换源。", flush=True)



if __name__ == "__main__":
    main()
