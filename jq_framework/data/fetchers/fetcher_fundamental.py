"""
基本面数据层：估值(日频) + 财务指标(季频)，akshare 拉取 + parquet 缓存。

防未来函数核心（最重要）：
- 估值用 stock_value_em，"数据日期"=交易日，天然与行情日同频对齐，无未来函数
- 财务用 stock_financial_analysis_indicator_em，必须用 NOTICE_DATE(公告披露日) 对齐行情日，
  绝不用 REPORT_DATE(报告期)——报告期 3-31 的财报通常 4 月底才公告，4 月前用即偷看未来
- build_fundamental_data 用 merge_asof(direction='backward') 把财务按公告日对齐到交易日，
  非公告日财务字段为 NaN，再 ffill 用最近一次公告值填充。因子层只需取 <= 当前 date 的最新行

缓存模式照搬 data/fetcher.py：parquet 存 data/cache，复用 _cache_path/_need_refresh/_load_config。
财务数据非日更，有效期用 config.data.fin_cache_days（默认 7 天）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

try:
    from jq_framework.data.fetchers import snapshot as _snapshot
except ImportError:  # 脚本直跑（无包上下文）
    from snapshot import _snapshot  # type: ignore

from .fetcher import _cache_path, _need_refresh, _load_config, _atomic_write_parquet, DataFetchError


# ============ 工具 ============

def _to_em_suffix(symbol: str) -> str:
    """6位码转东财后缀：6开头.SH，其余.SZ。财务接口必须带后缀（.SS 会报错）。"""
    return f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"


def _fin_cache_days() -> int:
    """基本面数据缓存有效期（天），默认 7。"""
    cfg = _load_config()
    return int(cfg.get("fin_cache_days", 7))


# ============ 估值（日频，天然无未来函数）============

# stock_value_em 中文列 → 英文标准列
_VAL_COL_MAP = {
    "数据日期": "date",
    "总市值": "market_cap",
    "流通市值": "circ_market_cap",
    "PE(TTM)": "pe_ttm",
    "PE(静)": "pe_static",
    "市净率": "pb",
    "市销率": "ps",
}


def get_valuation(symbol: str, read_only: bool = False) -> pd.DataFrame:
    """
    拉取单只股票估值序列（日频）。

    返回: DataFrame[date(datetime), market_cap, circ_market_cap, pe_ttm, pe_static, pb, ps]
    缓存: fin_value_em_{sym}.parquet，有效期 fin_cache_days（默认7天）

    read_only: True = cache-only（命中即读、不拉网不写）；未命中回退拉取。见 plans 点3。
    """
    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    path = _cache_path(cache_dir, f"fin_value_em_{symbol}")

    # 严格 cache-only（钉住回测用，QUANT_STRICT_CACHE=1）：命中即读、未命中 fail-loud，
    # 绝不回退拉网——与 read_only 的"未命中回退拉取"语义不同，见 data/fetchers/snapshot.py
    if cfg.get("strict_cache"):
        if not path.exists():
            raise _snapshot.StrictCacheMiss(f"严格缓存模式：快照缺 {path.name}（估值 {symbol}）")
        return pd.read_parquet(path)

    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        # 未命中回退拉取（预热应已写）
    elif not _need_refresh(path, _fin_cache_days()):
        return pd.read_parquet(path)

    try:
        import akshare as ak
        df = ak.stock_value_em(symbol=symbol)
        if df is None or df.empty:
            raise DataFetchError(f"估值接口返回空: {symbol}")
        # 只保留需要的列并重命名
        keep = [c for c in _VAL_COL_MAP if c in df.columns]
        df = df[keep].rename(columns={k: v for k, v in _VAL_COL_MAP.items() if k in keep})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"拉取估值 {symbol} 失败: {e}") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(path, df)
    return df


# ============ 财务指标（季频，用 NOTICE_DATE 防未来函数）============

# stock_financial_analysis_indicator_em 列 → 英文标准列
_FIN_COL_MAP = {
    "NOTICE_DATE": "date",            # 公告披露日（防未来函数用这个）
    "REPORT_DATE": "report_date",     # 报告期（仅作标签，不参与对齐）
    "ROEJQ": "roe",                   # ROE 加权(%)
    "TOTALOPERATEREVETZ": "rev_yoy",  # 营收同比增速
    "PARENTNETPROFITTZ": "netprofit_yoy",  # 归母净利同比增速
}


def get_financial_indicator(symbol: str, read_only: bool = False) -> pd.DataFrame:
    """
    拉取单只股票季度财务指标。

    symbol: 6位码（内部转 .SH/.SZ）
    返回: DataFrame[date(=NOTICE_DATE,datetime), report_date, roe, rev_yoy, netprofit_yoy]
    缓存: fin_indicator_em_{sym}.parquet，有效期 fin_cache_days（默认7天）

    注意：date 用公告日 NOTICE_DATE，绝不用报告期 REPORT_DATE 对齐行情

    read_only: True = cache-only（命中即读、不拉网不写）；未命中回退拉取。见 plans 点3。
    """
    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    path = _cache_path(cache_dir, f"fin_indicator_em_{symbol}")

    # 严格 cache-only（钉住回测用，QUANT_STRICT_CACHE=1）：命中即读、未命中 fail-loud
    if cfg.get("strict_cache"):
        if not path.exists():
            raise _snapshot.StrictCacheMiss(f"严格缓存模式：快照缺 {path.name}（财务指标 {symbol}）")
        return pd.read_parquet(path)

    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        # 未命中回退拉取（预热应已写）
    elif not _need_refresh(path, _fin_cache_days()):
        return pd.read_parquet(path)

    try:
        import akshare as ak
        df = ak.stock_financial_analysis_indicator_em(
            symbol=_to_em_suffix(symbol), indicator="按报告期")
        if df is None or df.empty:
            raise DataFetchError(f"财务指标接口返回空: {symbol}")
        keep = [c for c in _FIN_COL_MAP if c in df.columns]
        df = df[keep].rename(columns={k: v for k, v in _FIN_COL_MAP.items() if k in keep})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        # 丢弃公告日缺失的记录：无 NOTICE_DATE 无法对齐行情日，
        # 且 merge_asof 要求右表 merge key 无空值，否则整只股票合并失败
        df = df.dropna(subset=["date"])
        # 按公告日排序去重：同一公告日可能多条修正，保留最后一条（最新修正）
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        df = df.reset_index(drop=True)
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"拉取财务指标 {symbol} 失败: {e}") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(path, df)
    return df


# ============ 合并构建基本面数据（防未来函数核心落点）============

def build_fundamental_data(symbols: list[str], progress: bool = True,
                           read_only: bool = False) -> dict[str, pd.DataFrame]:
    """
    批量构建 {symbol: 合并基本面 df}，供基本面因子注入。

    对每只股票：
    1. 拉估值(日频, index=date) + 财务(季频, index=date=公告日)
    2. merge_asof(direction='backward') 把财务按公告日对齐到估值交易日
       —— 确保某交易日只能看到 <= 当日已公告的财报，这就是防未来函数的落点
    3. 财务字段在非公告日为 NaN → ffill 用最近一次公告值填充

    并发拉取（ThreadPoolExecutor，受 config.fetch_workers 限制），单只失败返回空 df 不中断。

    read_only: True = cache-only（透传 get_valuation/get_financial_indicator）。
        主进程预热后 worker 用此模式，杜绝并发写 parquet 竞态。见 plans 点3。
    """
    cfg = _load_config()
    workers = max(1, int(cfg.get("fetch_workers", 8)))

    out: dict[str, pd.DataFrame] = {}
    n = len(symbols)
    done = 0

    def _one(sym: str) -> tuple[str, pd.DataFrame]:
        try:
            val = get_valuation(sym, read_only=read_only)
            fin = get_financial_indicator(sym, read_only=read_only)
            if val is None or val.empty:
                return sym, pd.DataFrame()
            # 统一 date 精度（估值 ms / 财务 us 不一致，merge_asof 要求完全相同）
            val = val.copy()
            val["date"] = pd.to_datetime(val["date"]).astype("datetime64[ns]")
            merged = val.sort_values("date").reset_index(drop=True)
            if fin is not None and not fin.empty:
                fin = fin.copy()
                fin["date"] = pd.to_datetime(fin["date"], errors="coerce").astype("datetime64[ns]")
                fin = fin.dropna(subset=["date"])  # merge_asof 要求右表 key 无空值
                fin = fin.sort_values("date").reset_index(drop=True)
                merged = pd.merge_asof(merged, fin, on="date", direction="backward")
                # 财务字段在非公告日为 NaN，用最近一次公告值填充
                fin_cols = [c for c in ["roe", "rev_yoy", "netprofit_yoy", "report_date"]
                            if c in merged.columns]
                merged[fin_cols] = merged[fin_cols].ffill()
            return sym, merged
        except Exception as e:
            if progress:
                print(f"  {sym} 基本面失败: {str(e)[:60]}")
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            out[sym] = df
            done += 1
            if progress and (done % 50 == 0 or done == n):
                print(f"  [{done}/{n}] 基本面已处理...")
    # 键序确定性（2026-10-13，已知口径偏差 §10）：上方 as_completed 按线程完成序回填，
    # 键序逐调用随机；返回按输入 symbols 序重排（与 fetcher.get_batch_daily 同修）。
    return {sym: out[sym] for sym in symbols}


if __name__ == "__main__":
    # 自测：拉一只看合并结果
    print("构建 600519 基本面数据 ...")
    fund = build_fundamental_data(["600519"], progress=False)
    df = fund["600519"]
    print(f"  行数={len(df)}  日期范围={df['date'].iloc[0].date()}~{df['date'].iloc[-1].date()}")
    print(f"  列={list(df.columns)}")
    print(df[["date", "market_cap", "pe_ttm", "pb", "roe", "rev_yoy"]].tail(3).to_string())
