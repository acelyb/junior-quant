"""
数据层：akshare 拉取 A 股分钟级 K 线 + parquet 本地缓存（独立于日线 data/cache）。

与 data/fetcher.py 同范式（缓存键/原子写/read_only/批量并发/容错），但适配分钟级：
- 新浪源 stock_zh_a_minute（本环境东财源限流连不上，同 fetcher.py 用新浪源）
- 固定返回最新 1970 根 bar（滚动窗口），无 start/end 参数 → 增量=每次拉全窗口与缓存 concat 去重
- 返回列已英文（day/open/high/low/close/volume/amount），但 day≠date、volume/amount 是 str → _normalize_minute 做映射+类型转换
- 缓存目录 data/cache_min（与日线 data/cache 隔离，互不污染）

历史深度限制（重要）：
  新浪源 stock_zh_a_minute 固定返回最新 1970 根 bar（滚动窗口），5min 频率约覆盖最近 2 个月
  交易日，无法取更早历史。每次调用拉最新窗口与缓存 concat 去重，历史随时间积累。
  约 1.5 年才能攒满一个 walk-forward 训练段（252 交易日）。冷启动期（前 2 月）数据仅够
  短期 IC 检验，不适合训练定权。
"""
from __future__ import annotations

import time

import pandas as pd

# 复用 fetcher.py 的底层共享函数（消除重复，优化约束）：
#   _load_config / _cache_path / _need_refresh / _atomic_write_parquet / DataFetchError
from .fetcher import (
    DataFetchError,
    _load_config,
    _cache_path,
    _need_refresh,
    _atomic_write_parquet,
)


# 新浪分钟源限流识别：akshare 在新浪返回空/被限流响应时解析失败抛 IndexError
# （"list index out of range"）。这是暂时性故障（非数据缺失），值得短暂退避重试。
# 真·数据缺失（新股、停牌）新浪会返回空 df 而非抛 IndexError，故不混淆。
_THROTTLE_ERR = "list index out of range"
_MAX_RETRIES = 3          # 最多重试 3 次（含首次共 4 次），有界防无限占用（资源约束）
_RETRY_BACKOFF = 2.0      # 退避基数（秒），指数增长 2/4/8s，总最长等待 14s


def _sina_prefix(symbol: str) -> str:
    """akshare 新浪源 symbol 前缀：6开头=sh沪市，其余=sz深市。

    与 fetcher.py _fetch_stock_daily_raw 同逻辑（csi1000 全是沪深主板/创业板/科创板，
    无北交所，故不写 bj 分支）。csi1000 池变更前无需扩展。
    """
    return "sh" if symbol.startswith("6") else "sz"


def _normalize_minute(df: pd.DataFrame) -> pd.DataFrame:
    """统一分钟列名（新浪源中文列 → 英文标准列），date 转 datetime64[ns] 分钟精度。

    新浪源 stock_zh_a_minute 返回列已英文（day/open/high/low/close/volume/amount），
    与日线 _normalize 的差异：① day→date（分钟源叫 day 非 date）；② volume/amount 返回
    str 需转 numeric；③ date 为分钟精度（非天）；④ 列集含 amount（分钟级量价分析常用）。
    """
    df = df.copy()
    col_map = {"day": "date"}  # 其余列名已英文，只重命名 day→date
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])  # datetime64[us]→统一 astype ns
    df["date"] = df["date"].astype("datetime64[ns]")  # 统一 ns：防与日线 [ns] merge 时 dtype 不一致（历史坑）
    # volume/amount 新浪源返回 str，转 numeric（errors='coerce' 坏值→NaN，保留 NaN 不过滤）
    for c in ("volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("date").reset_index(drop=True)
    # 丢弃 volume==0 噪声行（同 fetcher._normalize L94-95：只丢精确 0，保留 NaN）
    if "volume" in df.columns:
        df = df[df["volume"] != 0].reset_index(drop=True)
    return df


def _fetch_stock_minute_raw(symbol: str, freq: str, adjust: str) -> pd.DataFrame:
    """实际向 akshare 新浪源拉取单只股票分钟 K 线，返回已 _normalize_minute 的 df。

    新浪源 stock_zh_a_minute 固定返回最新 1970 根 bar（滚动窗口），无 start/end 参数。
    含限流退避重试：新浪源对连续拉取限流时 akshare 抛 IndexError（"list index out of range"），
    属暂时性故障，短暂退避后重试（最多 _MAX_RETRIES 次）。真·数据缺失新浪返回空 df（非抛错），
    不进重试直接判失败。
    """
    try:
        import akshare as ak
    except ImportError as e:
        raise DataFetchError("未安装 akshare，请 pip install akshare") from e

    prefix = _sina_prefix(symbol)
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            df = ak.stock_zh_a_minute(symbol=f"{prefix}{symbol}", period=freq, adjust=adjust)
            if df is None or df.empty:
                raise DataFetchError(f"新浪源分钟返回空: {symbol}")
            df = _normalize_minute(df)
            return df
        except DataFetchError:
            raise  # 真·空（数据缺失），不重试
        except Exception as e:
            last_err = e
            # 非限流异常直接失败（如网络中断的某些分支），不浪费重试
            if _THROTTLE_ERR not in str(e):
                break
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))  # 2/4/8s 指数退避
    raise DataFetchError(f"拉取分钟 {symbol} 失败（限流重试 {_MAX_RETRIES} 次仍失败）: {last_err}") from last_err


def get_stock_minute(symbol: str, freq: str = "5", adjust: str = "qfq",
                     read_only: bool = False,
                     _net_lock=None, _last_call=None, _min_interval: float = 0.0) -> pd.DataFrame:
    """拉取单只股票分钟 K 线，带本地 parquet 缓存 + 滚动窗口增量更新。

    返回: DataFrame[date, open, high, low, close, volume, amount]
        date 为 datetime64[ns] 分钟精度，按时间升序。

    滚动窗口增量策略（区别于日线三段式，新浪分钟源无 start/end 参数）：
    - 有效期内（minute_max_cache_days）：直接读缓存，不碰网络（最快路径）
    - 过期且有缓存：拉最新 1970 bar 全窗口 → concat[cached, new]
        → drop_duplicates(subset=["date"], keep="last") → sort → 原子写
        （keep="last" 让新拉的窗口覆盖旧值，自然刷新 qfq 复权基准）
    - 无缓存：拉 1970 bar → 原子写
    无 full_refresh 全量重拉路径（分钟历史就 ~2 月，无"全量"概念；min_full_refresh_days
    设 >1 年实际永不触发）。

    read_only: True = cache-only（场景级进程并行用）：命中即读、绝不拉网绝不写；
        未命中回退拉取（此时该 worker 是唯一写者，无并发写竞态）。默认 False。

    _net_lock/_last_call/_min_interval: 批量级限流节流器（仅 get_batch_minute 传入）。
        网络调用经锁串行化并强制最小间隔，防多线程突发触发新浪限流。默认 None=不节流
        （单只调用无并发风险，保留快路径）。
    """
    cfg = _load_config()
    cache_dir = cfg.get("cache_min_dir", "data/cache_min")
    max_days = cfg.get("minute_max_cache_days", 1)

    # 缓存键含 freq 维度（同一标的多频率共存）；不含 start/end（新浪源无此参数）
    key = f"stock_min_{symbol}_{freq}_{adjust}"
    path = _cache_path(cache_dir, key)

    def _fetch_throttled() -> pd.DataFrame:
        """带批量级节流的拉取：持锁串行化 + 强制最小间隔，防突发限流。"""
        if _net_lock is not None:
            with _net_lock:
                elapsed = time.monotonic() - _last_call[0]
                if elapsed < _min_interval:
                    time.sleep(_min_interval - elapsed)
                _last_call[0] = time.monotonic()
        return _fetch_stock_minute_raw(symbol, freq, adjust)

    # read_only：cache-only。命中则读，跳过一切过期/增量/写逻辑；未命中回退拉取
    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        df = _fetch_throttled()
        _atomic_write_parquet(path, df)
        return df

    # 1) 快路径：缓存未过期，直接读
    if not _need_refresh(path, max_days):
        return pd.read_parquet(path)

    # 2) 滚动窗口合并路径：有缓存 → 拉最新窗口 → concat 去重 → 原子写
    if path.exists():
        try:
            cached = pd.read_parquet(path)
            new = _fetch_throttled()
            if new is not None and not new.empty:
                if cached is not None and not cached.empty:
                    merged = pd.concat([cached, new], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["date"], keep="last")
                    merged = merged.sort_values("date").reset_index(drop=True)
                else:
                    merged = new
                _atomic_write_parquet(path, merged)
                return merged
            # 新窗口为空（非交易时段等），刷新 mtime 标记本次检查过
            path.touch()
            return cached
        except Exception:
            # 合并失败回退全量拉取，保证不因增量逻辑中断
            pass

    # 3) 全量路径：无缓存 / 合并失败兜底
    df = _fetch_throttled()
    _atomic_write_parquet(path, df)
    return df


def get_batch_minute(symbols: list[str], freq: str = "5", adjust: str = "qfq",
                     progress: bool = True, read_only: bool = False) -> dict[str, pd.DataFrame]:
    """批量拉取多只股票分钟数据，返回 {symbol: DataFrame}。

    并发拉取（线程池，受 minute_fetch_workers 限制，默认 4——比日线 8 严，分钟源限流更敏感）。
    对失败的单只做容错（返回空 df），不因个别失败中断整体。同 get_batch_daily 范式。

    限流防护（新浪分钟源对连续拉取敏感）：网络调用经全局锁 + 最小间隔节流（默认 0.3s/次），
    使实际请求速率 ≤ 1/0.3≈3.3 次/s，远低于触发限流的突发阈值。read_only 命中缓存时不占节流
    （纯本地读，无需限速）。节流是批量级的，单只 get_stock_minute 调用不受影响。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    cfg = _load_config()
    workers = max(1, int(cfg.get("minute_fetch_workers", 4)))
    min_interval = float(cfg.get("minute_min_interval", 0.3))  # 网络调用最小间隔(秒)

    # 线程安全的调用节流器：网络调用串行化 + 强制最小间隔（防多线程突发触发新浪限流）
    _net_lock = threading.Lock()
    _last_call = [0.0]  # 上次网络调用时间戳（mutable 容器供闭包改写）

    out = {}
    n = len(symbols)
    done = 0

    def _one(sym: str) -> tuple[str, pd.DataFrame]:
        try:
            return sym, get_stock_minute(
                sym, freq=freq, adjust=adjust, read_only=read_only,
                _net_lock=_net_lock, _last_call=_last_call, _min_interval=min_interval)
        except Exception as e:
            if progress:
                print(f"  {sym} 失败: {str(e)[:60]}")
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            out[sym] = df
            done += 1
            if progress and (done % 100 == 0 or done == n):
                print(f"  [{done}/{n}] 已处理...")
    # 键序确定性（2026-10-13，已知口径偏差 §10）：返回按输入 symbols 序重排
    # （与 fetcher.get_batch_daily 同修，切断完成序随机键序）。
    return {sym: out[sym] for sym in symbols}
