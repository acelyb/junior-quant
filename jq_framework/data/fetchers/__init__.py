"""data.fetchers —— 项目统一数据拉取层。

所有拉取数据的功能统一在此模块：
  - fetcher.py              akshare 主行情（A股/期货日线、指数成分、行业映射）
  - fetcher_fundamental.py  akshare 基本面（估值/财务指标）
  - fetcher_minute.py       akshare 分钟线（新浪源）
  - fetcher_fundflow.py     资金流（efinance 浅史 / tushare 深史，按 source 选）
  - fetcher_consensus.py    一致预期（tushare 历史发布会时点）
  - _ratelimit.py           共享限流/重试原语（所有源共用）
  - _source.py              选库逻辑（同一数据类可多数据源）

对外主入口直接在此 re-export，调用方 `from data.fetchers import get_stock_daily` 即可，
无需关心内部按源 vs 直接拉取的分工。

选库：资金流/一致预期等可多源数据，见 `sources_of(data_class)` / `SourceRegistry`。
"""
from __future__ import annotations

# ---- 主行情（akshare）----
from .fetcher import (
    DataFetchError,
    _load_config,
    _cache_path,
    _need_refresh,
    _atomic_write_parquet,
    INDEX_SYMBOLS,
    COMPOSITE_UNIVERSES,
    get_stock_daily,
    get_futures_daily,
    load_symbol,
    get_index_constituents,
    get_batch_daily,
    get_index_daily,
    get_index_real_daily,
    get_stock_industry_map,
)
# ---- 基本面（akshare）----
from .fetcher_fundamental import (
    get_valuation,
    get_financial_indicator,
    build_fundamental_data,
)
# ---- 分钟线（akshare 新浪源）----
from .fetcher_minute import (
    get_stock_minute,
    get_batch_minute,
)

# ---- 选库 ----
from ._source import sources_of, SourceRegistry, REGISTRY as SOURCE_REGISTRY

__all__ = [
    "DataFetchError", "_load_config", "_cache_path", "_need_refresh",
    "_atomic_write_parquet", "INDEX_SYMBOLS", "COMPOSITE_UNIVERSES",
    "get_stock_daily", "get_futures_daily", "load_symbol",
    "get_index_constituents", "get_batch_daily", "get_index_daily",
    "get_index_real_daily", "get_stock_industry_map",
    "get_valuation", "get_financial_indicator", "build_fundamental_data",
    "get_stock_minute", "get_batch_minute",
    "sources_of", "SourceRegistry", "SOURCE_REGISTRY",
]
