"""共享限流/重试原语 —— 所有拉取数据逻辑统一走这里，避免各抓一手限流。

从 data/fetcher_minute.py 抽出并泛化（原 `_fetch_stock_minute_raw` 的指数退避重试 +
`get_batch_minute` 的锁串行化节流）。项目里不同数据源（akshare/tushare/efinance）共享同一
限流纪律，防止：
  - 新浪/东财/东方财富源对连续高频请求的 IP 级封禁（小时级，见 memory/sina-minute-endpoint-rate-limit）
  - tushare 的 QPS/日配额超限（moneyflow 收费档，每日调用次数有限）

两个原语：
  1. retry_backoff(fn, *, is_retryable, retries, base, on_retry) —— 单条请求内，遇"暂时性
     限流异常"指数退避重试；真·数据缺失（空结果）不进重试直接判失败。
  2. Throttle(min_interval) / batch_throttle(lock, min_interval) —— 批量并发时持锁串行化 +
     强制最小间隔，防突发限流。

用法：
  from jq_framework.data.fetchers._ratelimit import retry_backoff, Throttle

  # 单条拉取带退避重试（is_retryable 判定是否值得重试）
  df = retry_backoff(
      lambda: ak.stock_zh_a_daily(symbol=sym, adjust="qfq"),
      is_retryable=lambda e: "list index out of range" in str(e),
      retries=3, base=2.0,
  )
  # 批处理节流（跨线程/跨票共享一个 Throttle）
  th = Throttle(min_interval=0.08)
  with th:
      df = fetch_one(code)   # 保证距上一次调用≥min_interval
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class DataFetchError(Exception):
    """数据拉取失败（真·失败，非限流）。"""


def retry_backoff(fn: Callable,
                  *,
                  is_retryable: Callable[[Exception], bool],
                  retries: int = 3,
                  base: float = 2.0,
                  on_retry: Callable[[int, Exception], None] | None = None,
                  error: type[Exception] = DataFetchError,
                  retry_error: type[Exception] | None = None):
    """指数退避重试拉取。

    fn: 一次网络调用，返回结果 df / 抛异常。
    is_retryable(e): 判断异常是否属"暂时性限流/网络抖动"——是则重试，否则直接失败。
    retries: 最多重试次数（不含首次调用；+1 共 retries+1 次尝试）。
    base: 退避基数秒，等待 base * 2**attempt（2/4/8...）。
    on_retry(attempt, e): 可选，重试前回调（打日志）。
    error: 重试耗尽后抛的异常类型。
    retry_error: 限制所有重试都只抛此类型（默认用 is_retryable 区分，不用此项）。

    返回 fn 首次成功结果；重试耗尽抛 error(f"重试{retries}次仍失败: {last_err}")。
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if not is_retryable(e):
                raise
            if attempt < retries:
                wait = base * (2 ** attempt)
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(wait)
    raise error(f"拉取失败（限流重试 {retries} 次仍失败）: {last_err}") from last_err


class Throttle:
    """批量级请求节流器：持锁串行化 + 强制最小间隔，防多线程/多票突发限流。

    与 fetcher_minute 的 `_fetch_throttled` 同语义：并发拉多只票时共享一个 Throttle，
    with th: 块内的实际网络调用被强制与前一次相距 ≥ min_interval 秒。

    单线程串行拉取也可用（无并发时锁无争用，仅保留最小间隔快路径）。
    """
    def __init__(self, min_interval: float = 0.0):
        self._lock = threading.Lock()
        self._last_call = [0.0]
        self._min_interval = min_interval

    def acquire(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call[0]
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call[0] = time.monotonic()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        return False
