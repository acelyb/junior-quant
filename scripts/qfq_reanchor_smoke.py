"""qfq_reanchor_smoke.py — fetcher qfq 接缝加固冒烟（离线，mock 网络层）。

覆盖面（对应 2026-10-23 加固的两处改动）：
  1. 标记回环：_record_anchor → _anchor_fresh 为真；无标记=陈旧；
     reanchor_days=0 关闭时恒真。
  2. 空重叠守卫：增量段与缓存零重叠 → 不盲拼，落全量重锚（文件被整段替换、
     标记已记录）。
  3. 正常增量（锚新、重叠一致）→ 合并追加，不记标记（增量不重置重锚时钟）。
  4. 重叠漂移（除权/基准变化）→ 落全量重锚。
  5. 重锚超期（标记 8 天前、内容仍新）→ 直接全量，不尝试增量。

全部走 get_stock_daily 真实路径，仅 mock _fetch_stock_daily_raw（按 start 参数
区分增量/全量调用）。临时缓存目录 + QUANT_SNAPSHOT_BARRIER=0，不碰 data/cache、
不产生快照。用法：python scripts/qfq_reanchor_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["QUANT_SNAPSHOT_BARRIER"] = "0"      # 冒烟不触发改写前快照屏障
_TMP = tempfile.mkdtemp(prefix="qfq_reanchor_smoke_")
os.environ["QUANT_CACHE_DIR"] = _TMP             # 隔离缓存目录，不碰 data/cache
sys.path.insert(0, str(ROOT))

import pandas as pd

from jq_framework.data.fetchers import fetcher

DATES = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=15)
SYM, START = "000001", "2020-01-01"              # 缓存键 start（全量调用标识）
KEY = f"stock_{SYM}_qfq_{START}"
PATH = Path(_TMP) / f"{KEY}.parquet"


def mk(sl: slice, mult: float = 1.0) -> pd.DataFrame:
    """按 DATES 全局位置构帧——同一日期在任何帧里 close 相同（mult 控制整体锚定）。"""
    n = sl.stop - sl.start
    return pd.DataFrame({
        "date": DATES[sl],
        "open": [10.0 * mult] * n, "high": [10.0 * mult] * n, "low": [10.0 * mult] * n,
        "close": [10.0 * mult * (1 + 0.01 * (sl.start + i)) for i in range(n)],
        "volume": [1_000_000.0] * n,
    })


def seed_cache(sl: slice, days_ago: float = 2.0, mult: float = 1.0) -> None:
    """写入缓存并回拨 mtime（越过 max_cache_days=1 的快路径）。"""
    fetcher._atomic_write_parquet(PATH, mk(sl, mult=mult))
    old = time.time() - days_ago * 86400
    os.utime(PATH, (old, old))


calls: list[str] = []
inc_resp: list[pd.DataFrame] = [mk(slice(3, 15))]


def mock_fetch(symbol, start, end, adjust):
    calls.append(start)
    return mk(slice(0, 15)) if start == START else inc_resp[0]


fetcher._fetch_stock_daily_raw = mock_fetch


def reset(mark_anchor: float | None = None) -> None:
    """重置：清调用记录；mark_anchor=None 删标记，否则写入指定时刻。"""
    calls.clear()
    inc_resp[0] = mk(slice(3, 15))
    mp = Path(_TMP) / fetcher._REANCHOR_FILE
    if mark_anchor is None:
        mp.unlink(missing_ok=True)
    else:
        fetcher._record_anchor(_TMP, KEY)
        ts = time.time() - mark_anchor
        times = fetcher._load_anchor_times(_TMP)
        times[KEY] = ts
        mp.write_text(__import__("json").dumps(times), encoding="utf-8")


# ---- 1. 标记回环 ----
reset()
assert not fetcher._anchor_fresh(_TMP, KEY, {}), "无标记应视为陈旧"
fetcher._record_anchor(_TMP, KEY)
assert fetcher._anchor_fresh(_TMP, KEY, {}), "刚记录的标记应为新"
assert fetcher._anchor_fresh(_TMP, KEY, {"reanchor_days": 0}), "reanchor_days=0 应恒真"
reset(mark_anchor=8 * 86400)
assert not fetcher._anchor_fresh(_TMP, KEY, {}), "8 天前标记应视为陈旧"
print("  [smoke] 标记回环 OK")

# ---- 2. 空重叠 → 全量重锚（不盲拼）----
reset()                       # 无标记 → 本就应全量；先补标记让增量路径可达再测空重叠
fetcher._record_anchor(_TMP, KEY)
seed_cache(slice(0, 10))
inc_resp[0] = mk(slice(11, 15))              # 与缓存 [0:10] 零重叠
df = fetcher.get_stock_daily(SYM, start=START)
assert START in calls, "零重叠必须落全量（锚定身份无从校验）"
assert len(df) == 15 and len(pd.read_parquet(PATH)) == 15, "文件应被全量整段替换"
assert fetcher._anchor_fresh(_TMP, KEY, {}), "全量完成应记录重锚标记"
print("  [smoke] 空重叠守卫 OK（落全量重锚）")

# ---- 3. 锚新 + 重叠一致 → 正常合并，不记标记 ----
reset()
fetcher._record_anchor(_TMP, KEY)            # 锚新
seed_cache(slice(0, 10))
ts_before = fetcher._load_anchor_times(_TMP)[KEY]
df = fetcher.get_stock_daily(SYM, start=START)
assert START not in calls and calls, "应走增量（一次网络调用）"
assert len(df) == 15, "增量合并应补齐到 15 行"
assert fetcher._load_anchor_times(_TMP)[KEY] == ts_before, "增量合并不得重置重锚时钟"
print("  [smoke] 正常增量合并 OK（标记时钟不动）")

# ---- 4. 重叠漂移 → 全量重锚 ----
reset()
fetcher._record_anchor(_TMP, KEY)
seed_cache(slice(0, 10))
inc_resp[0] = mk(slice(3, 15), mult=1.02)    # 重叠日 close 差 2% > 1e-6
df = fetcher.get_stock_daily(SYM, start=START)
assert START in calls, "重叠漂移必须落全量"
assert len(df) == 15
print("  [smoke] 重叠漂移守卫 OK（既有行为保持）")

# ---- 5. 重锚超期 → 直接全量，不尝试增量 ----
reset(mark_anchor=8 * 86400)                 # 标记 8 天前（内容仍新）
seed_cache(slice(0, 10))
df = fetcher.get_stock_daily(SYM, start=START)
assert calls == [START], f"超期应直接全量（实际调用 {calls}）"
assert fetcher._anchor_fresh(_TMP, KEY, {}), "全量后标记应刷新"
print("  [smoke] 重锚周期强制 OK（超期直落全量）")

print("\n== qfq 接缝加固冒烟 全部通过 ==")
