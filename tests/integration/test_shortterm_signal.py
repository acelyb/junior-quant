"""tests/futures/test_shortterm_signal.py — 前向观察期快照工具的守护。

覆盖纯函数三件：note 解析（days= 连写提取/无 days= 出场类 note→0）、
快照增量合并（既有日不重算/新日追加/按 date+variety 去重）、空快照加载。
replay_signals 的窗口口径与引擎一致性由 test_time_stop 段首重置探针 +
绝对基准锚点共同守护，此处不重复引擎级测试。
"""
from __future__ import annotations

# 私有库路径（研究脚本已移入 junior-quant-alpha）
import sys as _sys
from pathlib import Path as _Path
PRIVATE_PROBES = _Path(__file__).resolve().parents[2] / 'private'
for _d in ('probes', 'experiments'):
    _p = PRIVATE_PROBES / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
        break

import unittest

import pandas as pd

from futures_shortterm_signal import (
    _days_from_note, load_snapshot, merge_snapshot,
)


def _row(date, var, direction="flat", days=0):
    return {"date": date, "variety": var, "direction": direction,
            "days": days, "entry_n": 30, "note": "breakout: 空仓观望"}


class DaysFromNoteTest(unittest.TestCase):

    def test_hold_note_with_days(self):
        # 持仓类 note："持多 pos=1days=4"——days= 与 pos= 连写无分隔符
        self.assertEqual(_days_from_note("breakout: 持多 pos=1days=4"), 4)
        self.assertEqual(_days_from_note("breakout: 持空 pos=-1days=10"), 10)
        self.assertEqual(_days_from_note("breakout: 突破30日高 pos=1days=1"), 1)

    def test_exit_notes_have_no_days_eq(self):
        # 出场类 note 不含 "days="（"pos=7days" 无等号）→ 0（已平仓）
        self.assertEqual(_days_from_note("breakout: 出场:跌破10日低 pos=7days"), 0)
        self.assertEqual(_days_from_note("breakout: 时间止损:10日到期 pos=10days"), 0)

    def test_missing_or_malformed(self):
        self.assertEqual(_days_from_note("breakout: 空仓观望"), 0)
        self.assertEqual(_days_from_note("breakout: 数据不足(需31根)"), 0)
        self.assertEqual(_days_from_note("days=abc"), 0)
        self.assertEqual(_days_from_note(""), 0)


class MergeSnapshotTest(unittest.TestCase):

    def test_incremental_append_only(self):
        # 既有日不重算：旧行原样保留（即使新数据不同——增量语义）
        old = pd.DataFrame([_row("2026-01-05", "RB", "long", 3)])
        new = pd.DataFrame([_row("2026-01-05", "RB", "short", 7),   # 同日新算→丢弃
                            _row("2026-01-06", "RB", "flat", 0)])   # 新日→追加
        m = merge_snapshot(old, new)
        self.assertEqual(len(m), 2)
        rb5 = m[m["date"] == "2026-01-05"].iloc[0]
        self.assertEqual((rb5["direction"], rb5["days"]), ("long", 3))

    def test_dedup_by_date_variety(self):
        new = pd.DataFrame([_row("2026-01-06", "RB"), _row("2026-01-06", "RB")])
        m = merge_snapshot(pd.DataFrame(columns=["date", "variety", "direction",
                                                 "days", "entry_n", "note"]), new)
        self.assertEqual(len(m), 1)

    def test_empty_old_passthrough(self):
        new = pd.DataFrame([_row("2026-01-06", "RB", "long", 1)])
        m = merge_snapshot(pd.DataFrame(), new)
        self.assertEqual(len(m), 1)


class LoadSnapshotTest(unittest.TestCase):

    def test_missing_file_empty(self):
        df = load_snapshot("/tmp/_nonexistent_snapshot_xyz.csv")
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns),
                         ["date", "variety", "direction", "days", "entry_n", "note"])


if __name__ == "__main__":
    unittest.main()
