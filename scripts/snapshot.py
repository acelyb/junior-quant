"""
数据快照管理 CLI（验证强化与实验防重复执行计划 P1-4，2026-10-21）。

thin wrapper over data.fetchers.snapshot（算法与冻结语义的唯一权威实现处）。
用法：
  python scripts/snapshot.py freeze [--reason "批前冻结"]     # 冻结当前缓存状态
  python scripts/snapshot.py ls                                # 列出快照与引用状态
  python scripts/snapshot.py gc [--keep-newest 2] [--yes]      # 清理（默认 dry-run）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jq_framework.data.fetchers import snapshot as snap  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="数据快照冻结/列表/清理（P1-4）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser("freeze", help="把当前缓存状态冻结为不可变快照（内容级幂等）")
    p_freeze.add_argument("--reason", default="", help="冻结原因（写进 manifest）")
    p_freeze.add_argument("--cache-dir", default="data/cache",
                          help="被冻结的缓存目录（默认 data/cache）")

    sub.add_parser("ls", help="列出全部快照（时间升序）")

    p_gc = sub.add_parser("gc", help="删除登记簿不再引用且非最新的快照（默认 dry-run）")
    p_gc.add_argument("--keep-newest", type=int, default=2)
    p_gc.add_argument("--yes", action="store_true", help="确认执行删除（默认 dry-run 只列出）")

    args = parser.parse_args()
    if args.cmd == "freeze":
        sid = snap.freeze(args.cache_dir, reason=args.reason)
        print(f"[冻结完成] 快照 id: {sid}")
        print(f"  目录: {snap.snapshots_root() / sid}")
        print("  内容级幂等：状态未变则复用既有快照，不新增代际")
    elif args.cmd == "ls":
        snaps = snap.list_snapshots()
        if not snaps:
            print("（无快照）")
            return
        print(f"{'snapshot_id':<34}{'files':>7}  {'reason':<14}created_at")
        for m in snaps:
            print(f"{m['snapshot_id']:<34}{m.get('n_cache_files', 0):>7}  "
                  f"{m.get('reason', ''):<14}{m.get('created_at', '')}")
    else:
        victims = snap.gc(keep_newest=args.keep_newest, assume_yes=args.yes)
        if args.yes:
            print(f"[清理完成] 删除 {len(victims)} 份快照: {victims}")
        else:
            print("（dry-run：加 --yes 执行删除）")


if __name__ == "__main__":
    main()
