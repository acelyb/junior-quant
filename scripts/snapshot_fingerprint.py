"""
数据快照指纹 CLI（原 scripts/world_fingerprint.py 的替代，P0-3 算法不变，2026-10-21）。

算法与全部历史登记行逐字节兼容（见 data/fetchers/snapshot.py）——本脚本只是 thin CLI：
指纹计算/登记的唯一权威实现处在 data.fetchers.snapshot，供回测钩子与 CLI 共用同一实现，
杜绝两处算法漂移。

字段名已从 world_fingerprint 改为 snapshot_fingerprint（登记簿列名同步迁移，值不变）。

用法：
  python scripts/snapshot_fingerprint.py --universe csi1000            # 打印 JSON
  python scripts/snapshot_fingerprint.py --universe csi1000 --record  # 追加到指纹日志
  python scripts/snapshot_fingerprint.py --universe csi1000 --snapshot <id>  # 对快照重算（=冻结时刻指纹）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jq_framework.data.fetchers import snapshot as snap  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="数据快照指纹（P0-3 算法，值兼容历史登记行）")
    parser.add_argument("--universe", default="csi1000")
    parser.add_argument("--record", action="store_true",
                        help=f"追加到 {snap.FINGERPRINT_LOG_CSV.relative_to(ROOT)}")
    parser.add_argument("--snapshot", default=None,
                        help="对指定快照（id/前缀）重算指纹而非活动缓存——硬链接保留 mtime_ns，"
                             "重算值 = 冻结时刻指纹，可用于核对历史登记行")
    args = parser.parse_args()

    if args.snapshot:
        snap_dir = snap.resolve_snapshot(args.snapshot)
        fp = snap.fingerprint(args.universe, snap_dir)
        fp["source"] = f"snapshot:{snap_dir.name}"
    else:
        fp = snap.fingerprint(args.universe)
    print(json.dumps(fp, ensure_ascii=False, indent=2))
    if args.record:
        snap.record(fp)
        print(f"[已登记] {snap.FINGERPRINT_LOG_CSV}")


if __name__ == "__main__":
    main()
