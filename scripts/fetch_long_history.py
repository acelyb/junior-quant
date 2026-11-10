"""
拉取长历史日线数据集（2010+），隔离到 data/cache_long，不污染 data/cache（回归门保护对象）。

通过环境变量 QUANT_CACHE_DIR=data/cache_long 让 fetcher._load_config 把 cache_dir 指向独立目录
（见 data/fetcher.py:26-40）。缓存键含 start（stock_{sym}_{adj}_{start}），故 2010 文件与
2020 文件名本就不冲突，目录隔离进一步保证 data/cache 的回归基线纯净。

日线端点稳定（akshare stock_zh_a_daily，连续拉 csi1000 全池不封，区别于会小时级长封的分钟端点，
见 memory sina-minute-endpoint-rate-limit）。遵循 fetch_workers=8 限流纪律。

用法：
  python scripts/fetch_long_history.py                    # 默认 csi1000，2010-01-01 起
  python scripts/fetch_long_history.py --universe csi500
  python scripts/fetch_long_history.py --start 2005-01-01  # 更早（更多样本但幸存者偏差加重）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="csi1000",
                        help="指数成分股池：csi300/csi500/csi1000（默认 csi1000）")
    parser.add_argument("--start", default="2010-01-01",
                        help="拉取起始日期（默认 2010-01-04 可拉，更早样本更多但幸存者偏差加重）")
    parser.add_argument("--cache-dir", default="data/cache_long",
                        help="隔离缓存目录（默认 data/cache_long）")
    args = parser.parse_args()

    # 环境变量隔离：必须在 import fetcher 之前设置（_load_config 每次调用读 env）
    os.environ["QUANT_CACHE_DIR"] = args.cache_dir

    from jq_framework.data.fetchers.fetcher import _load_config, get_index_constituents, get_batch_daily

    print(f"[长历史] cache_dir = {_load_config().get('cache_dir')}")
    print(f"[长历史] 拉取 {args.universe} 成分股，起始 {args.start} ...")
    syms = get_index_constituents(args.universe)
    print(f"[长历史] {len(syms)} 只股票")
    get_batch_daily(syms, start=args.start, adjust="qfq", progress=True)
    print(f"[长历史] 完成，数据落 {args.cache_dir}/（data/cache 未动）")


if __name__ == "__main__":
    main()
