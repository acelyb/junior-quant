"""期货 carry 数据源缺口短路单元测试（2026-09-18 M4 回填引入 #11 的回归守护）。

背景 bug：M4 把 carry 默认起点改读 config futures.start（2018 起）。对 6 个**无 carry 源**
品种（I/JM/J/M/Y/P，东财 get_roll_yield 与新浪逐合约价均无期限结构数据），`_need_refresh`
会因缓存缺失/过期判 True → 触发 2018 起全量逐日重拉；而每日出错 JSONDecodeError 仍走
retry_backoff(is_retryable=True, retries=3, base=2.0) ≈ 10s/天 → 每品种 ~2000 天 ≈ 数小时，
且每次 walk-forward 对每个 carry 策略都会重拉一遍 → 病理级卡死（实测单次健康卡 >54min 未出）。

修复：fetcher 定义 NO_CARRY_SOURCE，get_roll_yield_series 对无源品种**短路返回空**（远端
为空与本地空等价，重拉只是重复试错权威性空数据），不触发任何网络/重试。

守护（纯函数、网络无关、单进程）：
  - 无源品种短路：get_roll_yield_series 应立即返回空的 4 列帧，不拉网不重试。
  - 白名单单源性：backfill_carry 复用的是同一 NO_CARRY_SOURCE（防两份漂移）。

运行：python -m unittest tests.futures.test_carry_source_gap
"""
from __future__ import annotations

import unittest

import pandas as pd

from jq_framework.data.fetchers.fetcher_futures import NO_CARRY_SOURCE, get_roll_yield_series

# 实测（2026-09-18）东财 get_roll_yield 与新浪逐合约价均无数据（JSONDecodeError）
NO_SOURCE = {"I", "JM", "J", "M", "Y", "P"}
_COLS = ["date", "roll_yield", "near_by", "deferred"]


class CarrySourceGapTest(unittest.TestCase):

    def test_no_carry_source_singleton(self):
        """白名单单源：fetcher 与 backfill 脚本共用同一 NO_CARRY_SOURCE。"""
        import scripts.backfill_carry as bc
        # backfill 不得再私自定义一套 → 同对象引用（防两份清单漂移）
        self.assertIs(get_roll_yield_series.__globals__["NO_CARRY_SOURCE"],
                      bc.NO_CARRY_SOURCE)

    def test_no_source_shortcircuits_empty(self):
        """无源品种短路：即使缺失缓存/显式 2018 起，也返回空帧、不拉网不重试。"""
        for v in NO_SOURCE:
            df = get_roll_yield_series(v, start_day="20180101")
            self.assertTrue(df.empty, f"{v} 短路后应返回空帧")
            self.assertEqual(list(df.columns), _COLS)

    def test_sourced_variety_not_shortcircuited(self):
        """有源品种（RB）不得被白名单误伤：NO_CARRY_SOURCE 不含 RB。"""
        self.assertNotIn("RB", NO_CARRY_SOURCE)
        self.assertNotIn("CU", NO_CARRY_SOURCE)


if __name__ == "__main__":
    unittest.main()
