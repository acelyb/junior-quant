"""板块集中度复评单元测试（强化计划第六步）。

守护重点：
  1. 回撤段检测：峰→谷→复峰语义、min_depth 过滤、尾部未恢复、嵌套取整段、深度降序
  2. 段内尾部指标：同日历窗各 NAV 自计量（段内 MDD/最差5日/恢复天数/未恢复）
  3. 集中度统计：逐日份额 → max_share/bind_days/effective_bets 汇总
  4. 软形态：组合信号 _concentration_lines（份额行/有效下注行/>25% 告警/空仓空列表）
  5. 份额口径修复回归：cap 在 budget=1.5 下封住份额（缺陷场景）——见
     test_capped_weight.py CappedShareSemanticsTest

不依赖网络 / 不碰缓存：本地构造数据，纯函数可复现。

运行：python -m unittest tests.futures.test_sector_evidence
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

# 私有库路径（研究脚本已移入 junior-quant-alpha）
import sys as _sys
from pathlib import Path as _Path
PRIVATE_PROBES = _Path(__file__).resolve().parents[2] / 'private'
for _d in ('probes', 'experiments'):
    _p = PRIVATE_PROBES / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
        break

from jq_framework.backtest.futures_walk_forward import effective_bets, sector_shares
from scripts.futures_combination_signal import _concentration_lines
from futures_sector_evidence import (
    concentration_stats, drawdown_episodes, episode_metrics,
)


def _nav(vals: list[float]) -> pd.Series:
    return pd.Series(vals, dtype=float)


class DrawdownEpisodesTest(unittest.TestCase):

    def test_monotonic_up_no_episodes(self):
        self.assertEqual(drawdown_episodes(_nav([1.0, 1.1, 1.2, 1.3])), [])

    def test_single_episode_with_depth(self):
        eps = drawdown_episodes(_nav([1.0, 0.9, 0.95, 1.0]))
        self.assertEqual(len(eps), 1)
        self.assertEqual((eps[0]["peak_i"], eps[0]["trough_i"], eps[0]["recovery_i"]),
                         (0, 1, 3))
        self.assertAlmostEqual(eps[0]["depth"], 0.10)

    def test_min_depth_filters(self):
        # 回撤 4% < 5% 阈值 → 无段
        self.assertEqual(drawdown_episodes(_nav([1.0, 0.96, 1.0]), min_depth=0.05), [])
        self.assertEqual(len(drawdown_episodes(_nav([1.0, 0.96, 1.0]), min_depth=0.03)), 1)

    def test_unrecovered_tail(self):
        eps = drawdown_episodes(_nav([1.0, 0.9, 0.8, 0.85]))
        self.assertEqual(len(eps), 1)
        self.assertIsNone(eps[0]["recovery_i"])
        self.assertEqual(eps[0]["trough_i"], 2)
        self.assertAlmostEqual(eps[0]["depth"], 0.20)

    def test_nested_deeper_trough_wins(self):
        # 段内更深的谷（0.8）取代首个谷（0.9）——整段一次
        eps = drawdown_episodes(_nav([1.0, 0.9, 0.95, 0.8, 0.9, 1.0]))
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["trough_i"], 3)
        self.assertAlmostEqual(eps[0]["depth"], 0.20)

    def test_two_episodes_sorted_by_depth(self):
        eps = drawdown_episodes(_nav([1.0, 0.95, 1.0, 0.9, 0.85, 0.9, 1.0]))
        self.assertEqual(len(eps), 2)
        self.assertGreater(eps[0]["depth"], eps[1]["depth"])   # 深度降序
        self.assertEqual(eps[0]["peak_i"], 2)                  # 第二段更深排前

    def test_recovery_requires_regaining_peak(self):
        # 0.99 < 1.0 未复峰：段延续到 idx3 才恢复
        eps = drawdown_episodes(_nav([1.0, 0.9, 0.99, 1.0]))
        self.assertEqual(eps[0]["recovery_i"], 3)

    def test_short_series(self):
        self.assertEqual(drawdown_episodes(_nav([1.0])), [])
        self.assertEqual(drawdown_episodes(pd.Series([], dtype=float)), [])


class EpisodeMetricsTest(unittest.TestCase):

    def test_basic_segment_metrics(self):
        nav = _nav([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.95, 1.0, 1.05])
        m = episode_metrics(nav, peak_i=0, end_i=8)
        self.assertAlmostEqual(m["mdd"], -0.1)                    # 段内自窗内高点
        self.assertAlmostEqual(m["worst_5d"], 0.9 / 1.0 - 1.0)    # 5 日窗含崩盘日
        self.assertEqual(m["trough_i"], 6)
        self.assertEqual(m["recovery_days"], 2)                   # 谷(6)→回起点(8)
        self.assertTrue(m["recovered"])

    def test_unrecovered_counts_to_end(self):
        nav = _nav([1.0, 0.9, 0.85, 0.88, 0.87])
        m = episode_metrics(nav, peak_i=0, end_i=4)
        self.assertAlmostEqual(m["mdd"], -0.15)
        self.assertFalse(m["recovered"])
        self.assertEqual(m["recovery_days"], 2)                   # 段末(4)−谷(2)

    def test_other_nav_window_semantics(self):
        # 同一日历窗下另一 NAV 自计量：窗内自创新高再回落——段内 MDD 相对其
        # 自身窗内高点；谷值 1.05 从未跌破窗起点 1.0 → 立即恢复（0 天）
        nav = _nav([1.0, 1.1, 1.2, 1.05, 1.15, 1.3])
        m = episode_metrics(nav, peak_i=0, end_i=4)
        self.assertAlmostEqual(m["mdd"], 1.05 / 1.2 - 1.0)        # 自窗内高点 1.2
        self.assertEqual(m["trough_i"], 3)
        self.assertTrue(m["recovered"])
        self.assertEqual(m["recovery_days"], 0)                   # 谷即已 ≥ 窗起点

    def test_worst5d_uses_pre_window_values(self):
        # t-5 落在窗前：用真实序列值（非 NaN、不截断）
        nav = _nav([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])
        m = episode_metrics(nav, peak_i=5, end_i=7)
        self.assertAlmostEqual(m["worst_5d"], 0.5 / 1.0 - 1.0)    # idx7/idx2


class ConcentrationStatsTest(unittest.TestCase):

    def test_basic_stats(self):
        # 3 日权重日志：黑4/贵2/能化3 品种均匀 |w| → 份额 黑4/9、贵2/9、能化3/9
        wl = {v: [1.0, 1.0, 1.0] for v in ("RB", "I", "JM", "J", "AU", "AG", "TA", "MA", "SC")}
        s = concentration_stats(wl, cap=0.25)
        self.assertEqual(s["n_active_days"], 3)
        self.assertAlmostEqual(s["max_share_mean"], 4.0 / 9.0)
        self.assertAlmostEqual(s["bind_days"], 1.0)               # 黑 44% > 25% 每日绑定
        # 有效下注 = 1/HHI = 1/(0.25²·... )：3 类份额 (4/9, 3/9, 2/9)
        hhi = (16.0 + 9.0 + 4.0) / 81.0
        self.assertAlmostEqual(s["eff_bets_mean"], 1.0 / hhi)

    def test_flat_day_skipped(self):
        wl = {"RB": [0.0, 1.0], "AU": [0.0, 1.0]}
        s = concentration_stats(wl)
        self.assertEqual(s["n_active_days"], 1)                   # 全平日跳过
        self.assertAlmostEqual(s["max_share_mean"], 0.5)

    def test_empty_log(self):
        s = concentration_stats({})
        self.assertEqual(s["n_active_days"], 0)
        self.assertTrue(np.isnan(s["max_share_mean"]))


class SectorHelpersTest(unittest.TestCase):

    def test_sector_shares_aggregates_abs(self):
        sh = sector_shares({"RB": 0.3, "I": -0.2, "AU": 0.1, "CU": 0.4})
        self.assertAlmostEqual(sh["黑色"], 0.5 / 1.0)
        self.assertAlmostEqual(sh["贵金属"], 0.1)
        self.assertAlmostEqual(sh["有色"], 0.4)

    def test_sector_shares_flat_empty(self):
        self.assertEqual(sector_shares({"RB": 0.0, "AU": 0.0}), {})

    def test_effective_bets_extremes(self):
        self.assertAlmostEqual(effective_bets({"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2}), 5.0)
        self.assertAlmostEqual(effective_bets({"a": 1.0}), 1.0)
        self.assertEqual(effective_bets({}), 0.0)


class ConcentrationLinesTest(unittest.TestCase):
    """软形态（组合信号出口）：份额行 + 有效下注行 + >25% 告警。"""

    def test_lines_with_warning(self):
        w = {v: 1.0 for v in ("RB", "I", "JM", "J", "AU", "AG", "TA", "MA", "SC")}
        lines = _concentration_lines(w)
        self.assertEqual(len(lines), 3)                            # 份额 + 下注 + 告警
        self.assertIn("板块集中度", lines[0])
        self.assertIn("黑色 44%", lines[0])                        # 4/9 降序居首
        self.assertIn("有效独立下注", lines[1])
        self.assertIn("> 25%", lines[2])

    def test_lines_without_warning(self):
        # 五类各 2 品均匀 → 份额均 20% ≤ 25% → 两行、无告警
        w = {v: 1.0 for v in ("RB", "I", "CU", "AL", "AU", "AG", "TA", "MA", "M", "Y")}
        lines = _concentration_lines(w)
        self.assertEqual(len(lines), 2)
        self.assertNotIn("⚠️", "".join(lines))

    def test_flat_returns_empty(self):
        self.assertEqual(_concentration_lines({"RB": 0.0}), [])
        self.assertEqual(_concentration_lines({}), [])


if __name__ == "__main__":
    unittest.main()
