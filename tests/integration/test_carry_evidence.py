"""futures_carry_evidence.py 纯逻辑单元测试（强化计划第三步：carry 裁定证据包）。

守护重点：
  1. 分组划分与生产口径一致：三组恰好瓜分 config 17 品种、无重叠无遗漏
     （组定义错了 → 证据包整张表失真，裁定会被误导）
  2. _group_cfg 深拷贝隔离：改 B 选项权重不能污染原 config（A 选项会跟着变）
  3. _corr 相关计算：同曲线 1.0 / 反向 −1.0 / 日期无交或 <30 共同日 → NaN
  4. _stats_row 键提取

不依赖网络：只打纯函数与 config 文件本身（不跑引擎）。
运行：python -m unittest tests.futures.test_carry_evidence
"""
from __future__ import annotations

import unittest

import pandas as pd
import yaml

from futures_carry_evidence import (
    GROUPS,
    SOURCED,
    _corr,
    _daily_returns,
    _group_cfg,
    _stats_row,
)

# 私有库路径（研究脚本已移入 junior-quant-alpha）
import sys as _sys
from pathlib import Path as _Path
PRIVATE_PROBES = _Path(__file__).resolve().parents[2] / 'private'
for _d in ('probes', 'experiments'):
    _p = PRIVATE_PROBES / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
        break

from jq_framework.config.loader import load_config

_CFG = load_config()


def _nav(dates: list[str], path: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "nav": path})


class GroupPartitionTest(unittest.TestCase):
    """分组划分 = 生产 17 品种的无交并集（与 daily_runner carry 实测口径一致）。"""

    def test_covers_all_config_varieties_exactly_once(self):
        symbols = _CFG["futures"]["symbols"]
        varieties = {s[:-1] if s.endswith("0") else s for s in symbols}
        grouped = [v for vs in GROUPS.values() for v in vs]
        self.assertEqual(len(grouped), len(set(grouped)), "组内有重复品种")
        self.assertEqual(set(grouped), varieties, "分组与 config 17 品种不一致")
        self.assertEqual(len(grouped), 17)

    def test_group_sizes(self):
        self.assertEqual(len(GROUPS["完整(2018起)"]), 4)   # RB/CU/AU/TA
        self.assertEqual(len(GROUPS["短史(2021-09起)"]), 7)  # AL/ZN/NI/AG/MA/CF/SC
        self.assertEqual(len(GROUPS["无源"]), 6)            # I/JM/J/M/Y/P

    def test_sourced_is_full_plus_short_disjoint_from_nosrc(self):
        self.assertEqual(SOURCED, GROUPS["完整(2018起)"] + GROUPS["短史(2021-09起)"])
        self.assertEqual(len(SOURCED), 11)
        self.assertFalse(set(SOURCED) & set(GROUPS["无源"]))


class GroupCfgTest(unittest.TestCase):
    """_group_cfg：品种子集 + 同样本起点 + B 选项权重覆盖，且深拷贝隔离。"""

    def test_symbols_and_start(self):
        c = _group_cfg(_CFG, ["RB", "CU"], "2020-01-16")
        self.assertEqual(c["futures"]["symbols"], ["RB0", "CU0"])
        self.assertEqual(c["futures"]["start"], "2020-01-16")
        # 原 config 未被改动
        self.assertNotEqual(_CFG["futures"]["start"], "2020-01-16")
        self.assertGreater(len(_CFG["futures"]["symbols"]), 2)

    def test_weight_override_isolated(self):
        w0 = _CFG["strategies"]["tsmom_carry_cta"]["params"]["weight"]
        c = _group_cfg(_CFG, ["RB"], "2020-01-16", weight=0.7)
        self.assertEqual(c["strategies"]["tsmom_carry_cta"]["params"]["weight"], 0.7)
        # 原 config 权重不动（A 选项不能被 B 的覆盖污染）
        self.assertEqual(_CFG["strategies"]["tsmom_carry_cta"]["params"]["weight"], w0)

    def test_deep_copy_no_shared_mutable(self):
        c = _group_cfg(_CFG, ["RB"], "2020-01-16")
        c["futures"]["symbols"].append("XX0")
        c["strategies"]["tsmom_carry_cta"]["params"]["weight"] = 0.9
        self.assertNotIn("XX0", _CFG["futures"]["symbols"])
        self.assertEqual(_CFG["strategies"]["tsmom_carry_cta"]["params"]["weight"], 0.5)


class CorrTest(unittest.TestCase):
    """_corr：拼接 NAV 日收益的相关（裁定表的"与 tsmom 相关"列）。"""

    def _dates(self, n: int) -> pd.DatetimeIndex:
        return pd.date_range("2024-01-01", periods=n, freq="D")

    def test_identical_curves_corr_one(self):
        d = self._dates(60)
        path = list(1.0 + 0.001 * pd.Series(range(60)))
        a = _nav(list(d), path)
        b = _nav(list(d), path)
        self.assertAlmostEqual(_corr(a, b), 1.0)

    def test_anticorrelated(self):
        # 真负相关 = 日收益逐日取反（NAV 反向线性不成立：其日收益同向单调）
        d = list(self._dates(60))
        r = pd.Series([0.01, -0.02, 0.03, -0.01] * 15)[:59]
        a = _nav(d, [1.0] + list((1.0 + r).cumprod()))
        b = _nav(d, [1.0] + list((1.0 - r).cumprod()))
        self.assertAlmostEqual(_corr(a, b), -1.0)

    def test_disjoint_dates_nan(self):
        d1 = pd.date_range("2024-01-01", periods=60)
        d2 = pd.date_range("2025-01-01", periods=60)
        path = list(1.0 + 0.001 * pd.Series(range(60)))
        self.assertTrue(pd.isna(_corr(_nav(list(d1), path), _nav(list(d2), path))))

    def test_short_overlap_nan(self):
        # 共同日 < 30 → 样本太薄不算相关
        d1 = pd.date_range("2024-01-01", periods=60)
        d2 = pd.date_range("2024-02-01", periods=60)  # 仅 29 天交叠
        path = list(1.0 + 0.001 * pd.Series(range(60)))
        self.assertTrue(pd.isna(_corr(_nav(list(d1), path), _nav(list(d2), path))))

    def test_empty_inputs_nan(self):
        self.assertTrue(pd.isna(_corr(None, None)))
        empty = pd.DataFrame({"date": [], "nav": []})
        self.assertTrue(pd.isna(_corr(empty, empty)))


class StatsRowTest(unittest.TestCase):
    def test_pulls_engine_keys(self):
        res = {"stats": {"ann_return": 0.12, "sharpe": 0.8,
                         "max_drawdown": -0.2, "n_days": 1000}}
        r = _stats_row(res)
        self.assertEqual(r["ann_return"], 0.12)
        self.assertEqual(r["sharpe"], 0.8)
        self.assertEqual(r["max_drawdown"], -0.2)
        self.assertEqual(r["n_days"], 1000)

    def test_missing_keys_default_zero(self):
        r = _stats_row({"stats": {}})
        self.assertEqual(r["ann_return"], 0.0)
        self.assertEqual(r["n_days"], 0)


class DailyReturnsTest(unittest.TestCase):
    def test_pct_change_on_date_index(self):
        a = _nav(["2024-01-01", "2024-01-02", "2024-01-03"],
                 [1.0, 1.1, 0.99])
        r = _daily_returns(a)
        self.assertEqual(len(r), 2)
        self.assertAlmostEqual(r.iloc[0], 0.1)
        self.assertAlmostEqual(r.iloc[1], -0.1)


if __name__ == "__main__":
    unittest.main()
