"""tests/futures/test_time_stop.py — 中短线 Phase 0c 状态型管道的守护。

两块：
  1. TimeStopWrapper 状态机（脚本化内层）：段首"已在条件内"即入场（不空窗）、
     最长 max_days 个报向日、到期后同向不重入、翻转才重入、持仓中反向即反手、
     内层 hold 维持 / flat 平仓、每品种状态独立、从不出 hold（hold≡空仓的
     语义债在包装器内解决，而非引擎层——零回归路线）。
  2. 引擎段首状态重置：windowed 模式逐段重建策略实例（有状态伪策略每段
     各得一次"首日 long"）；单段模式行为不变（只建一次）。
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import jq_framework.backtest.futures_walk_forward as wf
from jq_framework.strategies.base import Signal
from jq_alpha.strategies.cta.time_stop import TimeStopWrapper, build_inner


def _bars(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2024-01-02", periods=n),
                         "settle": [100.0 + i for i in range(n)],
                         "close": [100.0 + i for i in range(n)]})


class _Scripted:
    """按脚本逐日返回方向的内层（hold/flat/long/short 序列；越界取末值）。"""

    def __init__(self, script, params=None):
        self.script = list(script)
        self.i = 0

    def on_bar(self, bars):
        d = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return Signal(direction=d, price=1.0)


def _wrapper(script, max_days=10) -> TimeStopWrapper:
    """真配置构造（走 build_inner 校验），再换成脚本化内层驱动状态机。"""
    w = TimeStopWrapper(params={
        "max_days": max_days,
        "inner": {"module": "jq_alpha.strategies.cta.tsmom", "class": "TsmomStrategy",
                  "params": {"lookback": 250}},
    })
    w.inner = _Scripted(script)
    return w


def _drive(w: TimeStopWrapper, n: int, var: str = "RB") -> list[str]:
    """逐日报向 n 次，返回方向序列（引擎口径：每品种每日恰好一次 on_bar）。"""
    out = []
    w.set_active_var(var)
    b = _bars()
    for _ in range(n):
        out.append(w.on_bar(b).direction)
    return out


class TimeStopStateMachineTest(unittest.TestCase):
    """包装器状态机的全部边界（对应模块 docstring 口径）。"""

    def test_entry_on_first_signal_no_gap(self):
        # 段首内层已给出方向（"已在条件内"）→ 首个报向日即入场，不空窗
        self.assertEqual(_drive(_wrapper(["long"] * 4), 4),
                         ["long", "long", "long", "long"])

    def test_max_days_then_flat(self):
        # 最长 max_days 个报向日：3 天 long，第 4 天起 flat（时间止损）
        self.assertEqual(_drive(_wrapper(["long"] * 6, max_days=3), 6),
                         ["long", "long", "long", "flat", "flat", "flat"])

    def test_no_reentry_while_inner_continues_same_direction(self):
        # 时间止损离场后，内层延续原方向 → 不重入（等翻转）
        dirs = _drive(_wrapper(["long"] * 8, max_days=2), 8)
        self.assertEqual(dirs, ["long", "long", "flat", "flat", "flat",
                                "flat", "flat", "flat"])

    def test_reentry_only_on_flip(self):
        # 离场后内层翻转 → 立即重入（新时钟）；重入后时间止损照走
        dirs = _drive(_wrapper(["long"] * 5 + ["short"] * 3, max_days=2), 8)
        self.assertEqual(dirs, ["long", "long", "flat", "flat", "flat",
                               "short", "short", "flat"])

    def test_reversal_while_holding_reopens_clock(self):
        # 持仓中内层反向 → 立即反手，时间时钟重开（note 的 days 回到 1）
        w = _wrapper(["long", "long", "short", "short"], max_days=10)
        w.set_active_var("RB")
        b = _bars()
        s = [w.on_bar(b) for _ in range(4)]
        self.assertEqual([x.direction for x in s],
                         ["long", "long", "short", "short"])
        self.assertIn("days=1", s[2].note, "反手日时钟须重开")

    def test_inner_hold_maintains_position_and_clock_ticks(self):
        # 内层无意见（hold）：维持持仓，时间止损照走
        self.assertEqual(_drive(_wrapper(["long", "hold", "hold"], max_days=10), 3),
                         ["long", "long", "long"])
        # max_days=1 时 hold 日即到期 → flat
        self.assertEqual(_drive(_wrapper(["long", "hold"], max_days=1), 2),
                         ["long", "flat"])

    def test_inner_flat_closes_and_clears_flip_memory(self):
        # 内层明确 flat：平仓并清翻转记忆 → 其后同向非零方向也算翻转、可入场
        self.assertEqual(_drive(_wrapper(["long", "flat", "long", "long"], max_days=10), 4),
                         ["long", "flat", "long", "long"])

    def test_never_reports_hold(self):
        # 显式报向路线：任何状态下都不出 hold（hold≡空仓语义债在包装器内消化）
        for script, md in [(["hold"] * 3, 10), (["long", "hold"], 1),
                           (["long", "flat", "hold"], 10)]:
            for d in _drive(_wrapper(script, max_days=md), len(script)):
                self.assertIn(d, ("long", "short", "flat"))

    def test_per_variety_state_independent(self):
        # 每品种独立状态：RB 已持仓 2 天时切到 CU 首日入场，互不污染
        w = _wrapper(["long"], max_days=10)
        w.set_active_var("RB")
        b = _bars()
        w.on_bar(b)                       # RB 入场 days=1
        w.on_bar(b)                       # RB days=2
        w.set_active_var("CU")
        s_cu = w.on_bar(b)                # CU 首日独立入场
        self.assertEqual(s_cu.direction, "long")
        self.assertIn("days=1", s_cu.note)
        w.set_active_var("RB")
        s_rb = w.on_bar(b)                # RB 时钟不被 CU 打断
        self.assertIn("days=3", s_rb.note)

    def test_build_inner_real_class_and_params(self):
        w = TimeStopWrapper(params={
            "max_days": 10,
            "inner": {"module": "jq_alpha.strategies.cta.tsmom", "class": "TsmomStrategy",
                      "params": {"lookback": 10, "price": "settle"}},
        })
        from jq_alpha.strategies.cta.tsmom import TsmomStrategy
        self.assertIsInstance(w.inner, TsmomStrategy)
        self.assertEqual(w.inner.params["lookback"], 10)
        self.assertEqual(w.factor_legs, {}, "有状态：禁止自引用出腿")

    def test_missing_inner_raises(self):
        with self.assertRaises(ValueError):
            TimeStopWrapper(params={"max_days": 10})
        with self.assertRaises(ValueError):
            build_inner({"module": "jq_alpha.strategies.cta.tsmom"})


class _OnceStrat:
    """有状态伪策略：每次构造计数；首个 on_bar 报 long，其余 flat。

    若引擎不逐段重建实例，第二段起将全程 flat——本类即段首状态重置的探针。
    """

    n_built = 0

    def __init__(self, params=None):
        self.params = params or {}
        self._done = False
        _OnceStrat.n_built += 1

    def on_bar(self, bars):
        if self._done:
            return Signal(direction="flat", price=1.0)
        self._done = True
        return Signal(direction="long", price=1.0)


class SegmentResetTest(unittest.TestCase):
    """引擎 0c：windowed 逐段重建策略实例；单段只建一次。"""

    def setUp(self):
        _OnceStrat.n_built = 0
        n = 60
        idx = pd.date_range("2025-01-05", periods=n, freq="B")
        px = 100.0 + 0.01 * np.arange(n)
        self.df = pd.DataFrame({"date": idx, "open": 1.0, "high": 1.0,
                                "low": 1.0, "close": px, "settle": px,
                                "volume": 1000.0})
        self.orig = (wf.get_futures_main_continuous, wf.get_roll_yield_series,
                     wf.get_futures_rules, wf.build_contract_series)
        wf.get_futures_main_continuous = lambda v, contract="X0", read_only=False: self.df.copy()
        wf.get_roll_yield_series = lambda v, read_only=False: pd.DataFrame()
        wf.get_futures_rules = lambda read_only=False: {}
        wf.build_contract_series = lambda v, read_only=False: pd.DataFrame()

    def tearDown(self):
        (wf.get_futures_main_continuous, wf.get_roll_yield_series,
         wf.get_futures_rules, wf.build_contract_series) = self.orig

    def _run(self, **kw):
        # module 用 __name__ 动态引用：tests/futures 无 __init__.py，discover
        # 跑法下本模块以顶层名导入，硬编码 "tests.futures.test_time_stop" 会
        # 让引擎二次导入出另一份模块对象（另一份 _OnceStrat），计数恒 0
        cfg = {"strategies": {"fake_cta": {"module": __name__,
                                           "class": "_OnceStrat", "params": {}}},
               "futures": {"symbols": ["A0"], "rebalance_days": 1,
                           "margin": 0.1, "cost": {}, "risk": {}}}
        return wf.run_futures_portfolio(cfg, "fake_cta", contract="X0",
                                        weight_method="equal", verbose=False, **kw)

    def test_windowed_rebuilds_per_segment(self):
        res = self._run(train_days=20, test_days=10, step_days=10)
        n_seg = len(res["per_window"])
        self.assertGreater(n_seg, 1, "60 天数据 20/10/10 应切出多段")
        # 每段重建一次（段首 + 循环内首建，首建实例被替换不计行为差异）
        self.assertEqual(_OnceStrat.n_built, n_seg + 1)
        # 每段首日报向各得一次 long → 非零仓日数 == 段数（不重建则只有 1）
        long_days = sum(1 for w in res["weights_log"]["A"] if w != 0)
        self.assertEqual(long_days, n_seg)
        # 0d 同号串：伪策略首日 long 后全程 flat → 每段恰好一段长 1 的串
        # （注意：段首并非强制空仓——bars 含 train 全历史、方向即日延续；
        #  这里断串来自伪策略自身的 flat，非引擎行为）
        st = res["stats"]
        self.assertEqual(st["n_runs"], n_seg)
        self.assertAlmostEqual(st["avg_hold_days"], 1.0)

    def test_single_segment_builds_once(self):
        res = self._run()
        self.assertEqual(_OnceStrat.n_built, 1)
        long_days = sum(1 for w in res["weights_log"]["A"] if w != 0)
        self.assertEqual(long_days, 1)


if __name__ == "__main__":
    unittest.main()
