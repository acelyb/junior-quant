"""引擎 carry 注入泄漏回归测试（强化计划第三步证据包挖出的 bug）。

bug（已修）：`_run_date_loop` 原只在 `not carry_map[var].empty` 时注入 carry——
策略实例跨品种共享，无源品种（空 carry）跳过注入后 **保留上一品种注入的序列**，
吃到邻序品种的展期信号；结果依赖 futures.symbols 顺序（实测同集合换序 0.8pp
年化差）。修复 = 无条件注入（空帧注入后 CarryStrategy.on_bar 判 empty → hold，
与"无源 = 腿分 0"语义一致，daily_runner 一直就是这个语义）。

守护两点：
  1. **顺序无关**：同品种集合不同 symbols 顺序 → 逐日 NAV 完全一致；
  2. **无源品种不吃邻序 carry**：A(有源 carry 空/负) + B(无源) 组合里，B 的
     方向 = 纯 TSMOM（0.5×tsmom 取符号），与 B 单独跑一致。

不依赖网络：mock 引擎的三个数据入口（build_contract_series / get_roll_yield_series /
get_futures_rules），策略走真实 TsmomCarryStrategy。
运行：python -m unittest tests.futures.test_carry_leak
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


def _built_frame(idx, base: float) -> pd.DataFrame:
    """真 built 帧形态（adj_close/adj_return/is_roll 齐全，归一化后 contract_source=built）。
    价格单调上涨 → TSMOM(lookback 小) 恒 long。"""
    n = len(idx)
    drift = 0.5 * np.arange(n)
    return pd.DataFrame({
        "date": idx, "close": base + drift, "adj_close": base + drift,
        "adj_return": [0.0] + [0.005] * (n - 1),
        "hold": 1000.0, "is_roll": [False] * n,
    })


def _carry_frame(idx, roll: float) -> pd.DataFrame:
    """有源品种的展期收益率序列（常数 → carry 方向稳定）。"""
    return pd.DataFrame({
        "date": idx, "roll_yield": [roll] * len(idx),
        "near_by": ["n"] * len(idx), "deferred": ["d"] * len(idx),
    })


def _run_engine(symbols: list[str], carry_by_var: dict[str, pd.DataFrame]) -> dict:
    """mock 数据入口跑一次真实引擎（全轴单段、equal 权重、无成本、无风控）。"""
    import jq_framework.backtest.futures_walk_forward as wf
    idx = pd.date_range("2025-01-05", periods=60, freq="B")
    frames = {"A": _built_frame(idx, 100.0), "B": _built_frame(idx, 200.0)}
    empty_cols = pd.DataFrame(columns=["date", "roll_yield", "near_by", "deferred"])

    orig = (wf.get_futures_main_continuous, wf.get_roll_yield_series,
            wf.get_futures_rules, wf.build_contract_series)
    wf.get_futures_main_continuous = lambda v, contract="X0", read_only=False: (
        frames[v]).copy()
    wf.get_roll_yield_series = lambda v, read_only=False: (
        carry_by_var.get(v, empty_cols)).copy()
    wf.get_futures_rules = lambda read_only=False: {}
    wf.build_contract_series = lambda v, read_only=False: (frames[v]).copy()
    try:
        cfg = {"strategies": {"tc": {"module": "jq_alpha.strategies.cta.tsmom_carry",
                                     "class": "TsmomCarryStrategy",
                                     "params": {"weight": 0.5,
                                                "tsmom": {"lookback": 5, "price": "settle"},
                                                "carry": {"smooth": 3}}}},
               "futures": {"symbols": symbols, "rebalance_days": 1,
                           "margin": 0.1, "cost": {}, "risk": {}}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            return wf.run_futures_portfolio(cfg, "tc", contract="built",
                                            weight_method="equal", verbose=False)
    finally:
        (wf.get_futures_main_continuous, wf.get_roll_yield_series,
         wf.get_futures_rules, wf.build_contract_series) = orig


class CarryLeakOrderIndependenceTest(unittest.TestCase):
    """修复回归：同集合不同顺序 → NAV 完全一致（原 bug 下顺序决定无源品种吃谁的 carry）。"""

    def test_order_does_not_change_nav(self):
        # A 有源（carry 负 → 与 TSMOM 多反向 → 合成 0 → hold）；B 无源（应纯 TSMOM → long）
        idx = pd.date_range("2025-01-05", periods=60, freq="B")
        carry = {"A": _carry_frame(idx, -0.05)}
        r1 = _run_engine(["A0", "B0"], carry)
        r2 = _run_engine(["B0", "A0"], carry)
        nav1 = r1["nav_df"]["nav"].to_numpy()
        nav2 = r2["nav_df"]["nav"].to_numpy()
        self.assertEqual(len(nav1), len(nav2))
        np.testing.assert_allclose(nav1, nav2, rtol=1e-12)

    def test_b_alone_matches_mixed_when_no_carry(self):
        # 无源品种 B 在混合组合里的方向 = B 单独跑（有源邻序品种的 carry 不串味）。
        # 比符号而非原始权重：equal 权重下混合=B/budget 均分、单跑=B 独享 budget，
        # 数值天然差一个倍数；warm-up 日双方同为 0。
        idx = pd.date_range("2025-01-05", periods=60, freq="B")
        carry = {"A": _carry_frame(idx, -0.05)}
        mixed = _run_engine(["A0", "B0"], carry)
        alone = _run_engine(["B0"], {})
        sign_mixed = np.sign(np.asarray(mixed["weights_log"]["B"], dtype=float))
        sign_alone = np.sign(np.asarray(alone["weights_log"]["B"], dtype=float))
        np.testing.assert_array_equal(sign_mixed, sign_alone)
        self.assertTrue((sign_mixed > 0).sum() > 30,
                        "混合组合里无源 B 应在 warm-up 后保持 TSMOM 多头")


class CarryLeakSemanticsTest(unittest.TestCase):
    """注入语义：空 carry（含无列形态）注入后 → hold（腿分 0），非残留上一品种。"""

    def test_empty_with_columns_leads_hold_leg(self):
        # A 有源 carry 负（hold），B 无源（列全空帧）→ B 应 long（原 bug 下 B 吃 A 的
        # 负 carry → hold）。终值对照：B-only 组合上涨 → 混合组合 B 部分也在涨。
        idx = pd.date_range("2025-01-05", periods=60, freq="B")
        carry = {"A": _carry_frame(idx, -0.05)}
        r = _run_engine(["A0", "B0"], carry)
        # A 腿 hold → 组合收益全部来自 B（equal 权重各 0.5）→ 终值 > 1（B 单边涨）
        final = float(r["nav_df"]["nav"].iloc[-1])
        self.assertGreater(final, 1.0)

    def test_columnless_empty_frame_handled(self):
        # 异常兜底形态（pd.DataFrame() 无列）原样注入不崩、B 仍纯 TSMOM
        import jq_framework.backtest.futures_walk_forward as wf
        idx = pd.date_range("2025-01-05", periods=60, freq="B")
        carry = {"A": _carry_frame(idx, -0.05), "B": pd.DataFrame()}

        orig = (wf.get_futures_main_continuous, wf.get_roll_yield_series,
                wf.get_futures_rules, wf.build_contract_series)
        frames = {"A": _built_frame(idx, 100.0), "B": _built_frame(idx, 200.0)}
        wf.get_futures_main_continuous = lambda v, contract="X0", read_only=False: (
            frames[v]).copy()
        wf.get_roll_yield_series = lambda v, read_only=False: carry[v].copy()
        wf.get_futures_rules = lambda read_only=False: {}
        wf.build_contract_series = lambda v, read_only=False: (frames[v]).copy()
        try:
            cfg = {"strategies": {"tc": {"module": "jq_alpha.strategies.cta.tsmom_carry",
                                         "class": "TsmomCarryStrategy",
                                         "params": {"weight": 0.5,
                                                    "tsmom": {"lookback": 5, "price": "settle"},
                                                    "carry": {"smooth": 3}}}},
                   "futures": {"symbols": ["A0", "B0"], "rebalance_days": 1,
                               "margin": 0.1, "cost": {}, "risk": {}}}
            buf = io.StringIO()
            with redirect_stdout(buf):
                r = wf.run_futures_portfolio(cfg, "tc", contract="built",
                                             weight_method="equal", verbose=False)
            sign_b = np.sign(np.asarray(r["weights_log"]["B"], dtype=float))
            self.assertTrue((sign_b >= 0).all(), "无源 B 不应出现空头")
            self.assertTrue((sign_b > 0).sum() > 30,
                            "无列空帧注入后 B 仍应纯 TSMOM 多头（warm-up 后）")
        finally:
            (wf.get_futures_main_continuous, wf.get_roll_yield_series,
             wf.get_futures_rules, wf.build_contract_series) = orig


if __name__ == "__main__":
    unittest.main()
