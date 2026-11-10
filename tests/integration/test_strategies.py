"""期货时序策略层单元测试：TSMOM / Carry / Reversal / TsmomCarry / TripleFactor。

守护重点（这些是最易错、最影响期货信号正确性的桩子）：
  - 各策略 on_bar 的方向判定确定性（long/short/hold）
  - 数据不足 / 无效价格边界 → 正确 hold
  - settle 优先于 close 的价格口径
  - 容器策略（tsmom_carry / triple_factor）的加权合成方向
  - triple_factor 软权重 w(t) 门控（强趋势/低波动 → 反转腿归零）

不依赖网络 / 不碰缓存：全部用本地构造的 bars DataFrame，纯函数可复现。

运行：python -m unittest tests.futures.test_strategies
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jq_alpha.strategies.cta.carry import CarryStrategy
from jq_alpha.strategies.cta.reversal import ReversalStrategy
from jq_alpha.strategies.cta.triple_factor import TripleFactorStrategy
from jq_alpha.strategies.cta.tsmom import TsmomStrategy
from jq_alpha.strategies.cta.tsmom_carry import TsmomCarryStrategy


def make_bars(close: list[float], settle: list[float] | None = None) -> pd.DataFrame:
    """构造最小 bars DataFrame[date, close, settle]。settle 缺省 = close。"""
    n = len(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({"date": dates, "close": [float(c) for c in close]})
    if settle is not None:
        df["settle"] = [float(s) for s in settle]
    return df


class TsmomTest(unittest.TestCase):
    def test_long_when_price_up(self):
        # 单调上涨 → lookback 累计收益 > 0 → long
        bars = make_bars([100 + i for i in range(30)], settle=[100 + i for i in range(30)])
        sig = TsmomStrategy(params={"lookback": 5}).on_bar(bars)
        self.assertEqual(sig.direction, "long")
        self.assertEqual(sig.target_pos, 1.0)

    def test_short_when_price_down(self):
        bars = make_bars([200 - 2 * i for i in range(30)], settle=[200 - 2 * i for i in range(30)])
        sig = TsmomStrategy(params={"lookback": 5}).on_bar(bars)
        self.assertEqual(sig.direction, "short")

    def test_hold_when_insufficient_data(self):
        # lookback=250 但仅 10 根 → 数据不足 hold
        bars = make_bars([100 + i for i in range(10)])
        sig = TsmomStrategy(params={"lookback": 250}).on_bar(bars)
        self.assertEqual(sig.direction, "hold")
        self.assertIn("数据不足", sig.note)

    def test_hold_when_no_price(self):
        # 无 close 无 settle → hold
        bars = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
        sig = TsmomStrategy(params={"lookback": 5}).on_bar(bars)
        self.assertEqual(sig.direction, "hold")

    def test_target_pos_clamped(self):
        bars = make_bars(list(range(100, 130)))
        sig = TsmomStrategy(params={"lookback": 5, "target_pos": 2.5}).on_bar(bars)
        self.assertLessEqual(sig.target_pos, 1.0)
        sig0 = TsmomStrategy(params={"lookback": 5, "target_pos": -1.0}).on_bar(bars)
        self.assertGreaterEqual(sig0.target_pos, 0.0)


class CarryTest(unittest.TestCase):
    def _carry_df(self, yields: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(yields), freq="B"),
            "roll_yield": yields,
        })

    def test_long_when_backwardation(self):
        # 升水(正展期收益) → long
        c = CarryStrategy(params={"smooth": 3})
        c.inject_carry(self._carry_df([0.01, 0.02, 0.03]))
        sig = c.on_bar(make_bars([100] * 10))
        self.assertEqual(sig.direction, "long")

    def test_short_when_contango(self):
        c = CarryStrategy(params={"smooth": 3})
        c.inject_carry(self._carry_df([-0.01, -0.02, -0.03]))
        sig = c.on_bar(make_bars([100] * 10))
        self.assertEqual(sig.direction, "short")

    def test_hold_without_carry(self):
        # 未注入 carry → hold（不主动拉网，框架无关可测）
        sig = CarryStrategy().on_bar(make_bars([100] * 10))
        self.assertEqual(sig.direction, "hold")

    def test_hold_when_carry_zero(self):
        c = CarryStrategy(params={"smooth": 3})
        c.inject_carry(self._carry_df([0.0, 0.0, 0.0]))
        sig = c.on_bar(make_bars([100] * 10))
        self.assertEqual(sig.direction, "hold")

    def test_carry_ignores_future_beyond_bars(self):
        # carry 只取 bars 覆盖范围内最新可得值；bars 范围外的未来 carry 不得影响方向
        # bars 覆盖 5 个交易日：valid=前 5 根全 +0.03 → long。
        # 若误用了 bars 外的未来 carry（-0.05），smooth=3 的均值会转负 → short（即泄漏）。
        c = CarryStrategy(params={"smooth": 3})
        c.inject_carry(self._carry_df([0.03, 0.03, 0.03, 0.03, 0.03,
                                       -0.05, -0.05, -0.05]))  # 未来全负，不得泄漏
        sig = c.on_bar(make_bars([100] * 5))  # 只覆盖前 5 个交易日
        self.assertEqual(sig.direction, "long")


class ReversalTest(unittest.TestCase):
    def test_short_when_price_up(self):
        # 涨过头 → 高抛(空)：反转 = -sign(累计收益)
        bars = make_bars([100 + 3 * i for i in range(10)])
        sig = ReversalStrategy(params={"lookback": 5}).on_bar(bars)
        self.assertEqual(sig.direction, "short")

    def test_long_when_price_down(self):
        bars = make_bars([200 - 3 * i for i in range(10)])
        sig = ReversalStrategy(params={"lookback": 5}).on_bar(bars)
        self.assertEqual(sig.direction, "long")

    def test_hold_when_insufficient(self):
        bars = make_bars([100] * 3)
        sig = ReversalStrategy(params={"lookback": 5}).on_bar(bars)
        self.assertEqual(sig.direction, "hold")


class TsmomCarryTest(unittest.TestCase):
    def _strat(self, **kw):
        # 显示传入 tsmom lookback，避免默认 250 需超长历史（容器测试只造短 bars）
        params = {"weight": 0.5, "tsmom": {"lookback": 5},
                  "carry": {"smooth": 3}, **kw}
        return TsmomCarryStrategy(params=params)

    def test_long_when_both_agree_long(self):
        # TSMOM 多 + carry 多 → 加权和 > 0 → long
        c = self._strat()
        c.inject_carry(pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=8, freq="B"),
            "roll_yield": [0.02] * 8,
        }))
        sig = c.on_bar(make_bars(list(range(100, 108))))  # 上涨 → TSMOM long
        self.assertEqual(sig.direction, "long")

    def test_hold_when_both_flat(self):
        # 无 carry + TSMOM 数据不足(lookback>bars) → 合成 0 → hold
        bars = make_bars([100] * 5)
        c = self._strat(tsmom={"lookback": 250})  # 数据不足
        sig = c.on_bar(bars)
        self.assertEqual(sig.direction, "hold")

    def test_dirties_tsmom_wins_with_majority_weight(self):
        # TSMOM(多) + carry(无方向) = 0.5*+1 → 多
        c = self._strat()
        c.inject_carry(pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=8, freq="B"),
            "roll_yield": [0.0] * 8,  # carry 无方向
        }))
        sig = c.on_bar(make_bars(list(range(100, 108))))  # 上涨
        self.assertEqual(sig.direction, "long")

    def test_no_carry_injected_equals_plain_tsmom(self):
        # 第三步证据包机制断言：无源品种（carry 未注入 → 腿分=0）时，
        # combined = w×tmom_score，sign 与纯 TSMOM 相同（w∈(0,1]）→ 逐日等价，
        # carry 的真实作用（否决/确认）只发生在有源品种。多/空两向 + A(0.5)/B(0.7) 权重都验。
        tsmom = TsmomStrategy(params={"lookback": 5})
        for weight in (0.5, 0.7):
            c = self._strat(weight=weight)  # 不 inject_carry = 无源品种
            up = make_bars(list(range(100, 108)))    # 上涨 → tsmom long
            down = make_bars(list(range(108, 100, -1)))  # 下跌 → tsmom short
            self.assertEqual(c.on_bar(up).direction, tsmom.on_bar(up).direction)
            self.assertEqual(c.on_bar(down).direction, tsmom.on_bar(down).direction)
            self.assertEqual(c.on_bar(down).direction, "short")


class TripleFactorTest(unittest.TestCase):
    def _base_bars(self, close: list[float]) -> pd.DataFrame:
        return make_bars(close, settle=close)

    def _tf(self, **params):
        # 传小子参 lookback 让短 bars 够用；默认 250 会数据不足
        p = {"tsmom": {"lookback": 5}, "reversal": {"lookback": 3},
             "carry": {"smooth": 3}, **params}
        return TripleFactorStrategy(params=p)

    def test_strong_trend_reversal_weight_zero(self):
        # 强趋势（偏离均线远）→ t→1 → w_rev≈0（§8.6 门控铁律）
        tf = self._tf()
        # 单调强上涨 120 根，trend_lookback=60 内价格远高于均线 → t 接近 1
        bars = self._base_bars([100 * (1.002 ** i) for i in range(120)])
        tf.on_bar(bars)
        self.assertLess(tf.regime_weights["w_rev"], 0.05)
        # w_tm / w_cy 应近似对半分余量
        self.assertAlmostEqual(tf.regime_weights["w_tm"] + tf.regime_weights["w_cy"]
                               + tf.regime_weights["w_rev"], 1.0, places=6)

    def test_flat_market_reversal_regime_activated(self):
        # 高波动 + 震荡（价格紧贴均线、无趋势）→ w_rev 抬升
        tf = self._tf()
        # 小幅锯齿围绕均线摆动（振幅 4% < trend_th 8% → 低趋势 t，高波动 v）→ 激活反转腿
        close = [100 * (1.0 + 0.04 * (-1.0 if (i // 3) % 2 else 1.0))
                 for i in range(300)]
        tf.on_bar(self._base_bars(close))
        self.assertGreater(tf.regime_weights["w_rev"], 0.1)
        self.assertLess(tf.regime_weights["w_tm"], 0.7)  # 震荡 → TSMOM 权重不应独大

    def test_weights_sum_to_one(self):
        tf = self._tf()
        for t in (0.0, 0.5, 1.0):
            for v in (0.0, 0.5, 1.0):
                w = tf._soft_weights(t, v)
                self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
                self.assertGreaterEqual(w["w_tm"], 0.0)
                self.assertGreaterEqual(w["w_cy"], 0.0)
                self.assertGreaterEqual(w["w_rev"], 0.0)

    def test_strong_trend_direction_tsmom(self):
        # 强趋势下 TripleFactor 方向 = TSMOM 方向（反转腿归零后趋势主导）
        tf = self._tf()
        bars = self._base_bars([100 * (1.003 ** i) for i in range(120)])
        sig = tf.on_bar(bars)
        self.assertEqual(sig.direction, "long")


if __name__ == "__main__":
    unittest.main()
