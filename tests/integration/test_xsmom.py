"""横截面动量（XSec MOM）策略层单元测试。

守护重点（区别于 TSMOM 的本质：截面相对强弱，不是自身绝对收益符号）：
  - 强品种 → long、弱品种 → short、中间 → hold 的横截面分位判定
  - 多空分组比例由 quant 控制（0.33 = 前/后 1/3）
  - `set_active_var` 逐品种取方向；归因腿与主信号走同一 directions 缓存
  - 品种不足（<4）/ 单品种数据不足 → 退化 hold（不加噪声）
  - settle 优先于 close 的价格口径（与 tsmom 一致）

不依赖网络 / 不碰缓存：本地构造多品种 bars DataFrame，纯函数可复现。

运行：python -m unittest tests.futures.test_xsmom
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jq_alpha.strategies.cta.cross_sectional_momentum import CrossSectionalMomentumStrategy


def make_bars(close: list[float], settle: list[float] | None = None) -> pd.DataFrame:
    """构造最小 bars DataFrame[date, close, settle]。settle 缺省 = close。"""
    n = len(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({"date": dates, "close": [float(c) for c in close]})
    if settle is not None:
        df["settle"] = [float(s) for s in settle]
    return df


def ascending(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """单调上涨序列（用于构造"强"品种）。"""
    return make_bars([start + i * step for i in range(n)])


def descending(n: int, start: float = 200.0, step: float = -1.0) -> pd.DataFrame:
    """单调下跌序列（用于构造"弱"品种）。"""
    return make_bars([start + i * step for i in range(n)])


class XSecMomTest(unittest.TestCase):

    def _ranked_directions(self, bars_map: dict[str, pd.DataFrame],
                           **params) -> dict[str, str]:
        """跑 set_universe 排名 + 逐品种 set_active_var/on_bar，返回 var → direction。"""
        strat = CrossSectionalMomentumStrategy(params=params)
        strat.set_universe(bars_map)
        out = {}
        for var in bars_map:
            strat.set_active_var(var)
            out[var] = strat.on_bar(bars_map[var]).direction
        return out

    # 12 品种 3 簇，收益间隙宽 → 分位边界落在中部大空隙，判定无歧义：
    #   强 ~+0.34..+0.37（唯一 > hi_med → long）
    #   中 ~+0.21..+0.31（居中 → hold）
    #   弱 ~-0.05..-0.10（唯一 < lo_med → short）
    STRONG = {"S%d" % i: (300.0, (5 + i) * 1.5) for i in range(4)}
    WEAK = {"W%d" % i: (9000.0, -(5 + i) * 1.5) for i in range(4)}
    MID = {"M%d" % i: (100.0 + (i - 1.5) * 30.0, 1.0) for i in range(4)}

    def _sep_universe(self) -> dict[str, pd.DataFrame]:
        n = 200
        bars_map = {}
        for v, (start, step) in {**self.STRONG, **self.WEAK, **self.MID}.items():
            bars_map[v] = make_bars([start + step * j for j in range(n)])
        return bars_map

    def test_strong_long_weak_short(self):
        """明显强(涨) → long，明显弱(跌) → short，中间 → hold。"""
        dirs = self._ranked_directions(self._sep_universe(), lookback=60, skip=3, quant=0.33)
        for v in self.STRONG:
            self.assertEqual(dirs[v], "long", f"强品种 {v} 应为 long")
        for v in self.WEAK:
            self.assertEqual(dirs[v], "short", f"弱品种 {v} 应为 short")
        for v in self.MID:
            self.assertEqual(dirs[v], "hold", f"中间品种 {v} 应为 hold")

    def test_few_universe_holds_all(self):
        """品种不足 4 个 → 全部退化 hold（不加截面噪声）。"""
        bars_map = {"A": ascending(300), "B": ascending(300), "C": descending(300)}
        dirs = self._ranked_directions(bars_map)
        self.assertTrue(all(d == "hold" for d in dirs.values()))

    def test_insufficient_data_skipped(self):
        """品种数据不足被跳过；剩余品种据已覆盖的分位判向。"""
        # D/E/F/G 数据充足且强/弱分层明显；H 数据不足(<65)被排除 → 它 hold
        bars_map = {
            "D": ascending(300, start=100.0),
            "E": ascending(300, start=150.0),
            "F": descending(300, start=300.0),
            "G": descending(300, start=350.0),
            "H": make_bars([100.0] * 10),  # 数据不足
        }
        dirs = self._ranked_directions(bars_map, lookback=60, skip=3)
        self.assertEqual(dirs["D"], "long")
        self.assertEqual(dirs["F"], "short")
        self.assertEqual(dirs["H"], "hold", "数据不足的品种应 hold")

    def test_quant_controls_group_ratio(self):
        """quant=0.25 时多空更窄：只看最强/最弱 1/4，中间品种 hold。"""
        # 7 个品种，价格从低到高线性排列 → 收益排序与 index 对应
        n = 200
        bars_map = {}
        for i in range(7):
            # 收益 i/100：i=6 最强(+0.06 区间收益)，i=0 最弱(-0.06)
            c = [100.0 + (i - 3) / 100 * j for j in range(n)]
            bars_map[f"V{i}"] = make_bars(c)
        strat = CrossSectionalMomentumStrategy(params={"lookback": 60, "skip": 3, "quant": 0.25})
        strat.set_universe(bars_map)
        # 7 品种 × 0.25 = 1.75 → 前 1 强 long，后 1 弱 short（其余 hold，因中间 5 个不入组）
        strat.set_active_var("V6"); self.assertEqual(strat.on_bar(bars_map["V6"]).direction, "long", "V6 最强 → long")
        strat.set_active_var("V0"); self.assertEqual(strat.on_bar(bars_map["V0"]).direction, "short", "V0 最弱 → short")
        strat.set_active_var("V3"); self.assertEqual(strat.on_bar(bars_map["V3"]).direction, "hold", "V3 中间 → hold")

    def test_attribution_leg_consistent(self):
        """归因腿（factor_legs 指向自身）与主调用读同一 directions 缓存。"""
        bars_map = {
            "A": ascending(200, start=100.0),
            "B": ascending(200, start=150.0),
            "C": descending(200, start=300.0),
            "D": descending(200, start=350.0),
        }
        strat = CrossSectionalMomentumStrategy(params={"lookback": 60, "skip": 3})
        self.assertEqual(list(strat.factor_legs.keys()), ["xsmom"])
        strat.set_universe(bars_map)
        strat.set_active_var("C")
        leg_sig = strat.factor_legs["xsmom"].on_bar(bars_map["C"])  # 归因腿路径
        main_sig = strat.on_bar(bars_map["C"])                       # 主调用路径
        self.assertEqual(leg_sig.direction, main_sig.direction)
        self.assertEqual(leg_sig.direction, "short")

    def test_settle_preferred(self):
        """bars 带 settle 时用 settle 口径判向（与 tsmom 一致）。"""
        # settle 下跌但 close 上涨 → settle 优先 → short
        n = 200
        close = [100.0 + j for j in range(n)]
        settle = [200.0 - j for j in range(n)]
        bars = make_bars(close, settle=settle)
        bars_map = {"A": bars,
                    "B": ascending(n, start=500.0),
                    "C": ascending(n, start=600.0),
                    "D": descending(n, start=800.0)}
        dirs = self._ranked_directions(bars_map, lookback=60, skip=3)
        self.assertEqual(dirs["A"], "short", "settle 下跌应判空（settle 优先于 close）")


if __name__ == "__main__":
    unittest.main()
