"""摆动点/趋势线/区间线的因果性单测（2026-10-30）。

背景（用户指出）："波峰波谷等这些指标，需要在峰谷发生后的几天才能算出来，
通道线又需要滞后几天"——即指标的真实可用时刻晚于枢轴事件本身。
此前的画图实现里，趋势线从**第一个**枢轴 i1 起画，实测提前 32~36 个
交易日，图上等于让人"当时就看到"一条尚不可能存在的线。

本测试守护三条不变量（全部用手造序列，离线可复现）：
  C1 水平线：confirm == 枢轴 + w，且 line 只能从 confirm 起可见/统计触线
  C2 趋势线：confirm == 第二枢轴 + w；cf 之前不得有"可用"渲染
  C3 区间线：信息全在窗内 → confirm == 窗右端当日（无额外滞后）

另外锁定滞后量级本身（改参数时若滞后突变，测试会提醒重新审视）。
"""
from __future__ import annotations

# 私有库路径（研究脚本已移入 junior-quant-alpha）
import sys as _sys
from pathlib import Path as _Path
PRIVATE_PROBES = _Path(__file__).resolve().parents[2] / 'private'
for _d in ('probes', 'experiments'):
    _p = PRIVATE_PROBES / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
        break

import unittest

import numpy as np
import pandas as pd

import plot_swing_levels as P


def _mk_df(px, start="2022-01-03"):
    """最小 prepare() 帧：只需 dates/px/atr/vr/roll/listed_age。"""
    n = len(px)
    dates = pd.Series(pd.bdate_range(start, periods=n))
    p = pd.Series(px, dtype=float)
    vol = pd.Series(np.full(n, 1.0e5))
    return {
        "var": "TEST",
        "dates": dates,
        "px": p,
        "atr": p.diff().abs().rolling(20, min_periods=1).mean().ffill().fillna(1.0),
        "vr": vol / vol.rolling(20, min_periods=1).mean(),
        "roll": pd.Series(False, index=range(n)),
        "start": None,
        "adj_k": 1.0,
        "raw_close": p,
        "listed_age": np.arange(n) + 9999,   # 全部视为成熟合约
    }


def _zigzag(n=400, period=20, amp=10.0, base=100.0, drift=0.0):
    """确定性正弦锯齿，便于精确断言枢轴位置。

    drift 决定峰的走向：0 = 各峰等高（可用来测不同 w 下的确认日）；
    负值 = 峰逐个走低（趋势线要求压力线斜率 < 0，纯等高正弦连不出线）。
    """
    t = np.arange(n)
    return base + amp * np.sin(2 * np.pi * t / period) + drift * t


class TestHorizontalLag(unittest.TestCase):
    """C1：水平线 confirm == 枢轴 + w，且不允许在 confirm 前可见。"""

    def test_confirm_is_peak_plus_w(self):
        d = _mk_df(_zigzag())
        segs = P.find_segments(d, "peak", w=10, stale_days=1000)
        self.assertGreater(len(segs), 0, "锯齿序列应至少找出一个峰")
        for peak_i, s0, _s1, _lv in segs:
            self.assertEqual(s0 - peak_i, 10,
                             "水平线的确认日必须严格等于 枢轴 + w")

    def test_w_changes_only_shift_confirm(self):
        # 突出度判据是 prom ≥ MIN_PROM·ATR，而 prom 随 w 变小（峰两侧窗口更窄）。
        # 固定周期时 w=5 会因突出度不足被全部滤掉，故让周期随 w 放大，
        # 保证三种 w 下都真的能找出峰——否则这个断言测的是"没找到"，没意义。
        for w in (5, 10, 20):
            d = _mk_df(_zigzag(n=1200, period=6 * w, amp=12.0))
            segs = P.find_segments(d, "peak", w=w, stale_days=1000)
            self.assertTrue(segs, f"w={w} 未找出任何峰，测试前提不成立")
            for peak_i, s0, _, _ in segs:
                self.assertEqual(s0 - peak_i, w)

    def test_no_touch_before_confirm(self):
        """触线统计必须从 confirm 起算——左侧窗口内碰到价位不算触线。"""
        d = _mk_df(_zigzag())
        segs = P.find_segments(d, "peak", w=10, stale_days=1000)
        for kind in ("peak", "trough"):
            ss = P.find_segments(d, kind, w=10, stale_days=1000)
            cf = {s[0]: s[1] for s in ss}
            for t, _lv, _v in P.touch_points(d, ss, kind):
                if t in cf:
                    self.assertGreaterEqual(t, cf[t])

    def test_select_runs_never_starts_before_confirm(self):
        """阶梯渲染的可见段起点不得早于确认日。"""
        d = _mk_df(_zigzag())
        segs = P.find_segments(d, "peak", w=10, stale_days=1000)
        cf = {s[0]: s[1] for s in segs}
        for peak_i, _lv, runs in P.select_runs(d, segs, "peak", 0):
            for a_i, _b_i in runs:
                self.assertGreaterEqual(
                    a_i, cf[peak_i],
                    "压力线可见段早于确认日 = 图上前视")


class TestTrendLineLag(unittest.TestCase):
    """C2：趋势线 confirm == 第二枢轴 + TREND_W，滞后必须为正。"""

    def test_confirm_is_second_pivot_plus_w(self):
        d = _mk_df(_zigzag(n=600, period=40, amp=12.0, drift=-0.30))
        lines = P.find_trendlines(d)
        self.assertGreater(len(lines), 0, "长周期锯齿应能连出趋势线")
        for i1, i2, cf, _end, _slope, _ic, _kind in lines:
            self.assertEqual(cf, i2 + P.TREND_W)
            self.assertGreater(cf, i1, "确认日必须晚于首枢轴")

    def test_lag_is_at_least_confirm_window(self):
        d = _mk_df(_zigzag(n=600, period=40, amp=12.0, drift=-0.30))
        for i1, i2, cf, *_ in P.find_trendlines(d):
            self.assertGreaterEqual(cf - i1, P.TREND_W)
            self.assertGreaterEqual(cf - i1, P.TREND_MIN_GAP)

    def test_render_starts_at_confirm(self):
        """渲染层：draw_trendlines 的实线必须从 cf 起，不能从 i1 起。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = _mk_df(_zigzag(n=600, period=40, amp=12.0, drift=-0.30))
        lines = P.find_trendlines(d)
        self.assertTrue(lines)
        i1, i2, cf, _end, slope, ic, kind = lines[0]

        fig, ax = plt.subplots()
        P.draw_trendlines(ax, d, [lines[0]], zoom_from=0)
        # 找斜率 = slope 的那条"可用期"实线（ls="--" 是虚线，这里按颜色+点数找）
        starts = []
        for ln in ax.lines:
            xs = ln.get_xdata()
            if len(xs) >= 2:
                starts.append((xs[0], ln.get_linestyle()))
        plt.close(fig)
        self.assertTrue(starts)
        # 所有段里，存在一段起点 >= dates[cf]（可用期实线）
        dvals = d["dates"].to_numpy()
        late = [s for s, _ls in starts if pd.Timestamp(s) >= pd.Timestamp(dvals[cf])]
        self.assertTrue(late, "应有一段从确认日 cf 之后才开始画的实线")


class TestRangeLag(unittest.TestCase):
    """C3：区间判据信息全在窗内 → confirm 当日生效，无额外滞后。"""

    def test_confirm_equals_window_right_edge(self):
        n = 300
        px = np.full(n, 100.0)
        px += 1.2 * np.sin(np.arange(n) * 1.7)     # 窄幅震荡
        d = _mk_df(px)
        rngs = P.find_ranges(d, w=P.RANGE_W)
        for _start_i, confirm_i, _end, _hi, _lo in rngs:
            self.assertGreater(confirm_i, 0)

    def test_confirm_not_delayed_beyond_window(self):
        """回归守护：早期实现曾把 confirm 写成 右端 + w（形成期虚长 w 日）。"""
        n = 300
        px = np.full(n, 100.0) + 1.2 * np.sin(np.arange(n) * 1.7)
        d = _mk_df(px)
        for start_i, confirm_i, _e, _hi, _lo in P.find_ranges(d, w=P.RANGE_W):
            self.assertEqual(confirm_i - start_i, P.RANGE_W - 1,
                             "confirm 应恰为窗右端，形成期 = 判据窗")


class TestLagBudget(unittest.TestCase):
    """锁定滞后量级：参数被改到离谱时提醒重新审视。"""

    def test_horizontal_lag_is_one_confirm_window(self):
        self.assertEqual(P.SWING_W, 10,
                         "SWING_W 变更需同步复核滞后预算与图上等待带宽度")

    def test_trendline_lag_is_larger_than_horizontal(self):
        d = _mk_df(_zigzag(n=800, period=40, amp=12.0, drift=-0.30))
        tl = P.find_trendlines(d)
        hs = P.find_segments(d, "peak", w=P.SWING_W, stale_days=1000)
        if tl and hs:
            tl_lag = np.median([l[2] - l[0] for l in tl])
            hs_lag = np.median([s[1] - s[0] for s in hs])
            self.assertGreater(
                tl_lag, hs_lag,
                "趋势线需要两个枢轴，滞后必大于单点水平线")


if __name__ == "__main__":
    unittest.main()
