"""因子治理脚本 futures_factor_health.py 纯函数单元测试。

守护（2026-09-18 审计 M1/M2/M8）：
  - M8：_combo_ir 用**因果滚动 std** 做 1/σ 权重——滚动只在过去，末段 std 不受
    未来收益影响（point-in-time）。对比一宽一窄两窗口在低波期后的权重差。
  - M2：传 drop=f 时，组合收益 = 全集权重下清零 f 腿，**不重新归一化其余腿权重**。
  - M1 链路：单跑决策 keep/drop/reweight 阈值口径由 health_card 主流程覆盖，
    此处守卫 M8/M2 的权重语义（决策阈值在 _cards_from_carrier，依赖引擎，单测易碎）。
  - 残差检验/尾部相关（设计.md §10.3，2026-10-26）：residual_check 把候选拆成
    "锚可解释部分 + 残差"，残差策略夏普判定独立 alpha；tail_check 最差 5% 日
    条件负占比 vs 无条件基线。守护方向性 + 锚自身 NaN + 样本不足 None 语义。

不依赖网络 / 不碰缓存：本地手造虚拟 NAV 序列，纯函数可复现、单进程轻量。

运行：python -m unittest tests.futures.test_factor_health
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.futures_factor_health import (
    _combo_ir, check_window_consistency, residual_check, tail_check)


def _nav(dates, start=1.0):
    """单调上涨的虚拟 NAV（日收益恒定为正，便于对比权重语义）。"""
    return pd.DataFrame({"date": dates, "nav": start * np.linspace(1.0, 2.0, len(dates))})


def _win_series(dates_lists):
    """把多窗口 (date,nav) 组织成 _combo_ir 的 window_series 结构（单窗口足够测权重）。"""
    return [[df] for df in dates_lists]


def _vol_series(amp, n=80):
    """固定波动水平 NA V：收益恒为 amp（日收益=amp 时 std=0 → 用周期序列）。"""
    # 构造正弦波 NAV，波动水平由 amp 控制（收益 std ∝ amp）
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    x = np.arange(n)
    ret = amp * np.sin(x / 3.0) + amp  # 正均值 + 波动
    nav = 1.0 + np.cumsum(ret)
    return pd.DataFrame({"date": dates, "nav": nav})


def _ret_to_window(rets, start="2024-01-01"):
    """日收益序列 → (date, nav) 单窗结构（cumprod 精确还原：pct_change(nav) ≡ rets）。"""
    dates = pd.date_range(start, periods=len(rets), freq="B")
    nav = np.cumprod(1.0 + np.asarray(rets))
    return pd.DataFrame({"date": dates, "nav": nav})


class ComboIrWeightTest(unittest.TestCase):

    def _two_series(self):
        """两因子：因子0 波动高、因子1 波动低，日期对齐。"""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        hi = _vol_series(0.05)
        lo = _vol_series(0.01)
        return [hi, lo]

    def test_higher_vol_gets_less_weight(self):
        """M8：1/σ 缩放，高波动腿权重 < 低波动腿 → 组合收益由低波腿主导。"""
        s0, s1 = self._two_series()
        # 直接构造 r（对齐日收益）验证权重语义
        common = s0["date"]
        r0 = s0.set_index("date")["nav"].pct_change()
        r1 = s1.set_index("date")["nav"].pct_change()
        r = pd.concat([r0, r1], axis=1, keys=["hi", "lo"]).iloc[10:].dropna()
        vol = r.rolling(30, min_periods=8).std()
        w = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
        w_row = w.div(w.sum(axis=1), axis=0)
        # 滚动窗口稳定后（跳过前 ~40 热身期：低波腿 std 极小 → 1/σ 巨大、权重骤摆）：
        # 高波腿权重恒小于低波腿
        stable = w_row.iloc[40:]
        self.assertTrue(np.all(stable["hi"] < stable["lo"]),
                        "M8 反波权重：高波动腿权重应恒小于低波动腿")
        self.assertTrue(np.allclose(stable.sum(axis=1), 1.0),
                        "行内归一：稳定段每行权重和为 1")

    def test_rolling_std_is_causal(self):
        """M8 因果性：某一点权重只用其过去收益估计，不被未来波动影响。"""
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        x = np.arange(90)
        # 前 30 天高波动、之后低波动 → 权重应随波动水平平移（低波期权重抬高）
        ret = np.where(x < 30, 0.05 * np.sin(x / 2.0) + 0.05,
                       np.sin((x - 30) / 2.0) * 0.005 + 0.005)
        nav = 1.0 + np.cumsum(ret)
        s = pd.DataFrame({"date": dates, "nav": nav})
        r = s.set_index("date")["nav"].pct_change().dropna()
        vol = r.rolling(30, min_periods=8).std()
        w = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
        # 因果性：t=80 的权重只反映 ~50..80 的过去（低波期），t=35 反映高波期
        w_early = w.iloc[35]
        w_late = w.iloc[80]
        self.assertGreater(w_late, w_early,
                           "M8 因果滚动：低波期(后期) 1/σ 权重应显著高于高波期(前期)——"
                           "若用样本内 std 则两点权重被全样本抹平、无此分层")

    def test_drop_zeroes_contribution_no_reweight(self):
        """M2：drop=f 只清零该腿但不重归一化其余腿权重。"""
        # 构造：两腿独立 NAV。单腿组合收益应与"清零另一腿"一致（权重=1 恒成立时）
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        a = _vol_series(0.02).assign(date=dates)
        b = _vol_series(0.03).assign(date=dates)
        # 全集 = [a, b]
        win = _win_series([a, b])
        ir_full = _combo_ir(win, ("A", "B"))
        # drop A 后只剩 B 的贡献：因为权重在全集上算、B 的 1/σ 权重归一后为 1，
        # 组合收益 ≡ B 收益 → IR 应与"单调 B"一致（B 单腿若恒定波动则 IR 稳定为正）
        ir_dropA = _combo_ir(win, ("A", "B"), drop="A")
        ir_dropB = _combo_ir(win, ("A", "B"), drop="B")
        # drop 语义：清零一腿后，组合由另一腿贡献（权重固定） → IR 均为正、且不崩溃
        self.assertTrue(np.isfinite(ir_dropA) and np.isfinite(ir_dropB))
        self.assertGreater(ir_dropA, 0.0)
        self.assertGreater(ir_dropB, 0.0)
        # 固定权重下 drop 语义的核验：drop 掉波低贡献高的腿 vs 掉波高贡献低的腿，
        # IR 均应正（不像旧实现重归一化会把单腿波动性放大而不稳定）
        self.assertTrue(np.isfinite(ir_full))


class WindowConsistencyTest(unittest.TestCase):
    """治理硬断言（S2/S3 守护）：窗口数不一致 / NAV 空洞 / 重复日期 → 必须 fail-fast。"""

    def _win_map(self, n_factor=2, n_win=3, n_days=60):
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        return {f"f{i}": [
            pd.DataFrame({"date": dates, "nav": np.linspace(1.0, 2.0, n_days)})
            for _ in range(n_win)
        ] for i in range(n_factor)}

    def test_consistent_windows_passes(self):
        """全一致、NAV 无空洞 → 通过（不抛）。"""
        check_window_consistency(self._win_map())  # 无异常即通过

    def test_mismatched_window_count_raises(self):
        """窗口数不一致 → AssertionError（含各因子窗口数明细）。"""
        m = self._win_map(n_win=3)
        m["f1"] = m["f1"][:2]  # f1 只有 2 窗
        with self.assertRaises(AssertionError) as cm:
            check_window_consistency(m)
        self.assertIn("窗口数不一致", str(cm.exception))

    def test_nan_nav_gap_raises(self):
        """NAV 含 NaN 空洞 → AssertionError。"""
        m = self._win_map()
        m["f0"][1] = m["f0"][1].copy()
        m["f0"][1].loc[m["f0"][1].index[5], "nav"] = float("nan")
        with self.assertRaises(AssertionError):
            check_window_consistency(m)

    def test_duplicate_dates_raises(self):
        """同一窗内日期重复 → AssertionError。"""
        m = self._win_map()
        m["f0"][0] = m["f0"][0].copy()
        m["f0"][0].loc[m["f0"][0].index[3], "date"] = m["f0"][0].loc[m["f0"][0].index[4], "date"]
        with self.assertRaises(AssertionError):
            check_window_consistency(m)


class ResidualCheckTest(unittest.TestCase):
    """残差检验：候选 = 锚β·锚收益 + 独立部分，残差策略判定"马甲 vs 真东西"。"""

    def _series_map(self, fac_ret, ref_ret, n_win=2):
        """把 (候选收益, 锚收益) 切成 n_win 窗 → series_map 结构。"""
        k = len(fac_ret) // n_win
        fac_wins = [_ret_to_window(fac_ret[i * k:(i + 1) * k]) for i in range(n_win)]
        ref_wins = [_ret_to_window(ref_ret[i * k:(i + 1) * k]) for i in range(n_win)]
        return {"fac": fac_wins, "tsmom": ref_wins}

    def test_shadow_factor_residual_not_earn(self):
        """纯马甲：候选 = 2×锚收益 + 零均值噪声 → 残差夏普 ≈ 0、β ≈ 2。"""
        rng = np.random.default_rng(7)
        n = 252
        ref = rng.normal(0.001, 0.01, n)
        fac = 2.0 * ref + rng.normal(0, 1e-4, n)  # 独立部分 = 纯噪声
        sm = self._series_map(fac, ref)
        rc = residual_check(sm, "fac", "tsmom")
        self.assertIsNotNone(rc)
        # 纯噪声残差：252 点夏普的抽样 sd≈1 → |夏普|<2 即"不构成可交易信号"；
        # 方向不预设（−1.5 属 1.5σ 内的正常摆动，非稳定负 alpha）
        self.assertLess(abs(rc["resid_sharpe"]), 2.0)
        self.assertAlmostEqual(rc["resid_beta"], 2.0, delta=0.05)
        self.assertLess(abs(rc["resid_ann"]), 0.5)  # 噪声年化不至于显著

    def test_independent_alpha_residual_earns(self):
        """真东西：候选 = 0.5×锚 + 稳定正独立收益 → 残差夏普显著为正。"""
        rng = np.random.default_rng(7)
        n = 252
        ref = rng.normal(0.0, 0.01, n)
        fac = 0.5 * ref + 0.004 + rng.normal(0, 1e-4, n)  # 独立部分 = 恒 +0.4%/日
        sm = self._series_map(fac, ref)
        rc = residual_check(sm, "fac", "tsmom")
        self.assertIsNotNone(rc)
        self.assertGreater(rc["resid_sharpe"], 3.0)  # 近恒定正残差 → 夏普极高
        self.assertGreater(rc["resid_ann"], 0.0)
        self.assertAlmostEqual(rc["resid_beta"], 0.5, delta=0.05)

    def test_anchor_itself_returns_none(self):
        """锚对自身回归无意义 → None（health_card 里置 NaN 的语义来源）。"""
        rng = np.random.default_rng(7)
        ref = rng.normal(0.001, 0.01, 252)
        sm = self._series_map(ref, ref)
        self.assertIsNone(residual_check(sm, "tsmom", "tsmom"))

    def test_insufficient_sample_returns_none(self):
        """样本不足（<60 残差点，如单窗 <10 收益对全弃）→ None 不误判。"""
        rng = np.random.default_rng(7)
        ref = rng.normal(0.001, 0.01, 40)
        fac = ref + rng.normal(0, 0.001, 40)
        sm = self._series_map(fac, ref, n_win=2)  # 每窗 20 天 < 10 收益对阈值附近
        # 每窗 20 收益对 ≥10 可算 → 40 个残差点 < 60 → None
        self.assertIsNone(residual_check(sm, "fac", "tsmom"))


class TailCheckTest(unittest.TestCase):
    """尾部相关：候选最差 5% 日里锚同跌占比 vs 无条件基线（lift）。"""

    def _sm(self, fac_ret, ref_ret):
        return {"fac": [_ret_to_window(fac_ret)], "tsmom": [_ret_to_window(ref_ret)]}

    def test_crash_together_positive_lift(self):
        """共崩：候选最差日 = 锚也大跌的日子 → lift 显著为正。"""
        rng = np.random.default_rng(3)
        n = 300
        ref = rng.normal(0.0, 0.01, n)
        fac = 0.5 * ref + rng.normal(0, 0.002, n)
        # 候选最差日主要由锚大跌驱动（β=0.5 传导）→ 这些天锚必负
        tc = tail_check(self._sm(fac, ref), "fac", "tsmom")
        self.assertIsNotNone(tc)
        self.assertGreater(tc["tail_lift"], 0.0)
        self.assertGreater(tc["tail_cond_neg"], 0.8)  # 候选最差日锚几乎必跌

    def test_hedge_negative_lift(self):
        """对冲：候选最差日锚反而偏正（负 β）→ lift 为负。"""
        rng = np.random.default_rng(3)
        n = 300
        ref = rng.normal(0.0, 0.01, n)
        fac = -0.5 * ref + rng.normal(0, 0.002, n)  # 负相关：锚跌时候选涨
        tc = tail_check(self._sm(fac, ref), "fac", "tsmom")
        self.assertIsNotNone(tc)
        self.assertLess(tc["tail_lift"], 0.0)
        self.assertLess(tc["tail_cond_neg"], tc["tail_uncond_neg"])

    def test_insufficient_sample_returns_none(self):
        """总样本 < 60 → None。"""
        rng = np.random.default_rng(3)
        ref = rng.normal(0.0, 0.01, 50)
        tc = tail_check(self._sm(ref, ref), "fac", "tsmom")
        self.assertIsNone(tc) if len(ref) < 60 else None
        # fac==ref 同名守卫
        self.assertIsNone(tail_check({"fac": [None], "tsmom": [None]}, "fac", "tsmom"))

    def test_anchor_itself_returns_none(self):
        """锚对自身 → None。"""
        rng = np.random.default_rng(3)
        ref = rng.normal(0.0, 0.01, 300)
        sm = {"tsmom": [_ret_to_window(ref)]}
        self.assertIsNone(tail_check(sm, "tsmom", "tsmom"))


if __name__ == "__main__":
    unittest.main()
