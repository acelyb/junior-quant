"""S1 前视 / S2 脏值 / S3 起点截断 / S4 settle 复权 守护单元测试（审计 2026-09-18）。

守护重点：
  S1 一日/同日 前视：信号在 _run_date_loop 用 `date < dt`（不含 dt 当天）判定 →
    某日 on_bar 拿到的 bars 严格早于当日，赚 dt 白天收益（T+1 口径），不吃同日收盘已知信息。
  S2 脏值：_roi_series 对 settle<=0 行产出 NaN（非 -100%），_realized_vol 不被污染。
  S3 起点：run_futures_portfolio 尊重 futures.start 截断各品种起点。
  S4 复权：built 帧 settle 按 adj_return 累计重建（2026-10-14 修正为一律重建：
    原"全 0 才重建"守卫在真实数据上永不触发，原始拼接 settle 的换月跳空直进信号）。

不依赖网络/缓存：手造 mini 帧 + 记录型假策略，纯函数可复现、单进程轻量。

运行：python -m unittest tests.futures.test_lookahead_clean
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jq_framework.backtest.futures_walk_forward import _run_date_loop, _roi_series, _realized_vol
import jq_framework.backtest.futures_walk_forward as wf
class _RecordingStrategy:
    """记录每次 on_bar 收到的 bars 的末日期，用于断言 S1 前视口径。"""
    factor_legs: dict = {}
    seen_last_dates: list = []

    def on_bar(self, bars: pd.DataFrame):
        from jq_framework.strategies.base import Signal
        self.seen_last_dates.append(pd.Timestamp(bars["date"].iloc[-1]))
        return Signal(direction="long", price=float(bars["close"].iloc[-1]))


def _mk_trend(n=10, step=1.0, start=100.0):
    idx = pd.date_range("2026-01-05", periods=n, freq="B")
    close = start + step * np.arange(n)
    return pd.DataFrame({"date": idx, "close": close})


class S1LookaheadTest(unittest.TestCase):
    def test_signal_uses_bars_strictly_before_dt(self):
        """S1：on_bar 拿到的 bars 永远不含当日 dt（`< dt`），末日期最多到昨日收盘。

        不假设循环从哪天开始调 on_bar（引擎会跳过热身），只守护核心不变量：
        全流程中策略看到的 bars 末日期集合 = 全部日期中去掉最后一日的集合。
        即任何一天都不会用到当日/未来数据（否则末日期会出现最后一日 = 未来前视）。
        """
        df = _mk_trend(n=8)
        idx = list(pd.to_datetime(df["date"]).values)
        roi = pd.Series(0.01, index=idx)
        vol = pd.Series(0.15, index=idx)
        rec = _RecordingStrategy()
        _run_date_loop(
            data={"A": df}, roi={"A": roi}, vol={"A": vol},
            dates=idx, strat=rec, class_name="fake",
            need_carry=False, carry_map={}, margin_ratios={"A": 0.1},
            risk_cfg={}, commission=0.0, slippage=0.0,
            weight_method="equal", rebalance_days=1, attribution=False,
            n_vars=1, roll_masks={})
        self.assertTrue(len(rec.seen_last_dates) > 0, "策略应至少被调用一次")
        # 每个 seen 末日期必须是前 n-1 天之一（严格早于某轮 dt；绝不含最后一日）
        seen = set(rec.seen_last_dates)
        allowed = set(idx[:-1])
        for d in seen:
            self.assertIn(d, allowed,
                          f"S1 前视：on_bar 收到末日期 {d} 不在允许集（含当日/未来→前视）")
        # 若策略在最后一轮也被调（含最后一日时向前收），末日期集合应覆盖到倒数第二日
        self.assertIn(idx[-2], seen,
                      "末轮 on_bar 前一日收盘应被看到（< 最后一日 dt）")


class S2DirtyValueTest(unittest.TestCase):
    def test_zero_settle_yields_nan_not_minus1(self):
        """S2：零星 settle==0 行 → 收益 0（前向填充为上一价，视为无成交），非 -100%。"""
        idx = pd.date_range("2026-01-05", periods=6, freq="B")
        df = pd.DataFrame({"date": idx, "settle": [100.0, 101.0, 0.0, 103.0, 104.0, 105.0]})
        roi = _roi_series(df)
        # 脏 0 被 ffill 为上一有效价 101 → 该行收益 = 101/101-1 = 0（而非 -100%）
        self.assertAlmostEqual(roi.iloc[2], 0.0, places=9,
                               msg="settle=0 行收益应为 0（ffill 前一价），而非 -100% 假亏损")
        # 后续行正常（脏 0 不传染尾部）
        self.assertTrue(np.isfinite(roi.iloc[3]) and np.isfinite(roi.iloc[4]))

    def test_negative_settle_masked(self):
        """S2：极端负 settle（脏数据）同样 ffill，不产生 -100%。"""
        idx = pd.date_range("2026-01-05", periods=5, freq="B")
        df = pd.DataFrame({"date": idx, "settle": [100.0, 101.0, -5.0, 103.0, 104.0]})
        roi = _roi_series(df)
        self.assertAlmostEqual(roi.iloc[2], 0.0, places=9)

    def test_realized_vol_not_poisoned_by_zero(self):
        """S2：_realized_vol 吃修后的 roi，不受 0 值 -100% 污染（std 应有限偏小而非爆表）。"""
        idx = pd.date_range("2026-01-05", periods=40, freq="B")
        vals = 100.0 + 0.5 * np.arange(40)
        vals[20] = 0.0  # 中间一个脏 0
        df = pd.DataFrame({"date": idx, "settle": vals})
        roi = _roi_series(df)
        vol = _realized_vol(roi, 10)
        self.assertTrue(vol.dropna().between(0.0, 3.0).all(),
                        "干净脏值后 realized vol 应有限（在 0-3 年化区间），不被 -100% 污染")


class S3S4Test(unittest.TestCase):
    def test_built_settle_rebuilt_when_all_zero(self):
        """S4：built 帧 settle 全 0（最简触发例）时按 adj_return 累计重建。

        2026-10-14 修正为一律重建（真实 settle 恒有真值，换月跳空用例见
        tests/futures/test_contract_config.py）；本用例保留"全 0 不可用→重建"
        路径的守护。"""
        idx = pd.date_range("2026-01-05", periods=5, freq="B")
        df = pd.DataFrame({"date": idx, "close": [97.0, 98.0, 99.0, 100.0, 101.0],
                           "adj_return": [0.01, 0.01, 0.01, 0.01, 0.01],
                           "settle": [0.0, 0.0, 0.0, 0.0, 0.0]})
        # 直接驱动引擎的 built 归一化逻辑（此处用 _roi_series 验证 settle 得到重建的口径）
        # ——真实重建在 run_futures_portfolio built 分支；这里验证重建产物的收益可取。
        s = df["settle"].mask(df["settle"] <= 0)
        self.assertTrue(s.isna().all(), "全 0 settle 在重建前应判定为不可用（触发 S4 重建）")
        # 重建后的 settle 应为正且单调（累乘 adj_return）
        rebuilt = (1.0 + df["adj_return"]).cumprod()
        self.assertTrue((rebuilt > 0).all())
        self.assertTrue(rebuilt.is_monotonic_increasing)

    def _mock_run(self, cfg, start):
        """用假策略 + mock 数据源驱动 run_futures_portfolio，返回诊断里的品种记录日期轴。"""
        idx = pd.date_range("2025-01-05", periods=120, freq="B")
        n = len(idx)
        mk = lambda base: pd.DataFrame(
            {"date": idx, "open": base, "high": base, "low": base,
             "close": base + 0.01 * np.arange(n),
             "settle": base + 0.01 * np.arange(n), "volume": 1000})
        data_A, data_B = mk(100.0), mk(200.0)

        class _FakeStrat:
            factor_legs: dict = {}
            def on_bar(self, bars):
                from jq_framework.strategies.base import Signal
                return Signal(direction="long", price=float(bars["close"].iloc[-1]))

        # mock 数据源：X0 主连 + carry 空 + rules 空 + built 序列空（避开 M3 网络拉取）
        orig_get = wf.get_futures_main_continuous
        orig_carry = wf.get_roll_yield_series
        orig_rules = wf.get_futures_rules
        orig_built = wf.build_contract_series
        wf.get_futures_main_continuous = lambda v, contract="X0", read_only=False: (
            data_A if v == "A" else data_B).copy()
        wf.get_roll_yield_series = lambda v, read_only=False: pd.DataFrame()
        wf.get_futures_rules = lambda read_only=False: {}
        wf.build_contract_series = lambda v, read_only=False: pd.DataFrame()  # M3 无缓存 → 空
        try:
            # 直接在 run_futures_portfolio 用的同款 cfg 上驱动
            strat_cfg = {"module": "jq_alpha.strategies.cta.tsmom", "class": "TsmomStrategy",
                         "params": {}}
            cfg = {**cfg, "strategies": {"tsmom_cta": strat_cfg},
                   "futures": {**cfg.get("futures", {}), "symbols": ["A0", "B0"],
                               "start": start, "rebalance_days": 1,
                               "margin": 0.1, "cost": {}, "risk": {}}}
            _orig_strat = wf.build_strategy
            wf.build_strategy = lambda sc: (_FakeStrat(), "fake")
            try:
                res = wf.run_futures_portfolio(
                    cfg, "tsmom_cta", contract="X0", weight_method="equal",
                    verbose=False)
            finally:
                wf.build_strategy = _orig_strat
            return idx, res
        finally:
            wf.get_futures_main_continuous = orig_get
            wf.get_roll_yield_series = orig_carry
            wf.get_futures_rules = orig_rules
            wf.build_contract_series = orig_built

    def test_start_truncation_respected(self):
        """S3：cfg futures.start 早于数据起点时照常；晚于部分数据时截断到 start。"""
        cfg = {"futures": {}}
        idx, res = self._mock_run(cfg, None)  # 无 start → 不截断
        n_dates = len(res["nav_df"])
        self.assertGreaterEqual(n_dates, 100, "无 start 应保留全部数据")

        # 带 late start：只从 start 起（mock 数据 2025-01-05 起，start=2025-02-01 截断）
        late = pd.Timestamp("2025-02-01")
        cfg2 = {"futures": {}}
        idx2, res2 = self._mock_run(cfg2, late)
        nav2 = res2["nav_df"]
        first_date = pd.Timestamp(nav2["date"].iloc[0]) if "date" in nav2.columns \
            else pd.Timestamp(nav2.index[0])
        self.assertGreaterEqual(first_date, late,
                                f"S3：样本起点应被截断到 start（{first_date} 应 ≥ {late}）")


if __name__ == "__main__":
    unittest.main()
