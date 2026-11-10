"""换月伪 PnL 剔除单元测试（①a：引擎 _run_date_loop 的 roll_masks 剔除路径）。

守护重点（换月伪 PnL 是 built 口径相对 X0 的工程尾部——is_roll 列在
_build_contract_series_allc 已产出、引擎归一化后从未实现剔除，本轮落地）：
  1. 换月日（roll_masks[var][dt]=True）：品那种归因贡献 port_ret 归零，**不动信号方向**
     （信号 on_bar 走邻接连续帧天然无伪跳空，只剔除组合/归因收益层）。
  2. 非换月日正常累计：同行未被误伤。
  3. X0 路径（roll_masks=None/空）逐字不变：换月剔除是纯增量旁路，默认零影响。
  4. 基准 bench 与组合同口径剔除换月伪收益。

不依赖网络/缓存：手造两品种 mini 帧 + 固定方向假策略直接驱动 _run_date_loop，
换月日收益故意造大跳空（若未剔除会污染 port_ret）。

运行：python -m unittest tests.futures.test_rollover_pnl
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jq_framework.backtest.futures_walk_forward import _run_date_loop


class _FixedStrategy:
    """固定方向假策略：on_bar 恒返回 long，factor_legs 空（不开归因）。"""
    factor_legs: dict = {}

    def __init__(self, direction: str = "long"):
        self._dir = direction

    def on_bar(self, bars: pd.DataFrame):
        from jq_framework.strategies.base import Signal
        return Signal(direction=self._dir, price=float(bars["close"].iloc[-1]))


def _mk_dates(n: int = 8, roll_at: int | None = 3):
    """n 个连续交易日；第 roll_at 天（除 0 外）定为换月日。"""
    idx = pd.date_range("2026-01-05", periods=n, freq="B")
    is_roll = pd.Series(False, index=idx)
    if roll_at is not None:
        is_roll.iloc[roll_at] = True
    return idx, is_roll


def _mk_var(name: str, idx, ret: float, is_roll: pd.Series):
    """单品种：data 帧（on_bar 用 close 升序列）+ roi 收益序列 + roll_mask。"""
    n = len(idx)
    closes = 100.0 * (1 + ret) ** np.arange(n)
    data = pd.DataFrame({"date": idx, "close": closes})
    roi = pd.Series(ret, index=idx)
    roll_mask = is_roll.rename(name)
    return data, roi, roll_mask


class RolloverPnlZeroingTest(unittest.TestCase):
    def setUp(self):
        self.idx, _roll_shared = _mk_dates(n=8, roll_at=3)
        # A 不换月（mask 全 False）；B 在第 4 天(idx[3]) 换月
        idx_a, roll_a = _mk_dates(n=8, roll_at=None)
        self.da, self.ra, _ = _mk_var("A", idx_a, 0.01, roll_a)
        self.db, self.rb_raw, self.rb_mask = _mk_var("B", self.idx, 0.01, _roll_shared)
        self.db = self.db.copy()
        self.rb = self.rb_raw.copy()
        roll_dt = self.idx[3]
        self.rb.loc[roll_dt] = -0.50   # 换月伪跳空（raw close 之比而非 adj）
        self.roll_masks = {"A": roll_a.rename("A"), "B": self.rb_mask}
        self.risk_cfg = {}
        self.margin = {"A": 0.1, "B": 0.1}
        self.vol = {"A": pd.Series(0.15, index=self.idx),
                    "B": pd.Series(0.15, index=self.idx)}

    def _run(self, roll_masks):
        return _run_date_loop(
            data={"A": self.da, "B": self.db},
            roi={"A": self.ra, "B": self.rb},
            vol=self.vol,
            dates=list(self.idx), strat=_FixedStrategy("long"),
            class_name="fake", need_carry=False, carry_map={},
            margin_ratios=self.margin, risk_cfg=self.risk_cfg,
            commission=0.0, slippage=0.0, weight_method="equal",
            rebalance_days=1, attribution=False, n_vars=2,
            roll_masks=roll_masks)

    def test_roll_day_contrib_zeroed_with_mask(self):
        """给 roll_masks：换月日 B 的归因贡献 0（而非 -0.25=w(0.5)*(-0.5)）。"""
        res = self._run(self.roll_masks)
        roll_dt = self.idx[3]
        attr = dict(zip([r["date"] for r in res["nav_series"]],
                        res["attribution"]["B"]))
        self.assertAlmostEqual(attr[roll_dt], 0.0, places=6,
                               msg="换月日伪收益应被 mask 归零")

    def test_non_roll_day_kept(self):
        """非换月日 B 收益正常累计（未被误伤）。"""
        res = self._run(self.roll_masks)
        attr = dict(zip([r["date"] for r in res["nav_series"]],
                        res["attribution"]["B"]))
        # 第 4 天（idx[4]，非换月）B 应贡献 w_B * r_B = (1/2)*0.01
        self.assertAlmostEqual(attr[self.idx[4]], 0.5 * 0.01, places=6)

    def test_port_ret_roll_day_excludes_B(self):
        """换月日组合收益 = 仅 A 贡献（w_A*r_A），不含 B 的伪跳空。"""
        res = self._run(self.roll_masks)
        port = dict(zip([r["date"] for r in res["nav_series"]],
                        res["port_ret_series"]))
        # equal 两腿：w_A=w_B=0.5；换月日 B 归零 → port_ret=0.5*0.01=0.005
        self.assertAlmostEqual(port[self.idx[3]], 0.5 * 0.01, places=6)

    def test_x0_path_no_mask_unaffected(self):
        """roll_masks=None（X0 路径恒空）：换月日 B 保留 -0.50 伪收益。
        证明剔除是纯增量旁路，默认（X0）逐字不变。"""
        res = self._run(None)
        attr = dict(zip([r["date"] for r in res["nav_series"]],
                        res["attribution"]["B"]))
        # 未剔除 → B 换月日= w_B*(-0.50) = -0.25
        self.assertAlmostEqual(attr[self.idx[3]], -0.25, places=6)

    def test_bench_synced_with_combo(self):
        """基准 bench 与组合同口径：换月日 bench 也剔除 B 伪收益。"""
        with_mask = self._run(self.roll_masks)
        navs = {r["date"]: r["nav"] for r in with_mask["bench_series"]}
        dates = sorted(navs)

        def daily(d):
            i = dates.index(d)
            return navs[dates[i]] / navs[dates[i - 1]] - 1.0 if i > 0 \
                else navs[dates[i]] - 1.0

        # 换月日 bench_ret = 仅 A 贡献 0.5*0.01（B 伪跳空剔除）
        self.assertAlmostEqual(daily(self.idx[3]), 0.5 * 0.01, places=6,
                               msg="基准换月日 B 伪收益应剔除")


if __name__ == "__main__":
    unittest.main()
