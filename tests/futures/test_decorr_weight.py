"""相关感知权重 decorr_vol_parity 单元测试。

守护重点（组合风险预算 = 摊薄高相关簇、给低相关品种更多预算）：
  1. 高相关两腿(RB/I)在 decorr 下相对 vol_parity 被摊薄，低相关腿(CU)预算抬升
  2. 共线矩阵（ridge 病态）回退 vol_parity 不抛
  3. 样本不足（<10 日）回退 vol_parity
  4. 全平（raw_dirs 全 0）→ 空 dict

不依赖网络：手造高/低相关收益序列触发白化，纯函数可复现。

运行：python -m unittest tests.futures.test_decorr_weight
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jq_framework.backtest.futures_walk_forward import _decorr_vol_parity_weights, _make_weights


def _mk_rets(n: int = 200, seed: int = 0, corr_rb_i: float = 0.99) -> dict[str, pd.Series]:
    """手造收益：RB 与 I 高相关，CU 低相关（3 品种都显著波动）。"""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n)
    rb = common + rng.normal(0, 0.004, n)      # 波动 ~1.08%
    i = common * np.sign(corr_rb_i) * 1.0 + rng.normal(0, 0.004, n)
    # 确保 RB/I 高相关：由 common 主导
    rb = common * 0.9 + rng.normal(0, 0.002, n)
    i = common * 0.9 + rng.normal(0, 0.002, n)  # 与 rb 几乎同源 → 高相关
    cu = rng.normal(0, 0.012, n)                # 独立低相关、波动略高
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return {
        "RB": pd.Series(rb, index=idx),
        "I": pd.Series(i, index=idx),
        "CU": pd.Series(cu, index=idx),
    }


class DecorrVolParityTest(unittest.TestCase):

    def setUp(self):
        self.rets = _mk_rets()
        # 三方都做多（信号同向，便于对比权重分配）
        self.raw_dirs = {"RB": 1.0, "I": 1.0, "CU": 1.0}
        self.budget = 1.0
        self.vols = {k: float(np.std(v)) * np.sqrt(252) for k, v in self.rets.items()}

    def test_high_corr_cluster_thinned(self):
        """高相关簇 RB/I 在 decorr 下相对 vol_parity 摊薄，低相关 CU 预算抬升。"""
        w_base = _make_weights(self.raw_dirs, "vol_parity", self.vols, self.budget, 0.05)
        w_decorr = _decorr_vol_parity_weights(
            self.raw_dirs, self.vols, self.budget, self.rets, window=60)
        # CU 是低相关腿：decorr 后权重抬升
        self.assertGreater(w_decorr["CU"], w_base["CU"],
                           "低相关腿 CU 应获得更多预算")
        # 高相关簇合计占比下降
        cluster_base = w_base["RB"] + w_base["I"]
        cluster_decorr = w_decorr["RB"] + w_decorr["I"]
        self.assertLess(cluster_decorr, cluster_base,
                        "高相关簇 RBI 合计权重应摊薄")
        # 预算守恒：总|w|=budget
        self.assertAlmostEqual(sum(abs(v) for v in w_decorr.values()), self.budget, places=5)

    def test_collinear_ridge_falls_back(self):
        """共线矩阵（两腿完全同源）ridge 病态 → 回退 vol_parity 不抛。"""
        n = 200
        rng = np.random.default_rng(7)
        s = rng.normal(0, 0.01, n)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        rets = {
            "RB": pd.Series(s + rng.normal(0, 1e-9, n), index=idx),
            "I": pd.Series(s + rng.normal(0, 1e-9, n), index=idx),  # 几乎完全共线
            "CU": pd.Series(rng.normal(0, 0.012, n), index=idx),
        }
        vols = {k: 0.2 for k in rets}
        w = _decorr_vol_parity_weights(self.raw_dirs, vols, self.budget, rets, window=60)
        self.assertTrue(np.isfinite(sum(abs(v) for v in w.values())))
        self.assertAlmostEqual(sum(abs(v) for v in w.values()), self.budget, places=5)

    def test_insufficient_sample_falls_back(self):
        """样本不足（<10 日）→ 回退 vol_parity，与 _make_weights 一致。"""
        short = {k: v.tail(5) for k, v in self.rets.items()}
        w_decorr = _decorr_vol_parity_weights(
            self.raw_dirs, self.vols, self.budget, short, window=60)
        w_base = _make_weights(self.raw_dirs, "vol_parity", self.vols, self.budget, 0.05)
        for k in self.raw_dirs:
            self.assertAlmostEqual(w_decorr[k], w_base[k], places=6)

    def test_all_flat_returns_empty(self):
        """全平（raw_dirs 全 0）→ 空 dict。"""
        w = _decorr_vol_parity_weights(
            {"RB": 0.0, "I": 0.0, "CU": 0.0}, self.vols, self.budget, self.rets)
        self.assertEqual(w, {})


if __name__ == "__main__":
    unittest.main()
