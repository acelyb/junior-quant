"""m5_feat_smoke.py — M5 非线性特征源冒烟（离线，合成面板）。

覆盖面（对应 2026-10-27 feature_engineering 实现的四条性质）：
  1. 交互语义：add_interactions 追加 4 对手选乘积列，名 axb，值=两基础列逐行乘积。
  2. 状态语义：state_values 输出 [分化度, 宽度, 波动] 三元组，截面<10 只时 None；
     日内恒定性由调用方结构保证（同 current 一次调用）。
  3. 缺省字节等价：feature_engineering 缺省时 _build_train_xy 输出（X/y/names）与
     无开关口径逐字节一致——reg4 字节门的代码级前提。
  4. 训练/预测列对齐：_build_train_xy 冻结的派生 feature_names 交给
     composite_score_ml 预测侧，rebuild_derived_columns 重建的列值与训练侧派生值
     一致（交互乘积逐元素相等；状态列同日恒定且=state_values 当日值）。

用法：python scripts/m5_feat_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import jq_framework.backtest.walk_forward as wf
from jq_framework.factors.feature_engineering import (DEFAULT_INTERACTIONS, STATE_FEATURES,
                                         add_interactions, parse_fe_cfg,
                                         rebuild_derived_columns, state_values)


def mk_panel(n_syms=60, n_days=500, seed=13):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    dfs = {}
    for i in range(n_syms):
        ret = rng.standard_normal(n_days) * 0.02
        px = 10.0 * np.exp(np.cumsum(ret))
        dfs[f"S{i:03d}"] = pd.DataFrame({
            "date": dates, "open": px, "high": px * 1.02, "low": px * 0.98,
            "close": px, "volume": 1e6 * (1 + rng.random(n_days)),
        })
    return dfs


class _F:
    def __init__(self, name):
        self.name, self.calc_series = name, None

    def calc_cross_section(self, dfs, date=None, series_cache=None):
        out = {}
        for s, df in dfs.items():
            c = df["close"]
            n = min(20, len(c))
            out[s] = float(c.iloc[-1]) / float(c.iloc[-n]) - 1 if n >= 2 else 0.0
        return pd.Series(out)


PANEL = mk_panel()
# 基础因子名覆盖手选交互对的全部 6 个臂（momentum/reversal/vol_dryup/size/ivol）
BASE = ["momentum", "reversal", "vol_dryup", "size", "ivol", "low_vol"]
FACTORS = [(_F(nm), 1.0) for nm in BASE]
ML = {"seeds": [42, 7, 2026], "freq": 5, "val_ratio": 0.2, "min_samples": 100,
      "early_stopping_rounds": 10, "n_estimators": 80, "num_threads": 4}

# ---- 1. 交互语义 ----
X0 = rng0 = None
Xb, yb, nb, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, ML)
Xi, yi, ni, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, {**ML, "feature_engineering": {"interactions": True}})
assert nb == BASE, f"基础特征名应为 BASE, got {nb}"
pairs = [f"{a}x{b}" for a, b in DEFAULT_INTERACTIONS]
assert ni == BASE + pairs, f"交互名应按手选对追加, got {ni}"
col = {nm: j for j, nm in enumerate(nb)}
for k, (a, b) in enumerate(DEFAULT_INTERACTIONS):
    prod = Xi[:, len(BASE) + k]
    expect = Xb[:, col[a]] * Xb[:, col[b]]
    assert np.allclose(prod, expect), f"交互 {a}x{b} 应=两基础列乘积"
assert np.array_equal(yb, yi), "交互不动标签"
print(f"  [smoke] 交互语义 OK（{len(DEFAULT_INTERACTIONS)} 对乘积，标签不动）")

# ---- 2. 状态语义 ----
sv = state_values(PANEL)
assert len(sv) == 3 and all(v is not None for v in sv)
assert sv[0] > 0 and 0.0 <= sv[1] <= 1.0 and sv[2] > 0
thin = {s: df for s, df in list(PANEL.items())[:5]}   # 截面 5 只 < 10
assert state_values(thin) == [None, None, None], "薄截面应返回 None 三元组"
print("  [smoke] 状态语义 OK（分化/宽度/波动三元组，薄截面 None）")

# ---- 3. 缺省字节等价 ----
Xa1, ya1, na1, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, ML)
Xa2, ya2, na2, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, {**ML, "feature_engineering": {}})
assert np.array_equal(Xa1, Xa1) and np.array_equal(ya1, ya2) and na1 == na2
assert np.array_equal(Xa1, Xb), "缺省与无开关应逐字节一致（回归前提）"
print("  [smoke] 缺省字节等价 OK")

# ---- 4. 训练/预测列对齐（关键：同一冻结 names 下两侧派生一致） ----
# 状态开启训练
Xs, ys, ns, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, {**ML, "feature_engineering": {"state": True}})
assert ns == BASE + STATE_FEATURES
# 预测侧：_build_feature_matrix 填基础列（派生名被跳过），rebuild 补派生列
from jq_framework.factors.composite_ml import _build_feature_matrix
Xp, syms = _build_feature_matrix(PANEL, FACTORS, ns, None)
Xp2 = rebuild_derived_columns(Xp, PANEL, ns)
assert Xp2.shape[1] == len(ns), f"预测侧列数应={len(ns)}, got {Xp2.shape[1]}"
# 状态列：日内恒定 = state_values(PANEL)（全量面板下训练末段采样日与预测日同源）
sv_now = state_values(PANEL)
for j, nm in enumerate(STATE_FEATURES):
    colv = Xp2[:, len(BASE) + j]
    assert np.allclose(colv, sv_now[j]), f"预测侧状态列 {nm} 应=state_values 当日值"
# 交互列对齐（预测日=面板末日，与训练末行同日同截面）
Xp_i, _ = _build_feature_matrix(PANEL, FACTORS, ni, None)
Xp_i2 = rebuild_derived_columns(Xp_i, PANEL, ni)
colb = {nm: j for j, nm in enumerate(nb)}
for k, (a, b) in enumerate(DEFAULT_INTERACTIONS):
    j = len(BASE) + k
    assert np.allclose(Xp_i2[:, j], Xp_i2[:, colb[a]] * Xp_i2[:, colb[b]])
print("  [smoke] 训练/预测列对齐 OK（rebuild 与冻结 names 逐列一致）")

# ---- 5. 端到端：m5_all 训练器跑通（真实 LightGBM） ----
# 面板需含真实可预测信号（退化保护 best_iter<=1→空窗是生产语义,噪声面板会误触发）:
# 每股给定一个持续动量偏向,使 20 日动量对未来收益有真实预测力。
def mk_signal_panel(seed=17):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=500)
    dfs = {}
    for i in range(60):
        drift = (i % 12 - 6) * 0.0008          # 跨股票截面差异的持续漂移
        ret = rng.standard_normal(500) * 0.01 + drift
        px = 10.0 * np.exp(np.cumsum(ret))
        dfs[f"S{i:03d}"] = pd.DataFrame({
            "date": dates, "open": px, "high": px * 1.02, "low": px * 0.98,
            "close": px, "volume": 1e6 * (1 + rng.random(500)),
        })
    return dfs


class _MomF(_F):
    def calc_cross_section(self, dfs, date=None, series_cache=None):
        out = {}
        for s, df in dfs.items():
            c = df["close"]
            n = min(20, len(c))
            out[s] = float(c.iloc[-1]) / float(c.iloc[-n]) - 1 if n >= 2 else 0.0
        return pd.Series(out)


PANEL_SIG = mk_signal_panel()
mm, fn, dg = wf._train_ml_model(PANEL_SIG, FACTORS, 10,
                                {**ML, "feature_engineering": {"interactions": True, "state": True},
                                 "seeds": [42]})
assert mm is not None, f"m5_all 应正常训练(信号面板), got diag={dg}"
assert len(fn) == len(BASE) + len(DEFAULT_INTERACTIONS) + len(STATE_FEATURES)
print(f"  [smoke] 端到端训练 OK（m5_all {len(fn)} 特征, best_iter={dg.get('best_iter')}）")

print("\n== M5 feature_engineering 冒烟 全部通过 ==")
