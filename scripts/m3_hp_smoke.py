"""m3_hp_smoke.py — M3 超参抖动集成冒烟（离线，合成面板）。

覆盖面（对应 2026-10-26 hp_ensemble 实现的四条性质）：
  1. hp 路径端到端：enabled 时走 _train_ml_model_hp，返回 _SeedEnsemble，
     成员 = 超参组×种子（2×3=6），diag 带 hp_groups/best_iter_by_member。
  2. 确定性：同输入两次训练，预测比特一致。
  3. 缺省不进分支：hp_ensemble 缺省时行为与现行多种子路径逐字一致
     （成员 = 种子×3，diag 无 hp_groups 键）——字节回归的代码级前提。
  4. 退化语义：val_set 早停退化（best_iter ≤ min_best_iter）的成员被剔除、
     不连坐其余成员；全退化→空仓。

用法：python scripts/m3_hp_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import jq_framework.backtest.walk_forward as wf
def mk_panel(n_syms=40, n_days=400, seed=7):
    """合成价格面板：momentum 型信号 + 噪声。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    dfs = {}
    for i in range(n_syms):
        ret = 0.002 * rng.standard_normal(n_days).cumsum() * 0  # 平价底
        mom = np.sin(np.arange(n_days) / 40.0 + i) * 0.5        # 可学结构
        px = 10.0 * np.exp(np.cumsum(0.001 * mom + 0.01 * rng.standard_normal(n_days)))
        dfs[f"S{i:03d}"] = pd.DataFrame({
            "date": dates, "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "volume": 1e6 * (1 + rng.random(n_days)),
        })          # date 保持列（_build_train_xy 按 df["date"] 取）
    return dfs


class _F:
    def __init__(self, name, fn):
        self.name, self._fn = name, fn
    def calc_cross_section(self, dfs, date=None, series_cache=None):
        # 生产签名 (dfs, date) 兼容训练侧 (current, series_cache=…) 两态调用；
        # 返回 pd.Series(index=symbol)（FactorBase 契约）；
        # dfs 的值为截至当日的前缀切片（DatetimeIndex），末行即当日
        out = {}
        for s, df in dfs.items():
            if isinstance(df.index, pd.DatetimeIndex):
                out[s] = float(df["close"].iloc[-1])
            else:
                out[s] = float(self._fn(df))
        return pd.Series(out)
    calc_series = None          # _build_series_caches 探测属性，None 即走窗口内预算路径


PANEL = mk_panel()
_mom = _F("mom", lambda df: df["close"].iloc[-1] / df["close"].iloc[-20] - 1)
_rev = _F("rev", lambda df: -(df["close"].iloc[-1] / df["close"].iloc[-5] - 1))
FACTORS = [(_mom, 1.0), (_rev, 1.0)]          # _build_train_xy 期望 (factor, weight) 元组
ML = {"seeds": [42, 7, 2026], "freq": 5, "val_ratio": 0.2, "min_samples": 100,
      "early_stopping_rounds": 10, "n_estimators": 120, "num_threads": 4}

# ---- 1. hp 路径端到端 ----
hp_cfg = {"hp_ensemble": {"enabled": True}}
m, names, diag = wf._train_ml_model(PANEL, FACTORS, 10, {**ML, **hp_cfg})
assert m is not None, "hp 路径应训出模型"
assert diag["n_members"] == 6, f"应 2 组×3 种子=6 成员，实际 {diag['n_members']}"
assert "hp_groups" in diag and len(diag["hp_groups"]) == 2
assert set(diag["members"][0]) <= set("gs0123456789_"), "成员名应为 g{组}_s{种子}"
assert diag["has_val_set"] is True
print(f"  [smoke] hp 端到端 OK（6 成员，med best_iter={diag['best_iter']}）")

# ---- 2. 确定性 ----
m2, _, diag2 = wf._train_ml_model(PANEL, FACTORS, 10, {**ML, **hp_cfg})
_last = {s: df.set_index("date") for s, df in PANEL.items()}   # 预测侧：全量 df，末行=最新日
_cs = [f.calc_cross_section(_last) for f, _w in FACTORS]       # 每因子一条截面 Series(index=sym)
_Xp = np.column_stack([np.array([c[s] for s in sorted(PANEL)]) for c in _cs])  # (n_sym, n_feat)
assert _Xp.shape == (len(PANEL), len(FACTORS)), f"预测矩阵形状 {_Xp.shape}"
p1, p2 = m.predict(_Xp), m2.predict(_Xp)
assert np.array_equal(p1, p2), "同输入两次预测应比特一致"
print("  [smoke] 确定性 OK（预测比特一致）")

# ---- 3. 缺省不进分支 ----
m3, _, diag3 = wf._train_ml_model(PANEL, FACTORS, 10, ML)
assert "hp_groups" not in diag3, "缺省不应带 hp_groups"
assert diag3["n_members"] == 3, f"缺省应 3 种子成员，实际 {diag3['n_members']}"
assert diag3 == wf._train_ml_model(PANEL, FACTORS, 10, ML)[2], "缺省 diag 应可复现"
print("  [smoke] 缺省路径 OK（现行多种子路径不变）")

# ---- 4. 退化语义 ----
bad = {"hp_ensemble": {"enabled": True, "groups": [{}]},
       "seeds": [42], "min_best_iter": 10_000, "n_estimators": 5}
m4, _, diag4 = wf._train_ml_model(PANEL, FACTORS, 10, {**ML, **bad})
assert m4 is None and diag4["status"] == "empty" and diag4["reason"] == "degraded", \
    "全成员退化应空仓"
assert diag4.get("best_iter_by_member") == {"g0_s42": 4}, \
    f"退化成员应逐名登记，实际 {diag4.get('best_iter_by_member')}"
# 半退化不连坐：两组×单种子，min_best_iter 抬到组 A 与组 B 的早停轮数之间
# （先跑一轮看两组 best_iter，再把阈值卡在中间——构造真实"一组退化一组存活"）
import lightgbm as lgb  # noqa: F401  （内部已 import，占位保证可复跑）
_probe_cfg = {**ML, "hp_ensemble": {"enabled": True, "groups": [{}, {"learning_rate": 0.03, "max_depth": 5, "num_leaves": 16}]}, "seeds": [42]}
_mp, _, _md = wf._train_ml_model(PANEL, FACTORS, 10, _probe_cfg)
_bis = _md["best_iter_by_member"]
assert set(_bis) == {"g0_s42", "g1_s42"}, f"探针应两组各一成员，实际 {_bis}"
_lo, _hi = sorted(_bis.values())
if _lo < _hi:   # 两组早停轮数可分（合成面板几乎必然，否则跳过半退化构造）
    half = {"hp_ensemble": {"enabled": True,
                            "groups": [{}, {"learning_rate": 0.03, "max_depth": 5, "num_leaves": 16}]},
            "seeds": [42], "min_best_iter": _lo, "n_estimators": 120}
    m5, _, diag5 = wf._train_ml_model(PANEL, FACTORS, 10, {**ML, **half})
    assert diag5["status"] == "ok" and diag5["n_members"] == 1, \
        f"低轮组退化应被剔除、高轮组存活（实际 {diag5.get('n_members')}）"
    assert diag5["dropped_by_member"], "被剔成员应登记"
    print(f"  [smoke] 半退化不连坐 OK（阈值={_lo}，存活 {[m for m in diag5['members']]}）")
else:
    print("  [smoke] 半退化构造不适用（两组同轮），跳过")

print("\n== M3 hp_ensemble 冒烟 全部通过 ==")
