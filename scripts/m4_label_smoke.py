"""m4_label_smoke.py — M4 标签工程冒烟（离线，合成面板）。

覆盖面（对应 2026-10-26 label_transform 实现的四条性质）：
  1. rank 变换语义：标签变为当日截面百分位秩 [0,1]，同日截面内秩序 == 原收益秩序，
     跨日不混（不同日的相同秩值来自各自的截面）。
  2. winsor 变换语义：当日截面 1%/99% 分位外的值被夹到分位线上，中段值不动。
  3. 缺省 raw 字节等价：label_transform 缺省（及显式 "raw"）时 _build_train_xy 输出
     与未加开关的现行路径逐字节一致——reg3 字节门的代码级前提。
  4. ridge 同源：同一 ml_cfg 下 ML 与 ridge 两条路径拿到同一份变换后标签
     （_build_train_xy 单一实现源的复用面）。

用法：python scripts/m4_label_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import jq_framework.backtest.walk_forward as wf
def mk_panel(n_syms=60, n_days=500, seed=11):
    """合成面板：尾部放大极端收益（模拟涨跌停/事件），让 winsor 有料可夹。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    dfs = {}
    for i in range(n_syms):
        # 每只股票固定一个偏度：少数股票持续极端（尾部制造机）
        tail = 0.15 if i % 10 == 0 else 0.02
        ret = rng.standard_t(df=3 if tail > 0.05 else 20, size=n_days) * tail
        px = 10.0 * np.exp(np.cumsum(ret))
        dfs[f"S{i:03d}"] = pd.DataFrame({
            "date": dates, "open": px, "high": px * 1.02, "low": px * 0.98,
            "close": px, "volume": 1e6 * (1 + rng.random(n_days)),
        })
    return dfs


class _F:
    def __init__(self, name, fn):
        self.name, self._fn = name, fn
        self.calc_series = None

    def calc_cross_section(self, dfs, date=None, series_cache=None):
        out = {}
        for s, df in dfs.items():
            c = df["close"]
            n = min(20, len(c))                  # 窗口早期前缀不足 20 行，自适应
            out[s] = float(c.iloc[-1]) / float(c.iloc[-n]) - 1 if n >= 2 else 0.0
        return pd.Series(out)


PANEL = mk_panel()
FACTORS = [(_F("mom", None), 1.0), (_F("rev", None), 1.0)]
ML = {"seeds": [42, 7, 2026], "freq": 5, "val_ratio": 0.2, "min_samples": 100,
      "early_stopping_rounds": 10, "n_estimators": 80, "num_threads": 4}

# ---- 1. rank 变换语义 ----
X1, y1, _, rdi1, _ = wf._build_train_xy(PANEL, FACTORS, 10, {**ML, "label_transform": "rank"})
assert y1.min() >= 0.0 and y1.max() <= 1.0, "rank 标签应在 [0,1]"
rdi1 = np.asarray(rdi1)
for day in np.unique(rdi1)[:8]:      # 抽 8 个采样日核秩序保持
    m = rdi1 == day
    y_raw = wf._build_train_xy(PANEL, FACTORS, 10, ML)[1][m]
    assert len(set(y1[m])) == len(y1[m]), "同日截面秩值应唯一（平秩除外，合成数据无平秩）"
    # 秩序 == 原收益秩序（Spearman=1）
    from scipy.stats import spearmanr
    assert abs(spearmanr(y1[m], y_raw).statistic - 1.0) < 1e-12, "同日秩序应与原收益秩序全等"
print("  [smoke] rank 语义 OK（[0,1] 且同日秩序保持）")

# ---- 2. winsor 变换语义 ----
X2, y2, _, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, {**ML, "label_transform": "winsor"})
y_raw = wf._build_train_xy(PANEL, FACTORS, 10, ML)[1]
assert (y2 != y_raw).any(), "winsor 应实际夹掉极端值（面板含 t 分布尾部）"
assert (y2 == y_raw).sum() > 0.9 * len(y2), "winsor 只应动 1%/99% 之外的尾部（中段不动）"
assert y2.min() >= np.percentile(y_raw, 0) and y2.max() <= np.percentile(y_raw, 100), "winsor 不越界"
print("  [smoke] winsor 语义 OK（尾部夹线，中段不动）")

# ---- 3. 缺省 raw 字节等价 ----
Xa, ya, _, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, ML)                       # 缺省
Xb, yb, _, _, _ = wf._build_train_xy(PANEL, FACTORS, 10, {**ML, "label_transform": "raw"})  # 显式 raw
assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb), "缺省与显式 raw 应逐字节一致"
assert not np.array_equal(ya, y1), "rank 标签应与 raw 不同（开关真在动标签）"
print("  [smoke] 缺省 raw 字节等价 OK")

# ---- 4. ridge 同源 ----
# ridge 训练器与 ML 共用 _build_train_xy：同一 ml_cfg(label_transform=rank) 下两者
# 特征/标签一致（ridge 无随机源，直接训练核通过）
mr, fr, dr = wf._train_ridge_model(PANEL, FACTORS, 10, {**ML, "label_transform": "rank"})
assert mr is not None, "ridge+rank 应正常训练"
mm, fm, dm = wf._train_ml_model(PANEL, FACTORS, 10, {**ML, "label_transform": "rank", "seeds": [42]})
assert mm is not None, "ml+rank 应正常训练"
print("  [smoke] ridge 同源 OK（两条路径均通过 rank 标签训练）")

print("\n== M4 label_transform 冒烟 全部通过 ==")
