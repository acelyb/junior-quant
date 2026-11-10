"""M5 特征工程派生层（计划§八 M5，2026-10-27 预登记）。

非线性特征源：交互特征（现有因子截面两两乘积，机制先验手选对）+ 市场状态变量
（逐采样日截面级状态特征）。单一实现源，训练侧（walk_forward._build_train_xy）与
预测侧（factors.composite_ml._build_feature_matrix）共用同一函数——两侧口径分叉=
训练/预测特征不一致泄漏，故必须同函数。

缺省（ml_cfg.feature_engineering 缺省/空）= 两侧行为逐字不变（零派生）。此性质由
reg4 字节回归门（对 reg3 参照）+ 冒烟断言（scripts/m5_feat_smoke.py）双证。

设计要点：
- 交互特征在"截面已标准化"的因子值上做乘积（calc_cross_section 输出已是 rank/zscore
  标准化后的截面分），且在调用方现行 NaN 填充之后做（两臂都是填充值）——两侧填充
  口径本就逐字对齐，派生阶段不引入新 NaN，口径继续一致。
- 状态变量是"日内恒定、跨日变化"的截面级特征：训练样本跨约 252 个采样日，树可学
  "状态×个股特征"的条件效应——单股截面值给不了这个信息（每行只有该股当日位次）。
- 状态变量仅从面板自身价格派生（无新数据依赖，无未来函数：只用当日及以前收盘价）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 手选交互对（机制先验，非扫描产物——扫描=多重检验，预登记纪律）
DEFAULT_INTERACTIONS: list[tuple[str, str]] = [
    ("momentum", "vol_dryup"),    # 趋势的量能确认：缩量上行=真趋势
    ("reversal", "vol_dryup"),    # 低位缩量=抛压枯竭，反弹确认
    ("momentum", "size"),         # 小盘动量：小盘里动量更有效的条件效应
    ("ivol", "size"),             # 小盘低波异象：两风险因子的交互带
]

# 状态变量名（fs_ 前缀 = feature state）
STATE_FEATURES = ["fs_ret_disp", "fs_breadth", "fs_vol_level"]


def parse_fe_cfg(ml_cfg: dict) -> dict:
    """解析 ml_cfg.feature_engineering 块。缺省=关闭（不派生，字节等价路径）。"""
    fe = ml_cfg.get("feature_engineering") or {}
    return {
        "interactions": bool(fe.get("interactions", False)),
        "state": bool(fe.get("state", False)),
    }


def add_interactions(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """行级交互特征：手选对乘积，追加为尾部列。行级纯函数，训练/预测两侧通用。

    因子不在 feature_names 里的对静默跳过（如 alpha158 批）——两侧同规则派生同名列。
    """
    col = {nm: j for j, nm in enumerate(feature_names)}
    parts, names = [X], list(feature_names)
    for a, b in DEFAULT_INTERACTIONS:
        if a in col and b in col:
            parts.append((X[:, col[a]] * X[:, col[b]]).reshape(-1, 1))
            names.append(f"{a}x{b}")
    return np.hstack(parts), names


def state_values(price_dfs: dict[str, pd.DataFrame] | None) -> list[float | None]:
    """当日截面状态三元组（全部仅用当日及以前的收盘价，无未来函数）。

    fs_ret_disp  : 截面 20 日收益的截面标准差（分化度：分散市选股空间大）
    fs_breadth   : 截面 20 日收益>0 的占比（宽度：普涨普跌状态）
    fs_vol_level : 截面 20 日已实现波动（日收益 std）的均值（波动状态）

    返回 [v0, v1, v2]（顺序对齐 STATE_FEATURES），任一算不出（截面不足等）为 None
    → 调用方填 NaN（与两侧现行 NaN 兜底口径一致）。
    """
    rets, vols = [], []
    if price_dfs:
        for df in price_dfs.values():
            if df is None or len(df) < 21:
                continue
            c = df["close"].astype(float)
            r20 = float(c.iloc[-1] / c.iloc[-21] - 1.0)
            r = c.pct_change().iloc[-20:]
            if r.isna().any():
                continue
            rets.append(r20)
            vols.append(float(r.std()))
    if len(rets) < 10:  # 截面太薄，状态不可信 → None（不喂噪声）
        return [None, None, None]
    rs = pd.Series(rets)
    return [
        float(rs.std()),
        float((rs > 0).mean()),
        float(np.mean(vols)),
    ]


def rebuild_derived_columns(X: np.ndarray,
                            price_dfs: dict[str, pd.DataFrame] | None,
                            feature_names: list[str]) -> np.ndarray:
    """预测侧：按训练侧冻结的派生特征名重建派生列（composite_score_ml 用）。

    feature_names 的命名约定（训练侧 _build_train_xy 同构追加）：
      - 基础因子名在前（个数 = _build_feature_matrix 实际填充的列数）
      - 交互名 "a×b" 写作 "axb"（add_interactions 生成）
      - 状态名 fs_ 前缀（STATE_FEATURES）

    填充口径（与训练侧逐字对齐）：
      - 交互：两臂都是"填充后"值相乘（训练侧在 NaN 兜底后 add_interactions；
        预测侧 _build_feature_matrix 已填均值后相乘）——两侧填充口径本就逐字一致。
      - 状态：state_values(price_dfs) 当日值；None → 该列 0.0（训练侧语义：整列
        NaN → col_means 兜底 0.0；单日内恒定故列均值=当日值）。

    无派生名（缺省生产路径）→ X 原样返回（零改动）。
    """
    derived = [nm for nm in feature_names if "x" in nm or nm.startswith("fs_")]
    if not derived:
        return X
    n_base = len(feature_names) - len(derived)
    base_col = {nm: j for j, nm in enumerate(feature_names[:n_base])}
    parts = [X[:, :n_base]]
    for nm in feature_names[n_base:]:
        if nm.startswith("fs_"):
            sv = state_values(price_dfs)
            v = sv[STATE_FEATURES.index(nm)] if nm in STATE_FEATURES else None
            parts.append(np.full((X.shape[0], 1), 0.0 if v is None else float(v)))
        else:  # 交互 "axb"：拆出两臂（以 base_col 里存在的因子名贪心匹配）
            a, b = _split_interaction(nm, base_col)
            if a is None:
                parts.append(np.zeros((X.shape[0], 1)))
            else:
                parts.append((X[:, base_col[a]] * X[:, base_col[b]]).reshape(-1, 1))
    return np.hstack(parts)


def _split_interaction(nm: str, base_col: dict[str, int]) -> tuple[str | None, str | None]:
    """把交互名拆回两臂因子名。以第一个 'x' 切分（因子名不含 'x'，'axb'→a,b）。"""
    a, _, b = nm.partition("x")
    if a in base_col and b in base_col:
        return a, b
    return None, None
