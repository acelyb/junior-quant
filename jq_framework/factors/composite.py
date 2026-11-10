"""
多因子合成与选股。

把多个因子加权合成一个综合打分，按分排序选前 N 只，并做换手控制。

防未来函数：所有计算只用当前时点及之前数据。
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from jq_framework.factors.base import FactorBase


def load_factor(spec: dict) -> FactorBase:
    """从配置实例化因子。spec: {module, class, params[, family, layer]}

    family 由本函数从 spec 注入到因子实例（路径 A：调用点列表推导零改动，
    .family 随实例进入 composite_score）。缺省回退到 factor.name（各成一族）。
    layer 由本函数注入（簇B 时序分层，见 config multifactor.layers）。缺省 None，
    _build_layers 里回退到 "slow"。layers.enabled=false 时 .layer 从不被读，零回归。
    """
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class"])
    factor = cls(params=spec.get("params", {}))
    factor.family = spec.get("family") or factor.name
    factor.layer = spec.get("layer")
    return factor


def composite_score(price_dfs: dict[str, pd.DataFrame],
                    factors: list[tuple[FactorBase, float]],
                    combine_cfg: dict | None = None,
                    return_factors_cs: bool = False,
                    series_caches: dict[str, dict[str, pd.Series]] | None = None) -> pd.Series | tuple:
    """
    多因子加权合成打分。

    price_dfs: {symbol: df}，每个 df 已截到当前时点
    factors: [(因子实例, 权重), ...]，权重已含方向符号（ic 模式 sign*clip(|icir|)，
              static 模式配置 weight）
    combine_cfg: 两层合成配置（multifactor 的 {combine, families, orthogonalize} 子集）。
        None 或 combine != "twolevel" → 跑现有 flat 线性加权（逐字复现旧逻辑）。
        combine == "twolevel" → 族内正交+ICIR 加权合成 → 族间正交+风险预算。
    return_factors_cs: 若 True，返回 (score, {factor.name: cs})，其中 cs 是各因子在该时点
        的截面打分 Series（与归因 hook 现场重算逐字相同——正交在下游 numpy 矩阵上做，不
        mutate 这些 cs）。用于归因 hook 去重，避免调仓日把 calc_cross_section 算两遍。
        默认 False 保持旧签名（返回 Series），向后兼容。
    series_caches: 可选，{factor.name: {symbol: 全历史因子序列}}。提供时 calc_cross_section
        跳过逐股票 calc_value，按当前时点日期从预算序列索引——None 时走原逐点路径（逐字不变）。
    返回: pd.Series(index=symbol) 综合打分，越大越好；或 (score, cs_map) 当 return_factors_cs
    """
    combine = (combine_cfg or {}).get("combine", "flat")
    if combine == "twolevel":
        return _composite_twolevel(price_dfs, factors, combine_cfg, return_factors_cs, series_caches)

    scores = None
    cs_map: dict[str, pd.Series] = {}
    for factor, weight in factors:
        sc = series_caches.get(factor.name) if series_caches else None
        cs = factor.calc_cross_section(price_dfs, series_cache=sc)
        if return_factors_cs:
            cs_map[factor.name] = cs if not cs.empty else pd.Series(dtype=float)
        if cs.empty:
            continue
        weighted = cs * weight
        scores = weighted if scores is None else scores.add(weighted, fill_value=0)
    score = scores if scores is not None else pd.Series(dtype=float)
    return (score, cs_map) if return_factors_cs else score


# ============ 两层合成（族内正交 + 族间风险预算） ============

def _orthogonalize(X: np.ndarray) -> np.ndarray:
    """
    对称正交（order-independent，无族内排序偏差）。

    用 eigh 对 X'X 做白化 X_orth = X @ (X'X)^(-1/2)，再两步修正：
      1. 保波动：白化后按原始 std 重缩放每列——否则所有列变单位方差，
         风险平价退化为等权、size cap 永不生效。
      2. 保号：每列若与原始列负相关则翻号——白化符号任意，必须与 IC 训练方向对齐。

    单列（size 族）、单符号、共线（ridge）→ no-op / 回退。
    """
    if X.shape[1] <= 1 or X.shape[0] < 2:
        return X
    orig_std = X.std(axis=0)
    XtX = X.T @ X + 1e-10 * np.eye(X.shape[1])  # ridge 防病态
    try:
        eigvals, eigvecs = np.linalg.eigh(XtX)
        eigvals = np.clip(eigvals, 1e-12, None)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    except np.linalg.LinAlgError:
        return X
    X_orth = X @ inv_sqrt  # 白化（Z'Z=I，各列单位"范数"而非单位"std"——见下步校正）
    # 保波动 + 保号：白化后各列单位范数（std≈1/√n），需逐列重缩放到原始 std，
    # 否则族间波动被均匀缩小、虽然相对值不变但与"保波动"语义不符且让 1/vol 量纲混乱。
    for j in range(X_orth.shape[1]):
        s = X_orth[:, j].std()
        if orig_std[j] < 1e-12 or s < 1e-12:
            continue
        X_orth[:, j] *= orig_std[j] / s
        # 保号：与原始列负相关则翻号（白化符号任意，须与 IC 训练方向对齐）
        c = np.corrcoef(X_orth[:, j], X[:, j])[0, 1]
        if np.isnan(c):
            c = np.dot(X_orth[:, j], X[:, j])
        if c < 0:
            X_orth[:, j] *= -1
    return X_orth


def _risk_parity_weights(vols: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """
    截面风险平价（无未来函数）：w_f ∝ 1/σ_f，归一化，再加每族预算上限（瀑布 cap）。

    vols: [n_families] 各族今日截面 std
    caps: [n_families] 各族预算上限（1.0=不限）
    返回: 权重向量，和为 1
    """
    vols = np.where(vols < 1e-8, 1e-8, vols)  # 零波动族 floor
    w = 1.0 / vols
    w = w / w.sum()
    for _ in range(100):
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w[over] = caps[over]
        free = ~over
        if free.sum() == 0 or w[free].sum() <= 0:
            break
        w[free] += excess * (w[free] / w[free].sum())  # 超额按比例分给未触顶族
    return w


def _build_matrix(items, syms) -> np.ndarray:
    """把若干 pd.Series 对齐到 syms 并列拼成矩阵，NaN 用列均值填（截面中性）。"""
    n, k = len(syms), len(items)
    X = np.full((n, k), np.nan)
    for j, (_, _, cs) in enumerate(items):
        vals = cs.reindex(syms).values.astype(float)
        col_mean = np.nanmean(vals)
        if np.isnan(col_mean):
            col_mean = 0.0
        vals = np.where(np.isnan(vals), col_mean, vals)
        X[:, j] = vals
    return X


def _neutralize_size(score: pd.Series, size_cs: pd.Series) -> pd.Series:
    """截面 size 中性化：把 score 对 size 截面回归取残差，剥离 size 的纯线性暴露。

    score = α + β·size + ε，返回 ε。保留 size 与其他因子的协同信息（非线性/交叉项不被
    线性回归剥离），只滤掉 score 里可被 size 单变量线性解释的倾斜成分。每调仓日截面做一次，
    无未来函数。对应 Barra 风格中性的轻量替代（无组合优化器时的统计手段，见
    [[scoring-universe-leak-fix]] 同次诊断的 size 暴露讨论）。

    退化保护：截面 <10 只、size 全 NaN/常数、score 空 → 原 score 返回（no-op）。
    """
    if score.empty or size_cs is None or size_cs.empty:
        return score
    # 对齐到 score 的股票集
    s = size_cs.reindex(score.index)
    valid = ~(score.isna() | s.isna())
    if valid.sum() < 10:
        return score  # 截面太小，回归无意义
    x = s[valid].values.astype(float)
    y = score[valid].values.astype(float)
    if np.std(x) < 1e-12:
        return score  # size 截面常数（如全池同市值），无可剥离暴露
    # OLS: y = a + b*x，截面标准化 x 防数值问题
    x_std = (x - x.mean()) / (x.std() + 1e-12)
    A = np.column_stack([np.ones_like(x_std), x_std])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    out = score.copy().astype(float)
    out[valid] = resid
    # 未对齐/NaN 位置保留原值（不参与回归也不强行中性化）
    return out


def _composite_twolevel(price_dfs: dict[str, pd.DataFrame],
                        factors: list[tuple[FactorBase, float]],
                        combine_cfg: dict | None,
                        return_factors_cs: bool = False,
                        series_caches: dict[str, dict[str, pd.Series]] | None = None) -> pd.Series | tuple:
    """两层合成：族内正交+ICIR 加权 → 族间正交+风险预算。

    return_factors_cs=True 时返回 (score, {factor.name: cs})：cs 为各因子该时点截面打分
    （正交在 _build_matrix/_orthogonalize 的 numpy 矩阵上做，不 mutate 这些 cs，故与归因
    hook 现场重算逐字相同）。每个返回点（空/单族/正常）都带 cs_map，防退化日丢归因条目。
    """
    orth = (combine_cfg or {}).get("orthogonalize", {})
    within_orth = orth.get("within_family", True)
    across_orth = orth.get("across_family", True)
    across_weighting = orth.get("across_family_weighting", "risk_parity")
    families_cfg = (combine_cfg or {}).get("families", {})

    # 退化保护：截面太小无正交意义，回退 flat
    use_twolevel = len(factors) >= 2

    # 按 family 分组（缺省 family = factor.name，各成一族）
    groups: dict[str, list] = {}
    for factor, w in factors:
        fam = getattr(factor, "family", None) or factor.name
        groups.setdefault(fam, []).append((factor, w))

    # --- 族内：正交 + ICIR 加权 → 族合成 ---
    family_composites: dict[str, pd.Series] = {}
    family_caps: dict[str, float] = {}
    cs_out: dict[str, pd.Series] = {}  # 各因子原截面打分（供归因去重复用）
    for fam, fam_factors in groups.items():
        sc_map = series_caches if series_caches else None
        items = [(f, w, f.calc_cross_section(price_dfs,
                  series_cache=sc_map.get(f.name) if sc_map else None))
                 for f, w in fam_factors]
        # 收集每个因子的 cs（空也存空 Series，保证归因 hook 能 get 到占位）
        if return_factors_cs:
            for f, _, cs in items:
                cs_out[f.name] = cs if not cs.empty else pd.Series(dtype=float)
        items = [(f, w, cs) for f, w, cs in items if not cs.empty]
        if not items:
            continue  # 该族全部无效（如 sign==0 已剔除）→ 跳过
        syms = sorted(set().union(*[cs.index for _, _, cs in items]))
        if len(syms) < 5:
            # 小截面退化：族内不做正交，直接 ICIR 加权
            X = _build_matrix(items, syms)
        else:
            X = _build_matrix(items, syms)
            if use_twolevel and within_orth and X.shape[1] > 1:
                X = _orthogonalize(X)
        w_vec = np.array([w for _, w, _ in items])
        family_composites[fam] = pd.Series(X @ w_vec, index=syms)
        cap = families_cfg.get(fam, {}).get("budget_cap")
        family_caps[fam] = float(cap) if cap is not None else 1.0

    def _pack(score):
        return (score, cs_out) if return_factors_cs else score

    if not family_composites:
        return _pack(pd.Series(dtype=float))
    if len(family_composites) == 1:
        return _pack(next(iter(family_composites.values())))

    # --- 族间：正交 + 风险预算 → 最终 ---
    syms = sorted(set().union(*[fc.index for fc in family_composites.values()]))
    fam_names = list(family_composites.keys())
    F = _build_matrix([(n_, None, family_composites[n_]) for n_ in fam_names], syms)
    if use_twolevel and across_orth and F.shape[1] > 1 and len(syms) >= 5:
        F = _orthogonalize(F)
    if across_weighting == "risk_parity":
        vols = F.std(axis=0)
        caps = np.array([family_caps[n_] for n_ in fam_names])
        fam_w = _risk_parity_weights(vols, caps)
    else:  # "equal" — orth-only 档
        fam_w = np.ones(len(fam_names)) / len(fam_names)
    final = pd.Series(F @ fam_w, index=syms)

    # --- size 残差化（opt-in，剥离 size 纯暴露）---
    # combine_cfg.neutralize_size=true 时，把最终 score 对 size 族截面回归取残差。
    # size 族即 family="size" 的 family_composite（单因子一族，截面打分= size cs）。
    # 注意：用 family_composites["size"] 而非 cs_out["size"]，因 cs_out 仅归因模式填充，
    # 残差化不应依赖归因开关。族间正交后 size 族信号已被 _orthogonalize 处理，但残差化针对
    # 的是最终 score vs 原 size 暴露，故用未正交的 size 族原始截面做回归变量。
    if (combine_cfg or {}).get("neutralize_size", False):
        size_fc = family_composites.get("size")
        if size_fc is not None and not size_fc.empty:
            final = _neutralize_size(final, size_fc)
    return _pack(final)


def select_portfolio(score: pd.Series, top_n: int = 50,
                     prev_holdings: list[str] | None = None,
                     turnover_buffer: int = 0) -> list[str]:
    """
    按综合打分选股，带换手控制。

    score: 综合打分序列（越大越好）
    top_n: 目标持仓数量
    prev_holdings: 上一期持仓（用于换手控制）
    turnover_buffer: 换手缓冲。若设为 k，则：
        新股票必须进入前 (top_n - k) 名才纳入（提高纳入门槛），
        旧股票只有跌出前 (top_n + k) 名才剔除（降低剔除敏感度）。
        这样能显著降低换手率。0 表示不缓冲，纯按排名选。
    返回: 选中的股票代码列表
    """
    if score.empty:
        # 审计#22（2026-09-24）：空打分多因数据故障（因子全空/特征缺失）而非"卖出信号"，
        # 引擎侧将按空持仓处理（清仓）。此处大声告警防静默——空 score 的调仓日
        # 是"最大规模调仓被记 0 换手"的隐形事故源。
        print("[WARN] select_portfolio: 空 score 输入 → 返回空持仓（调用方将清仓/保持空仓），"
              "请检查因子数据是否故障", flush=True)
        return []
    ranked = score.sort_values(ascending=False)

    if prev_holdings is None or turnover_buffer == 0:
        return ranked.head(top_n).index.tolist()

    # 换手控制：结合新旧排名
    prev_set = set(prev_holdings)
    new_threshold = max(top_n - turnover_buffer, 1)
    keep_threshold = top_n + turnover_buffer

    ranked_syms = ranked.index.tolist()
    top_keep = set(ranked_syms[:keep_threshold])  # 前 (top_n+buffer) 名

    # 旧持仓中仍在 keep 范围内的保留
    held = [s for s in prev_holdings if s in top_keep]
    # 新增：必须进入更严格的 new_threshold 名才纳入
    candidates = [s for s in ranked_syms[:new_threshold] if s not in prev_set]

    # 填满 top_n
    portfolio = held + candidates
    # 若不足，从排名里补
    if len(portfolio) < top_n:
        for s in ranked_syms:
            if s not in portfolio:
                portfolio.append(s)
            if len(portfolio) >= top_n:
                break
    return portfolio[:top_n]


def compute_turnover(new_holdings: list[str], prev_holdings: list[str]) -> float:
    """
    计算单边换手率：变动股票数 / 持仓数。
    """
    if not new_holdings:
        # 审计#22（2026-09-24 修）：清仓（new 空、prev 非空）是一次 100% 卖出的调仓，
        # 原实现记 0.0 把最大规模换手藏没。按公式口径 changed/(2×max(len)) = N/(2N) = 0.5
        # （= 完整换仓的一半，只有卖出腿）。两边皆空才是真 0。
        return 0.5 if prev_holdings else 0.0
    new_set, prev_set = set(new_holdings), set(prev_holdings)
    changed = len(new_set - prev_set) + len(prev_set - new_set)
    return changed / (2 * len(new_holdings))
