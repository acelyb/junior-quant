"""
ML 合成分支（LightGBM）—— 线性合成的并行 opt-in 路径，非替代。

与 factors/composite.py 的关系：
- composite.py 走线性加权（flat=ICIR 等权 / twolevel=族内正交+族间风险预算）。
- 本模块用训练好的 LightGBM 模型对同一批因子截面值做非线性合成。
- 入口 composite_score_ml 的签名匹配 walk_forward.py 的 composite_score 调用形状，
  且 return_factors_cs=True 时返回 (score, cs_map)，让归因 hook（walk_forward.py:455-464）
  按 factor.name 取 cs 即可工作——cs 为各因子原始截面打分，与线性路径归因口径一致
  （combiner 无关的分解），ML 分支归因不会静默失败。

设计：
- 特征 = 各因子 calc_cross_section 截面值，按 symbol 对齐成矩阵 X（NaN→列均值填，
  与 composite.py:_build_matrix 同口径，截面中性）。
- 模型由 walk_forward 的 _train_ml_model 训练好后传入（per-window refit）。
- 因子数增长零代码改动：遍历 factors 列表即可，特征顺序由训练时冻结的 feature_names 约束。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from jq_framework.factors.base import FactorBase


def _build_feature_matrix(price_dfs: dict[str, pd.DataFrame],
                          factors: list[tuple[FactorBase, float]],
                          feature_names: list[str],
                          series_caches: dict[str, dict[str, pd.Series]] | None = None
                          ) -> tuple[np.ndarray, list[str]]:
    """把各因子截面值对齐成特征矩阵 X（NaN→列均值填，截面中性）。

    复用 composite.py:_build_matrix 的填充口径，保证训练/测试特征工程逐字一致。
    返回 (X, syms)：syms 为价格数据有的、且至少一个因子有值的 symbol 顺序（X 行序对齐）。

    series_caches: 可选 {factor.name: {symbol: 全历史因子序列}}。提供时 calc_cross_section
        跳过逐股票 calc_value，按当前时点日期从预算序列索引——全市场 1800 只 × 18 因子下
        把"每调仓日从头算 rolling"降为"每窗口每股票算一次 calc_series"（与 linear 路径
        _build_series_caches 同机制）。None 走原逐点路径（逐字不变，向后兼容）。
    """
    # 各因子截面打分：{factor.name: pd.Series(index=symbol)}
    cs_list = []
    for factor, _ in factors:
        sc = series_caches.get(factor.name) if series_caches else None
        cs = factor.calc_cross_section(price_dfs, series_cache=sc)
        cs_list.append((factor.name, cs))

    # 行对齐：取所有因子 cs index 的并集（与 composite.py flat 路径 scores.add 同口径）。
    # 但 ML 需要确定行序——用所有非空因子 index 的交集更稳（缺任一特征的样本由均值填兜底，
    # 但完全没有价格的 symbol 不进 X）。这里用并集，NaN 全填均值（与 _build_matrix 一致）。
    sym_sets = [cs.index for _, cs in cs_list if not cs.empty]
    if not sym_sets:
        return np.empty((0, len(feature_names))), []
    syms = sorted(set().union(*[set(s) for s in sym_sets]))

    n, k = len(syms), len(feature_names)
    X = np.full((n, k), np.nan)
    name_to_col = {nm: j for j, nm in enumerate(feature_names)}
    for nm, cs in cs_list:
        if nm not in name_to_col:
            continue  # 训练时没见过的因子（理论上不会发生，feature_names 训练时冻结）
        vals = cs.reindex(syms).values.astype(float)
        # 截面全 NaN（早期窗口基本面缺失等）时 nanmean 报 RuntimeWarning，已有兜底（col_mean=0），
        # 抑制 warning 避免长历史日志被刷屏。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_mean = np.nanmean(vals)
        if np.isnan(col_mean):
            col_mean = 0.0
        vals = np.where(np.isnan(vals), col_mean, vals)
        X[:, name_to_col[nm]] = vals
    return X, syms


def _build_cs_map(price_dfs: dict[str, pd.DataFrame],
                  factors: list[tuple[FactorBase, float]],
                  series_caches: dict[str, dict[str, pd.Series]] | None = None
                  ) -> dict[str, pd.Series]:
    """各因子原始截面打分 {factor.name: cs}，供归因 hook（combiner 无关分解）。

    series_caches 同 _build_feature_matrix：提供时走预算序列索引，跳过逐股票 calc_value。
    """
    cs_map = {}
    for factor, _ in factors:
        sc = series_caches.get(factor.name) if series_caches else None
        cs = factor.calc_cross_section(price_dfs, series_cache=sc)
        cs_map[factor.name] = cs if not cs.empty else pd.Series(dtype=float)
    return cs_map


def composite_score_ml(price_dfs: dict[str, pd.DataFrame],
                       factors: list[tuple[FactorBase, float]],
                       ml_model,
                       feature_names: list[str],
                       return_factors_cs: bool = False,
                       series_caches: dict[str, dict[str, pd.Series]] | None = None,
                       ml_cfg: dict | None = None,
                       ) -> pd.Series | tuple:
    """
    ML（LightGBM）非线性合成打分。

    price_dfs: {symbol: df}，每个 df 已截到当前时点
    factors: [(因子实例, 权重), ...]——ML 模式下 weight 不参与打分，仅用于归因 hook 遍历
    ml_model: 训练好的 LightGBM 模型（由 walk_forward._train_ml_model 训练）
    feature_names: 训练时冻结的特征列顺序（因子名），保证预测时列对齐
    return_factors_cs: True 时返回 (score, {factor.name: cs})，cs=各因子原始截面打分
    series_caches: 可选 {factor.name: {symbol: 全历史因子序列}}，预算序列加速（见 _build_feature_matrix）
    ml_cfg: 可选 walk_forward.ml 配置块。M5 特征工程（feature_engineering）启用时，预测侧
        按与训练侧 _build_train_xy 同一派生源补齐派生列（交互乘积+截面状态），列序一致。
        缺省 None / 无 feature_engineering 块 = 现行路径逐字不变。
    返回: pd.Series(index=symbol) ML 打分，越大越好；或 (score, cs_map)
    """
    X, syms = _build_feature_matrix(price_dfs, factors, feature_names, series_caches)
    # ---- M5 预测侧派生（factors/feature_engineering.py 单一实现源）：训练侧冻结的
    # feature_names 含派生名（如 momentumxvol_dryup / fs_ret_disp）时按名重建派生列，
    # 填充口径与训练侧逐字对齐（见 rebuild_derived_columns docstring）。
    # 缺省（无派生名）零改动，ml_cfg 不需要传——派生名自描述。----
    if X.size:
        from jq_framework.factors.feature_engineering import rebuild_derived_columns
        X = rebuild_derived_columns(X, price_dfs, feature_names)
    if len(syms) == 0 or ml_model is None:
        score = pd.Series(dtype=float)
        cs_map = _build_cs_map(price_dfs, factors, series_caches) if return_factors_cs else {}
        return (score, cs_map) if return_factors_cs else score

    preds = ml_model.predict(X)
    score = pd.Series(preds, index=syms)
    if return_factors_cs:
        cs_map = _build_cs_map(price_dfs, factors, series_caches)
        return score, cs_map
    return score
