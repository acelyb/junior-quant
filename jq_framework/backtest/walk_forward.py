"""
滚动样本外回测（walk-forward）—— 防过拟合的核心框架。

为什么需要它：
- 全样本回测会用"未来才知道"的因子方向/参数选股，偷看未来，结果不可信。
- walk-forward 把时间切成连续窗口，每个窗口：
    训练段 [t0, t1]：用过去数据算因子 IC，确定每个因子的方向（正向/反向）和有效性
    测试段 [t1, t2]：用训练段确定的方向选股回测，这段是"样本外"
  滚动推进，把各测试段拼接成全程样本外净值。
- 这样每个选股决策只用了当时及以前的信息，杜绝未来函数。

训练段做什么：
- 对每个因子算训练段内的平均 Rank IC，取其符号作为因子方向（IC>0 正向，IC<0 反向）
- 若某因子 IC 不显著（|t|<1 或 |IC|<0.01），训练段可降权或剔除（见 min_ic_abs）

与 portfolio_runner 的关系：
- portfolio_runner 是"全样本 + 固定方向"回测，用于快速看效果
- walk_forward 是"分段 + 动态方向"回测，用于出可信结论
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jq_framework.data.fetchers.fetcher import (get_index_constituents, get_batch_daily, get_index_daily,
                          get_stock_industry_map, get_index_real_daily)
from jq_framework.data.fetchers.fetcher_fundamental import build_fundamental_data
from jq_framework.factors.composite import load_factor, composite_score, select_portfolio, compute_turnover
from jq_framework.factors.composite_ml import composite_score_ml
from jq_framework.factors.ic import calc_factor_ic, ic_summary, _future_return
from jq_framework.backtest.direction import DirectionPolicy, resolve_policy
from jq_framework.risk.manager import RiskManager
from jq_framework.risk.timing import MarketTimer
from jq_framework.risk.cppi import CPPITimer
from jq_framework.risk.vol_target import VolTargetTimer
from jq_framework.backtest.limits import is_tradeable
from jq_framework.backtest.intraday_filter import filter_by_volume, filter_by_open_strength, filter_by_amplitude

# 训练段 IC 计算的并行化：各因子 IC 互相独立，多进程并行（线程因 GIL 在纯 Python
# 的逐日×逐股循环上不释放，实测反而更慢）。fork 模式下子进程继承父内存，
# train_price_dfs 不重复序列化；因子对象本身很小（~180B，行业/基本面数据 fork 继承）。
# 默认上限 8 进程（用户授权 8 核）；单因子或进程数<2 时直接串行，零开销。
_IC_MAX_WORKERS = max(1, min(8, (os.cpu_count() or 4)))
# fork 继承的训练段数据（仅 _ic_worker 读，不在任务参数里序列化大对象）
_ic_train_dfs: dict = {}

# worker 侧 DirectionPolicy 缓存：{(weighting, floor, cap): policy}。
# 在 fork 出的子进程里各建各的（极小对象，开销可忽略），避免每任务重复构造。
_policy_cache: dict = {}


def _resolve_policy_for_worker(weighting: str, floor: float, cap: float) -> DirectionPolicy:
    key = (weighting, floor, cap)
    pol = _policy_cache.get(key)
    if pol is None:
        pol = resolve_policy(weighting=weighting, clamp=(floor, cap))
        _policy_cache[key] = pol
    return pol


def _ic_worker(args):
    """模块级 worker（pickle 友好）：算单因子 IC 并返回方向/权重元组。

    train_price_dfs 从 fork 继承的模块全局读，不进任务参数——避免每任务序列化 10MB+。
    """
    factor, weight, horizon, freq, min_ic_abs, weighting, floor, cap = args
    ic_df = calc_factor_ic(_ic_train_dfs, factor, horizon=horizon, freq=freq)
    summary = ic_summary(ic_df["ic"]) if not ic_df.empty else {}
    ic_mean = summary.get("ic_mean", np.nan)
    t_stat = summary.get("t_stat", np.nan)
    icir = summary.get("icir", np.nan)
    if np.isnan(ic_mean) or abs(ic_mean) < min_ic_abs or (not np.isnan(t_stat) and abs(t_stat) < 1.0):
        sign = 0
    else:
        sign = 1 if ic_mean > 0 else -1
    # 权重换算交给 DirectionPolicy（默认内置 static / ic，行为与改造前逐位一致）
    w = _resolve_policy_for_worker(weighting, floor, cap).weight_from_ic(
        sign, ic_mean, icir, weight)
    return (factor, w, sign, ic_mean, t_stat, icir)


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def _segment_dates(dates: list, train_days: int, test_days: int, step_days: int):
    """
    把日期轴切成 walk-forward 窗口。

    每个窗口: 训练段 train_days 天 + 测试段 test_days 天。
    step_days: 窗口滚动步长（test_days 则不重叠，<test_days 则测试段重叠）。
    返回: [(train_dates, test_dates), ...]
    """
    windows = []
    n = len(dates)
    start = 0
    while start + train_days + test_days <= n:
        train = dates[start:start + train_days]
        test = dates[start + train_days:start + train_days + test_days]
        windows.append((train, test))
        start += step_days
    return windows


def _train_factor_directions(train_price_dfs, factors, horizon, freq, min_ic_abs,
                             weighting="static", clamp=(0.25, 1.5)):
    """
    训练段：对每个因子算 IC，返回 [(factor, weight, sign, ic_mean, t_stat, icir), ...]。
    sign: +1 正向 / -1 反向 / 0 剔除（IC 不显著）

    weighting:
      - "static": weight = 配置静态权重，方向用 sign 动态定符号（旧行为，保兼容）
      - "ic": weight = sign * clip(|icir|, floor, cap)（icir=NaN 回退 |ic_mean|），
              用训练段 IC 幅度加权，稳定正向因子被重配。配置 factors.weight 在 ic 模式下被忽略
    clamp: (floor, cap) IC 幅度裁剪范围，防 icir 爆炸单因子独大 / 低 icir 被丢弃

    并行：各因子 IC 独立，用 ProcessPoolExecutor 并行（_IC_MAX_WORKERS 进程）。
    fork 继承 train_price_dfs 到模块全局 _ic_train_dfs，子进程不重复序列化大对象。
    结果顺序与 factors 输入一致（map 保序），口径与串行逐字相同。
    """
    floor, cap = clamp
    tasks = [(factor, weight, horizon, freq, min_ic_abs, weighting, floor, cap)
             for factor, weight in factors]

    # 串行与并行共用：_ic_worker 从模块全局 _ic_train_dfs 读训练段数据
    global _ic_train_dfs
    _ic_train_dfs = train_price_dfs
    try:
        # 单因子或预算进程数<2 → 直接串行，避免建池开销（口径不变）
        if len(tasks) <= 1 or _IC_MAX_WORKERS < 2:
            return [_ic_worker(t) for t in tasks]

        from concurrent.futures import ProcessPoolExecutor
        # 每窗口新建池：fork 时子进程快照当前 _ic_train_dfs。池复用会冻结旧窗口数据，
        # 故不能用常驻池。建池开销（fork 几个子进程）相对 IC 计算可忽略。
        # tasks 只含小因子对象（~180B），train_price_dfs 靠 fork 继承不序列化。
        with ProcessPoolExecutor(max_workers=_IC_MAX_WORKERS) as ex:
            results = list(ex.map(_ic_worker, tasks))
    finally:
        _ic_train_dfs = {}
    return results


def _set_ic_train_dfs(train_dfs):
    """子进程 initializer（保留供非 fork start method 兜底，当前 fork 路径不用）。"""
    global _ic_train_dfs
    _ic_train_dfs = train_dfs


def _resolve_layers(mf: dict, universe: str) -> tuple[bool, dict | None]:
    """簇B时序分层解析（[[system-architecture-problems-and-roadmap]] 问题②：
    单一半衰期把所有信号挤进一条 reb42 管线，短窗 alpha 被持有 42 天磨成本）。

    multifactor.layers.enabled 缺省/false → (False, None)，run_walk_forward 走 legacy 单层
    路径逐字不变（零回归）。True → (True, {layer_name: {top_n, buffer, rebalance_days}})。
    每层先继承 layers.base 再被各自层覆盖；buffer/rebalance 依 universe 走分池覆盖
    （buffer_by_universe/rebalance_by_universe），与本文件 select 段 L683-702 同构。
    权重聚合（用户已定 A 分层子组合）：w_s = Σ_layers δ(l,s)×(1/n_layers)/N_l，见测试段分层分支。
    """
    lc = mf.get("layers", {}) or {}
    if not lc.get("enabled", False):
        return False, None
    base = lc.get("base", {}) or {}
    order = [k for k in lc.keys() if k not in ("enabled", "base")]
    if not order:
        raise ValueError("multifactor.layers.enabled=true 但未定义任何层")
    base_buf_pool = base.get("buffer_by_universe") or {}
    base_reb_pool = base.get("rebalance_by_universe") or {}
    specs = {}
    for name in order:
        l = lc.get(name) or {}
        l_buf_pool = l.get("buffer_by_universe") or {}
        l_reb_pool = l.get("rebalance_by_universe") or {}
        top_n = l.get("top_n", base.get("top_n", 20))
        # rebalance_days：层显式 > 层内分池 > base 分池 > base 全局
        if "rebalance_days" in l:
            reb = l["rebalance_days"]
        elif universe in l_reb_pool:
            reb = l_reb_pool[universe]
        elif universe in base_reb_pool:
            reb = base_reb_pool[universe]
        else:
            reb = base.get("rebalance_days", 1)
        # buffer：层内分池 > 层显式 turnover_buffer（快层更紧）> base 分池 > base 全局
        if universe in l_buf_pool:
            buf = l_buf_pool[universe]
        elif "turnover_buffer" in l:
            buf = l["turnover_buffer"]
        elif universe in base_buf_pool:
            buf = base_buf_pool[universe]
        else:
            buf = base.get("turnover_buffer", 0)
        specs[name] = {"top_n": int(top_n), "buffer": int(buf), "rebalance_days": max(1, int(reb))}
    return True, specs


def _feature_select(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                    sel_cfg: dict) -> tuple[list[str], dict]:
    """训练段内特征选择（业界 GKX/QLib 范式：变量选择为复杂度控制必要项）。

    双门筛选对齐 [[align-sota-ml-roadmap]]：
      1. OOF importance：5 折时序 OOF 训练 LGBM，取 gain 重要性均值，杀低于 top-N 阈值/或
         importance<最小绝对阈值的特征（冗余/弱信号特征会稀释模型，[[gate4-csi-all-revival-rejected]]
         ml_plus7 -8.38pp 已证加同族冗余=负增量）。
      2. 正交残差：对幸存特征做 Gram-Schmidt 正交化（按 importance 降序），后一特征对前者
         回归残差的方差比例 < min_resid_var 则判近冗余剔除（与 twolevel 族内正交同一思想）。
      3. 保留集合/数量由 sel_cfg 控制：max_features 硬上限（默认全保），manual 标志仅记录不保护。

    训练段内做（无未来数据泄漏），每窗口滚动重算。X/y 已是训练段全量（含后续早停验证集切分）。
    返回 (kept_feature_names, diag)。sel_cfg.enabled=false 时原样返回（零回归）。
    """
    if not sel_cfg.get("enabled", False):
        return feature_names, {"status": "disabled", "kept": len(feature_names)}
    import lightgbm as lgb
    from numpy.random import RandomState

    n = len(feature_names)
    if len(X) < 200 or n < 2:
        return feature_names, {"status": "noop_small", "kept": n}
    n_folds = int(sel_cfg.get("n_folds", 5))
    n_folds = max(2, min(n_folds, len(X) // 50))
    seed = int(sel_cfg.get("seed", 42))
    param = {
        "objective": "regression", "max_depth": 3, "num_leaves": 8,
        "learning_rate": 0.05, "min_child_samples": 50, "reg_lambda": 1.0,
        "colsample_bytree": 0.8, "verbose": -1, "seed": seed,
        "num_threads": int(sel_cfg.get("num_threads", 4)),
        # 同主训练的确定性开关（2026-10-08，防 OOF 重要性跨 run 漂移）
        "deterministic": bool(sel_cfg.get("deterministic", True)),
        "force_col_wise": bool(sel_cfg.get("force_col_wise", True)),
    }
    n_est = int(sel_cfg.get("n_estimators", 300))
    rng = RandomState(seed)
    # 时序分块 OOF（不随机打乱——防未来函数泄漏，同主训练口径）
    idx = np.arange(len(X))
    perm = idx  # 序列本身有序；按块切 5 折
    folds = np.array_split(perm, n_folds)
    importances = np.zeros(n)
    for fi, test_idx in enumerate(folds):
        train_mask = np.ones(len(X), bool)
        train_mask[test_idx] = False
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[~train_mask], y[~train_mask]
        if len(X_tr) < 100 or len(X_te) < 50:
            continue
        model = lgb.train(param, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names),
                          num_boost_round=n_est)
        importances += model.feature_importance(importance_type="gain") / n_folds
    if importances.sum() <= 0:
        return feature_names, {"status": "all_zero_imp", "kept": n}
    order = np.argsort(-importances)  # 重要性降序
    # 硬上限裁剪（max_features<0/缺省=全保）
    max_feats = int(sel_cfg.get("max_features", -1))
    keep_n = n if max_feats <= 0 else min(n, max_feats)
    # 可选绝对重要性下限（min_imp_frac：保留重要性≥max*frac 的特征）
    min_frac = float(sel_cfg.get("min_imp_frac", 0.0))
    imp_max = importances[order[0]]
    keep_idx = []
    for j in order:
        if len(keep_idx) >= keep_n:
            break
        if min_frac > 0 and importances[j] < imp_max * min_frac:
            break  # 重要性降序，后续更小，全断
        keep_idx.append(int(j))
    if sel_cfg.get("orthogonal_resid", True) and len(keep_idx) >= 2:
        # Gram-Schmidt 正交残差：按重要性降序，后一特征对已保留集合回归，残差方差比例<阈值判冗余
        min_resid = float(sel_cfg.get("min_resid_var", 0.05))
        kept_basis = [X[:, keep_idx[0]].astype(float)]  # 列向量列表
        out_idx = [keep_idx[0]]
        for j in keep_idx[1:]:
            vec = X[:, j].astype(float)
            for b in kept_basis:
                denom = float(b @ b)
                if denom > 1e-12:
                    vec = vec - (float(vec @ b) / denom) * b
            res_var = float(vec @ vec)
            if res_var <= min_resid * float(X[:, j] @ X[:, j]):
                continue  # 近线性相关，舍弃
            kept_basis.append(X[:, j].astype(float))
            out_idx.append(j)
    else:
        out_idx = keep_idx
    kept = [feature_names[j] for j in out_idx]
    return kept, {"status": "ok", "kept": len(kept), "total": n,
                  "imp": {feature_names[j]: float(importances[j]) for j in out_idx}}


def _build_series_caches(valid: dict, factors: list) -> dict:
    """为测试段预算各因子每股票的全历史因子序列缓存。

    只对实现了 calc_series 的因子做（未实现的因子返回空 dict，calc_cross_section
    自动回退逐点 calc_value 路径，逐字不变）。每窗口调用一次：把测试段"每调仓日×每股票
    从头算 rolling"降为"每窗口每股票算一次全序列、调仓日按日期索引"。
    返回 {factor.name: {symbol: Series(index=date)}}。
    """
    from jq_framework.factors.base import FactorBase
    caches = {}
    for f, _ in factors:
        if f.calc_series is FactorBase.calc_series:
            continue  # 未实现向量化，跳过（calc_cross_section 走原路径）
        sym_map = {}
        for sym, df in valid.items():
            if df is None or df.empty:
                continue
            d = df.set_index("date") if "date" in df.columns else df
            try:
                s = f.calc_series(d)
            except Exception:
                s = None
            if s is not None:
                sym_map[sym] = s
        caches[f.name] = sym_map
    return caches


def _apply_hard_gate(current: dict, date, cfg: dict) -> dict:
    """漏斗第1层：硬安全/流动性门（[[funnel-top10-rewrite]]）。

    在打分前把"不能买/不值得买"的票剔出选股宇宙（current），使打分排名不被垃圾股污染、
    select_portfolio 天然跳过。config: multifactor.hard_gate（默认关→零回归）。
    三条硬门（基于日线 OHLCV，无 ST 名称数据源故本轮跳过 ST）：
      - min_liquidity_turnover: 30 日日均成交额 (close×volume) 低于此值剔除（流动性红线）
      - max_suspension_days: 离当前调仓日的实际 bar 距今超过此天数 → 视为长期停牌剔除
      - min_listing_days: 上市(可见历史)不足此交易日数 → 新股剔除
    返回过滤后的 current。
    """
    cfg = cfg or {}
    min_liq = cfg.get("min_liquidity_turnover")  # 成交额阈值（元）
    max_susp = cfg.get("max_suspension_days")    # 停牌容忍天数
    min_list = cfg.get("min_listing_days")        # 最小上市交易日数
    if not (min_liq or max_susp or min_list):
        return current
    date = pd.Timestamp(date)
    out = {}
    n_liq = n_susp = n_ipo = 0
    for sym, df in current.items():
        if df is None or df.empty:
            continue
        # 上市天数：可见历史 bar 数（cut 到当前时点后的行数）
        if min_list and len(df) < min_list:
            n_ipo += 1
            continue
        # 停牌：最后一根 bar 距今过长（df 已 cut 到 <=date，last date 即最近实际交易日）
        if max_susp is not None and "date" in df.columns:
            last_bar = pd.Timestamp(df["date"].iloc[-1])
            if (date - last_bar).days > max_susp:
                n_susp += 1
                continue
        # 流动性：30 日日均成交额 = mean(close×volume)；vol 可缺失时退化为量
        if min_liq and len(df) >= 1:
            tail = df.tail(30)
            if "volume" in tail.columns and "close" in tail.columns:
                turnover = (tail["close"] * tail["volume"].fillna(0)).mean()
            elif "volume" in tail.columns:
                turnover = (tail["volume"].fillna(0) * 1.0).mean()
            else:
                turnover = float("inf")  # 无量列，不过度剔除
            if turnover < min_liq:
                n_liq += 1
                continue
        out[sym] = df
    if cfg.get("verbose"):
        print(f"   [硬门 {pd.Timestamp(date).date()}] 剔除 {len(current)-len(out)} 只"
              f"(流动性 {n_liq} / 停牌 {n_susp} / 新股 {n_ipo})，保留 {len(out)}", flush=True)
    return out


class _SeedEnsemble:
    """多种子集成预测器（P1-2 生产多种子集成，2026-10-21）。

    同一特征矩阵/标签/验证切分下逐种子训练的 LightGBM 列表，预测取成员均值。
    duck-type 兼容全部既有消费面：composite_ml.composite_score_ml 的 model.predict(X)、
    walk_forward 窗口日志的 model.feature_importance("gain")。单种子路径不构造此类
    （返回裸模型），历史 run 的字节复现不受影响。
    """

    def __init__(self, models, seeds):
        self.models = models
        self.seeds = seeds

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def feature_importance(self, importance_type="gain"):
        return np.mean([m.feature_importance(importance_type=importance_type)
                        for m in self.models], axis=0)


def _build_train_xy(train_price_dfs, factors, horizon, ml_cfg, series_caches=None,
                    pool_selector=None):
    """训练段特征/标签矩阵构建（P1-3 抽取，2026-10-21：ML 与 ridge 两训练档共用唯一实现）。

    自 _train_ml_model 原样语句迁移（含注释），语句零改动——ML 路径字节级不变由钉住
    复现 run（参照读数 +10.1308…%，compare_ml --snapshot）复核。返回
    (X, y, feature_names, row_day_idx, empty_diag)；empty_diag 非 None 表示样本不足，
    调用方须退化为空仓窗口（同迁移前语义）。
    """
    feature_names = [f.name for f, _ in factors]
    freq = int(ml_cfg.get("freq", 5))
    # M5 特征工程开关（缺省 None=关闭，现行路径逐字不变；见 factors/feature_engineering.py）
    from jq_framework.factors.feature_engineering import add_interactions, parse_fe_cfg, state_values, STATE_FEATURES
    fe_cfg = parse_fe_cfg(ml_cfg)
    fe_cfg = fe_cfg if (fe_cfg["interactions"] or fe_cfg["state"]) else None
    if fe_cfg is not None and fe_cfg["state"]:
        feature_names = feature_names + list(STATE_FEATURES)  # 冻结列序含状态名（预测侧按名重建）

    # 预算训练段因子序列缓存：全市场 1800 只 × 18 因子下，逐采样日逐股票 calc_value 是
    # ML 训练的 O(n_stocks×n_factors×n_sample_days) 瓶颈。改每股票算一次 calc_series、
    # 采样日按日期索引。由调用方传全历史 series_caches（跨窗口复用，见 _train_ml_model
    # docstring），未传时窗口内预算（旧行为）。calc_series 尾 == calc（烟测验证），口径
    # 逐字不变；未实现 calc_series 的因子缓存为空 dict，calc_cross_section 自动回退逐点路径。
    train_series_caches = series_caches if series_caches is not None \
        else _build_series_caches(train_price_dfs, factors)

    # 对齐所有股票日期轴（同 calc_factor_ic 口径）
    all_dates = sorted(set().union(*[set(df["date"]) for df in train_price_dfs.values()
                                     if not df.empty]))
    sym_data = {}
    for sym, df in train_price_dfs.items():
        if df.empty:
            continue
        sym_data[sym] = df.set_index("date")

    # 逐采样日建训练行：当日各因子截面值（特征）+ 各 symbol 未来 horizon 日收益（标签）
    feat_rows = []
    label_rows = []
    row_day_idx = []  # 每行对应的采样日序号（i 在采样循环里的值），供验证集按日切分+留 gap
    for i in range(0, len(all_dates) - horizon, freq):
        date = all_dates[i]
        # 截到 date 的数据（位置切片，含 date 行——与 calc_factor_ic 的 sub 同口径）
        current = {}
        for sym, d in sym_data.items():
            pos = int(np.searchsorted(d.index.values, np.datetime64(date)))
            if pos < len(d) and d.index[pos] == date:
                current[sym] = d.iloc[:pos + 1]
        if len(current) < 10:
            continue
        # ---- M6 逐日动态训练池（2026-10-30 预登记）：按当日可观测市值/流动性筛选训练样本。
        # 关键：池过滤必须在**截面打分之前**应用——calc_cross_section 在传入集合内做 rank
        # 标准化，先截池再打分意味着特征标定基准 = 训练池本身。这是 M6 的设计语义（训练域
        # 下沉），不是副作用；预登记第七节(b) 已登记该口径。
        # 缺省 pool_selector=None → 整块跳过，现行路径逐字不变（回归门钉证）。
        # 注意退化集：当日无任何新鲜市值时选择器返回**基线训练域**（选股池成分股），
        # 不是当日全部候选——早于市值数据起点的窗口因此按基线口径训练，而非全市场。
        if pool_selector is not None:
            _pooled = pool_selector(date, current.keys())
            if _pooled is not None:
                current = {s: d for s, d in current.items() if s in _pooled}
                if len(current) < 10:
                    continue      # 当日池过薄，该采样日跳过（与"截面不足"同语义）
        # 各因子截面打分（与预测侧 composite_ml 逐字一致）
        cs_list = []
        for factor, _ in factors:
            sc = train_series_caches.get(factor.name)
            cs = factor.calc_cross_section(current, series_cache=sc)
            cs_list.append((factor.name, cs))
        # 当日各因子截面均值（NaN 填充用）——与预测侧 _build_feature_matrix 逐字对齐：
        # 预测侧按"当日截面列均值"填 NaN，训练侧必须同口径，否则训练/预测 NaN 填充不一致
        # （[[ml-redundancy-tolerance-vs-linear]] 任务#3：原代码用跨所有训练行的全局列均值，
        # 与预测侧当日截面均值口径分叉，稀疏因子在 train/test 填充值系统性偏移）。
        cs_col_means = []
        for _, cs in cs_list:
            m = float(np.nanmean(cs.values)) if not cs.empty else 0.0
            cs_col_means.append(0.0 if not np.isfinite(m) else m)
        # ---- M5 市场状态变量（计划§八 M5，2026-10-27 预登记）：当日截面级状态特征，
        # 日内恒定（同采样日所有行同值）、跨日变化（树学"状态×个股特征"条件效应）。
        # 缺省 feature_engineering.state=false 时为零开销跳过（现行路径逐字不变）。----
        fe_state = fe_cfg["state"] if fe_cfg is not None else False
        day_state = state_values(current) if fe_state else None
        # 各 symbol 未来收益
        date_idx = all_dates.index(date)
        for sym, d in sym_data.items():
            if sym not in current:
                continue
            closes = d["close"]
            pos = int(np.searchsorted(closes.index.values, np.datetime64(date)))
            fr = _future_return(closes.index, closes, pos, horizon)
            if fr is None or np.isnan(fr):
                continue
            # 该 symbol 的特征行：各因子截面值，NaN→当日截面均值（与预测侧同口径）
            row = []
            for k, (_, cs) in enumerate(cs_list):
                v = cs.get(sym, np.nan)
                row.append(cs_col_means[k] if (v is None or not np.isfinite(v)) else v)
            if fe_state:
                for v in day_state:
                    row.append(np.nan if v is None else float(v))
            feat_rows.append(row)
            label_rows.append(fr)
            row_day_idx.append(i)

    if len(feat_rows) < int(ml_cfg.get("min_samples", 500)):
        diag = {"status": "empty", "reason": "insufficient_samples", "n_samples": len(feat_rows)}
        return None, None, feature_names, [], diag  # 样本不足，该窗口退化（ML/ridge 同语义）
    X = np.array(feat_rows, dtype=float)
    y = np.array(label_rows, dtype=float)
    # 残余 NaN 兜底（cs_col_means 已填当日截面均值，理论上无 NaN；防御极端全 nan 列）
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X = np.where(np.isnan(X), col_means, X)
    # ---- M5 交互特征（计划§八 M5）：行级乘积在 NaN 填充后追加（两臂都是填充值，
    # 与预测侧派生时机/口径逐字一致）。缺省关闭=零派生，X/feature_names 原样。----
    if fe_cfg is not None and fe_cfg["interactions"]:
        X, feature_names = add_interactions(X, feature_names)
    # 标签 NaN 行剔除
    valid_mask = ~np.isnan(y)
    X, y = X[valid_mask], y[valid_mask]
    # ---- M4 标签工程（计划§八 M4，2026-10-26 预登记）：label_transform 逐采样日截面变换
    # 训练标签。缺省 "raw"=原始未来收益，零改动；仅动标签，特征/预测侧不变——预测输出
    # 的是排序分，截面秩不变性使变换标签可直接对接。按 row_day_idx 分组做当日截面变换
    # （未来收益的截面比较只应在同日样本间进行）。----
    lt = str(ml_cfg.get("label_transform", "raw")).lower()
    if lt in ("rank", "winsor"):
        rdi_arr = np.asarray(row_day_idx)
        for day in np.unique(rdi_arr):
            m_day = rdi_arr == day
            yd = y[m_day]
            if lt == "rank":
                # 当日截面百分位秩（[0,1]，argsort 平秩防并列抖动）
                y[m_day] = (np.argsort(np.argsort(yd)).astype(float) + 0.5) / len(yd)
            else:  # winsor：当日截面 1%/99% 分位截断
                lo, hi = np.percentile(yd, [1.0, 99.0])
                y[m_day] = np.clip(yd, lo, hi)
    if len(X) < int(ml_cfg.get("min_samples", 500)):
        diag = {"status": "empty", "reason": "insufficient_samples", "n_samples": len(X)}
        return None, None, feature_names, [], diag
    return X, y, feature_names, row_day_idx, None


def _ml_base_params(ml_cfg: dict, seed: int) -> dict:
    """ML 基础超参字典（M1，2026-10-23 自 _train_ml_model 原样抽取，语句零改动——
    现行路径与 cv_select 分支共用同一构造，数值身份不变由字节回归门钉 run 复核）。"""
    return {
        "objective": ml_cfg.get("objective", "regression"),
        "max_depth": int(ml_cfg.get("max_depth", 3)),
        "num_leaves": int(ml_cfg.get("num_leaves", 8)),
        "learning_rate": float(ml_cfg.get("learning_rate", 0.05)),
        "min_child_samples": int(ml_cfg.get("min_child_samples", 50)),
        "reg_lambda": float(ml_cfg.get("reg_lambda", 1.0)),
        "colsample_bytree": float(ml_cfg.get("colsample_bytree", 0.8)),
        "verbose": -1,
        "seed": seed,
        # 确定性训练（2026-10-08）：实测同配置同数据同基准的两次 ML run 超额相差 17pp
        # （gate4_voldry 干净重跑 +5.44% vs reb_sweep ml_reb10 -11.81%），批内各场景
        # best_iter 也各自漂（209/174/234）——模型确实逐 run 在变。根因未定死：隔离
        # 玩具测试里 det=False 也能复现，故不全是 LightGBM 线程浮点序，另一嫌疑是管线
        # 层次序噪声（get_batch_daily 线程池完成序→字典序→特征/行序）。此处按最佳实践
        # 加固 deterministic+force_col_wise（消除 LightGBM 侧），残余方差由种子复跑
        # {reb10,reb21,reb42}×3种子 实测裁决。注意：开启后模型数值路径与历史 run 不同
        # （等效换一次 seed 抽签），历史结论按各自 run 日志留档。
        # ml_cfg.deterministic=false 可关回旧行为。
        "deterministic": bool(ml_cfg.get("deterministic", True)),
        "force_col_wise": bool(ml_cfg.get("force_col_wise", True)),
        # 线程数：默认-1=用满物理核。compare_ml 场景级多进程并行时若不约束，N 进程各开满核
        # → 28×N 线程在 28 核上抢（context-switch 雪崩）。单进程跑留 -1 最快；多进程并行设每进程
        # 配额（如 4）让总线程≈核数。ml_cfg.num_threads 可配；不配走 LightGBM 默认（行为不变）。
        "num_threads": int(ml_cfg.get("num_threads", -1)),
    }


def _ml_seeds(ml_cfg: dict) -> list[int]:
    """集成种子列表解析（M1，2026-10-23 自 _train_ml_model 原样抽取，语义零改动：
    seeds 列表→逐种子；单值→单元素；缺省→ml.seed；再缺省→42）。"""
    _seeds_cfg = ml_cfg.get("seeds")
    if isinstance(_seeds_cfg, (list, tuple)):
        seeds = [int(s) for s in _seeds_cfg if s is not None]
    elif _seeds_cfg is not None:
        seeds = [int(_seeds_cfg)]
    else:
        seeds = []
    if not seeds:
        seeds = [int(ml_cfg.get("seed", 42))]
    return seeds


def _train_ml_model_cv(X, y, feature_names, row_day_idx, horizon,
                       ml_cfg, cv_cfg, sel_diag=None):
    """M1 早停/超参 CV 化（计划§八 M1，2026-10-23 预登记）：扩窗式时序分块 CV 在小网格
    上选超参与等效轮数，替换"单片噪声验证集早停"——正则化强度不再由窗口末段 20% 单片
    验证集的噪声决定（病灶1：收缩旋钮被噪声拧）。

    机制（与预登记逐字对应）：
    - 采样日（row_day_idx 唯一值）切 n_folds+1 个顺序块；fold k 训练=前 k 块、验证=
      第 k+1 块；训练/验证间隔沿用现行 horizon 日 gap 语义（rdi ≤ 验证块最早日−horizon，
      防标签窗口重叠泄漏）。
    - 每组×每折：固定 CV 种子、fold 内早停（early_stopping_rounds 同现行值）；折得分=
      验证集 l2，组得分=折均值，组等效轮数=折 best_iter 中位数。
    - 终模型=最优组超参+等效轮数**定长**训练（无早停、无验证集、全样本），逐集成种子
      各一个（种子层语义与 P1-2 一致；无逐种子早停故退化判定在组层面）。
    - 退化语义对齐：最优组等效轮数 ≤ min_best_iter → 该窗空仓（与现行规则同义）；
      无任何有效组 → 空窗（reason=cv_no_valid_combo）。
    """
    import lightgbm as lgb

    n_folds = int(cv_cfg.get("n_folds", 3))
    early_rounds = int(cv_cfg.get("early_stopping_rounds",
                                  ml_cfg.get("early_stopping_rounds", 20)))
    cv_seed = int(cv_cfg.get("seed", 42))
    grid = cv_cfg.get("grid") or [
        {"learning_rate": 0.05, "max_depth": 3, "num_leaves": 8},    # 生产默认（锚点）
        {"learning_rate": 0.03, "max_depth": 3, "num_leaves": 8},
        {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 16},
        {"learning_rate": 0.03, "max_depth": 5, "num_leaves": 16},
    ]
    min_best_iter = int(ml_cfg.get("min_best_iter", 1))
    n_estimators = int(ml_cfg.get("n_estimators", 300))

    rdi = np.asarray(row_day_idx)
    blocks = np.array_split(np.unique(rdi), n_folds + 1)
    base = _ml_base_params(ml_cfg, cv_seed)

    combo_stats, fold_detail_by_combo = [], {}
    for combo in grid:
        p = dict(base)
        p.update(combo)
        tag = ",".join(f"{k}={combo[k]}" for k in sorted(combo))
        fold_scores, fold_rounds = [], []
        for k in range(1, n_folds + 1):
            val_days = blocks[k]
            min_val_day = int(val_days.min())
            val_mask = np.isin(rdi, val_days.tolist())
            train_mask = (~val_mask) & (rdi <= min_val_day - horizon)
            if train_mask.sum() < 1 or val_mask.sum() < 1:
                fold_scores.append(float("nan"))
                fold_rounds.append(0)
                continue
            tr_set = lgb.Dataset(X[train_mask], label=y[train_mask], feature_name=feature_names)
            va_set = lgb.Dataset(X[val_mask], label=y[val_mask], feature_name=feature_names)
            m = lgb.train(p, tr_set, num_boost_round=n_estimators, valid_sets=[va_set],
                          callbacks=[lgb.early_stopping(early_rounds, verbose=False)])
            fold_rounds.append(int(getattr(m, "best_iteration", 0) or 0))
            fold_scores.append(float(m.best_score["valid_0"]["l2"]))
        ok = [i for i, s in enumerate(fold_scores) if np.isfinite(s)]
        if not ok:
            combo_stats.append({"combo": tag, "score": None, "rounds": 0})
            continue
        combo_stats.append({
            "combo": tag,
            "score": float(np.mean([fold_scores[i] for i in ok])),
            "rounds": int(np.median([fold_rounds[i] for i in ok])),
        })
        fold_detail_by_combo[tag] = {
            "fold_scores": [round(s, 6) if np.isfinite(s) else None for s in fold_scores],
            "fold_rounds": fold_rounds,
        }

    valid = [c for c in combo_stats if c["score"] is not None]
    if not valid:
        diag = {"status": "empty", "reason": "cv_no_valid_combo", "cv": combo_stats}
        if sel_diag:
            diag["feature_selection"] = sel_diag
        return None, feature_names, diag
    best = min(valid, key=lambda c: c["score"])
    if best["rounds"] <= min_best_iter:
        diag = {"status": "empty", "reason": "degraded", "best_iter": best["rounds"],
                "min_best_iter": min_best_iter, "cv": combo_stats}
        if sel_diag:
            diag["feature_selection"] = sel_diag
        return None, feature_names, diag

    # 终模型：最优组超参 + 等效轮数定长训练（全样本、无早停），逐集成种子各一个
    best_params = dict(base)
    best_params.update(next(c for c in grid
                            if ",".join(f"{k}={c[k]}" for k in sorted(c)) == best["combo"]))
    seeds = _ml_seeds(ml_cfg)
    full_set = lgb.Dataset(X, label=y, feature_name=feature_names)
    models = []
    for s in seeds:
        p = dict(best_params)
        p["seed"] = s
        models.append(lgb.train(p, full_set, num_boost_round=best["rounds"]))

    diag = {"status": "ok", "best_iter": int(best["rounds"]),
            "n_estimators": int(best["rounds"]), "has_val_set": False,
            "n_train": len(X), "n_val": 0, "n_members": len(models), "seeds": seeds,
            "cv": {"chosen": best["combo"], "score": best["score"],
                   "rounds": int(best["rounds"]), "grid": combo_stats,
                   **fold_detail_by_combo.get(best["combo"], {})}}
    if sel_diag:
        diag["feature_selection"] = sel_diag
    return _SeedEnsemble(models, seeds), feature_names, diag


def _train_ml_model(train_price_dfs, factors, horizon, ml_cfg, series_caches=None,
                    pool_selector=None):
    """
    训练段：用所有因子的截面值作特征、未来 horizon 日收益作标签，训练 LightGBM 回归器。

    与 _train_factor_directions 并列（非扩展）：ML 自做特征选择，旁路 min_ic_abs 门；
    返回 (model, feature_names) 而非 6-tuple，故不破 static/ic 路径契约。

    特征工程逐字对齐预测侧（composite_ml._build_feature_matrix）：
      每个采样日对各因子调 calc_cross_section 得截面值 → 与同日各 symbol 未来收益对齐成行。
    样本量：~252 训练日 / freq × N 股。freq 去自相关（同 calc_factor_ic）。
    过拟合控制：浅树 + L2 + min_child_samples + 时序末段早停验证集 + walk-forward refit。

    series_caches: 可选，{factor.name: {symbol: 全历史因子序列}}。传入时跳过窗口内预算——
      walk_forward 每窗口的 train_dfs 是该窗口前缀（新 df 对象/新首尾日期），若窗口内对 Alpha158
      重算会每窗口触发全 158 列 _compute_all 重算 + 内容缓存跨窗口无限累积（csi500 单窗~数分钟、
      RSS 涨到 4.5GB+）。calc_cross_section 按日期索引缓存、只读训练前缀内的日期，故传全历史
      缓存结果逐字相同（多出的未来历史从不被索引）。None 时窗口内预算（向后兼容）。
      仅在 weighting=="ml" 的调用方预构建并复用；linear 路径不进此分支，零影响。

    返回: (model, feature_names)。model=None 表示样本不足（该窗口 ML 退化为空仓）。
    2026-10-21（P1-3）：特征/标签构建块原样迁移至 _build_train_xy（与 ridge 档共用），
    本函数保留特征选择、验证集切分与种子层，ML 单种子路径字节级不变。
    """
    import lightgbm as lgb

    # 特征/标签构建（P1-3 抽取至 _build_train_xy，语句原样迁移；ridge 档同源复用）
    X, y, feature_names, row_day_idx, _empty_diag = _build_train_xy(
        train_price_dfs, factors, horizon, ml_cfg, series_caches,
        pool_selector=pool_selector)
    if _empty_diag is not None:
        return None, feature_names, _empty_diag  # 样本不足，该窗口 ML 退化为空仓

    # 训练段内特征选择（业界范式复杂度控制，默认关→零回归）：
    # 对全特征（11 手工 + 注入的 Alpha158 等）做 OOF importance + 正交残差双门，列切片后重绑
    # feature_names/X，主模型只用保留子集训练 → 预测侧 feature_names 冻结也只用该子集（零改动）。
    sel_cfg = ml_cfg.get("feature_selection") or {}
    sel_diag = None
    if sel_cfg.get("enabled", False) and len(feature_names) >= 2:
        kept, sel_diag = _feature_select(X, y, feature_names, sel_cfg)
        keep_mask = [fn in set(kept) for fn in feature_names]  # 保持原始列顺序
        # 空筛退避：一个都不留时保持全量（宁可不筛不因错误空仓）
        if sum(keep_mask) >= 1:
            X = X[:, keep_mask]
            feature_names = [fn for fn, m in zip(feature_names, keep_mask) if m]

    # ---- M1 早停/超参 CV 化（计划§八 M1，2026-10-23 预登记）：cv_select 启用时替换
    # "单片末段验证集早停"——扩窗式时序分块 CV 选超参与等效轮数。缺省不进本分支，
    # 现行路径逐字不变（run 级证明=字节回归门：gate4_e5_wq4 --force --tag reg1 对
    # rev1 同名 json 除 meta.tag 外逐字段相等）。----
    cv_cfg = ml_cfg.get("cv_select") or {}
    if cv_cfg.get("enabled"):
        rdi_arr = np.asarray(row_day_idx) if row_day_idx is not None else np.array([])
        _cv_min_days = int(cv_cfg.get("min_days", 48))
        if len(rdi_arr) and len(np.unique(rdi_arr)) >= _cv_min_days:
            return _train_ml_model_cv(X, y, feature_names, row_day_idx, horizon,
                                      ml_cfg, cv_cfg, sel_diag)
        print(f"    [M1-cv] 采样日不足（{len(np.unique(rdi_arr))} < {_cv_min_days}），"
              f"该窗口回退现行末段验证集路径", flush=True)

    # 时序末段做早停验证集（不随机打乱——防未来函数泄漏）
    val_ratio = float(ml_cfg.get("val_ratio", 0.2))
    # 按采样日切分（非按行切）+ 训练/验证间留 horizon 个交易日的 gap：
    # 标签是未来 horizon 日收益，训练集末尾样本的标签覆盖 [t, t+horizon]，
    # 若验证集紧接训练集（无 gap），验证集特征时刻 t' 落在训练标签窗口内 → 标签信息泄漏 +
    # 时序自相关致早停乐观偏倚（[[ml-redundancy-tolerance-vs-linear]] 任务#2）。
    # row_day_idx 是每行的采样日序号 i（range(0,len-horizon,freq) 的 i），单调非降。
    X_tr, y_tr, val_set = X, y, None
    if len(X) > 100 and row_day_idx:
        rdi = np.asarray(row_day_idx)
        unique_days = np.unique(rdi)
        n_val_days = max(1, int(len(unique_days) * val_ratio))
        if n_val_days < len(unique_days):
            # 验证集 = 末 n_val_days 个采样日；训练集 = 其余日，中间留 gap（跳过 horizon 天内的采样日）
            val_days = set(unique_days[-n_val_days:].tolist())
            val_mask = np.isin(rdi, list(val_days))
            # gap：训练集排除与验证集最早日相距 < horizon 的采样日（标签重叠窗口）
            min_val_day = min(val_days)
            train_mask = (~val_mask) & (rdi <= min_val_day - horizon)
            if train_mask.sum() >= int(ml_cfg.get("min_samples", 500)) and val_mask.sum() >= 1:
                X_tr, y_tr = X[train_mask], y[train_mask]
                X_val, y_val = X[val_mask], y[val_mask]
                val_set = (X_val, y_val)  # 标记有验证集
            # 否则样本不足留 gap → 退回无验证集（不早停，用满 n_estimators）

    # 种子层（P1-2，2026-10-21）：ml.seeds=[s1,s2,...] 逐种子训练、预测取均值（_SeedEnsemble）。
    # 缺省或单元素 = 单种子路径，params 与历史行为逐字一致——全部历史 run 的可复现性不受影响。
    # （M1，2026-10-23：种子解析原样抽取至 _ml_seeds、超参构造原样抽取至 _ml_base_params，
    #  语义零改动——数值身份由字节回归门钉 run 复核。）
    seeds = _ml_seeds(ml_cfg)
    params = _ml_base_params(ml_cfg, seeds[0])
    n_estimators = int(ml_cfg.get("n_estimators", 300))
    early_rounds = int(ml_cfg.get("early_stopping_rounds", 20))

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    callbacks = []
    fit_kwargs = {}
    if val_set is not None and early_rounds > 0:
        # LightGBM 4.x valid_sets 需 Dataset 对象（非 (X,y) 元组）；只放验证集做早停指标
        val_dataset = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)
        fit_kwargs["valid_sets"] = [val_dataset]
        callbacks.append(lgb.early_stopping(early_rounds, verbose=False))

    # ---- M3 超参抖动集成（计划§八 M3，2026-10-26 预登记）：hp_ensemble 启用时把集成
    # 维度从"种子×N"扩展为"超参组×N_hp × 种子×N_seed"，每组沿用现行单片验证集早停取
    # best_iter（M1 的 CV 已裁，不引入），再定长训练；预测侧 _SeedEnsemble 成员均值
    # 天然覆盖。缺省不进本分支，现行路径逐字不变（纪律同 cv_select：先字节回归门）。
    # 位置在单种子早退之前——hp 集成对任意种子数（含单种子）生效。----
    hp_cfg = ml_cfg.get("hp_ensemble") or {}
    min_best_iter = int(ml_cfg.get("min_best_iter", 1))
    if hp_cfg.get("enabled"):
        hp_groups = hp_cfg.get("groups") or [
            {},  # 组 A = 生产默认（不覆盖任何超参）
            {"learning_rate": 0.03, "max_depth": 5, "num_leaves": 16},  # 组 B = 大容量（M1 的 CV 高频选择）
        ]
        return _train_ml_model_hp(X, y, feature_names, X_tr, y_tr, val_set,
                                  seeds, params, n_estimators, early_rounds,
                                  min_best_iter, hp_groups, ml_cfg, sel_diag)

    if len(seeds) == 1:
        model = lgb.train(params, train_set, num_boost_round=n_estimators,
                          callbacks=callbacks, **fit_kwargs)
        # 早停退化保护（任务#4）：best_iter<=1 表示首轮验证指标即恶化（模型对训练集的拟合在验证集
        # 立刻变差），是过拟合/标签噪声/验证集泄漏的退化信号。强行用这种模型预测 = 拿噪声当信号。
        # 退化阈值 min_best_iter 可配（默认 1）；退化则该窗口 ML 退化为空仓（返回 None），
        # 而非用 best_iter=1 的退化模型乱下单。无验证集（未早停）时 best_iteration=0，跳过此检查。
        if val_set is not None:
            best_iter = int(getattr(model, "best_iteration", 0) or 0)
            min_best_iter = int(ml_cfg.get("min_best_iter", 1))
            if best_iter <= min_best_iter:
                diag = {"status": "empty", "reason": "degraded", "best_iter": best_iter,
                        "min_best_iter": min_best_iter}
                if sel_diag:
                    diag["feature_selection"] = sel_diag
                return None, feature_names, diag  # 退化→空仓
            diag = {"status": "ok", "best_iter": best_iter, "n_estimators": n_estimators,
                    "has_val_set": True, "n_train": len(X_tr), "n_val": len(X_val)}
        else:
            # 无验证集（样本不足留 gap 或样本过小）：不早停、用满 n_estimators 轮，best_iteration=0。
            # 这是 [[ml-best-iter-degradation-blindspot]] 的盲区之二——看起来"没退化"，实则可能
            # 过拟合且未早停。diag 记录 has_val_set=False 供统计区分，不改变模型决策（仍返回模型）。
            diag = {"status": "ok", "best_iter": 0, "n_estimators": n_estimators,
                    "has_val_set": False, "n_train": len(X_tr), "n_val": 0}
        if sel_diag:
            diag["feature_selection"] = sel_diag
        return model, feature_names, diag

    # ---- 多种子分支（P1-2）：同 X/y/验证切分，逐种子训练，预测取均值 ----
    # 退化语义：单种子退化→整窗空仓；集成退化→剔除该种子成员（一次坏抽签不该连坐全部
    # 成员），全部成员退化才空仓（与单种子规则对齐）。每成员独立重建 Dataset/callbacks
    # （early_stopping 回调带跨训练状态，复用实例会泄漏停止轮）。
    # （M3：min_best_iter 解析与 hp 分支已上提至单种子早退前，语义零改动。）
    # （M3：min_best_iter 解析已上提至 hp 分支前，语义零改动。）
    models, kept_seeds, best_iter_by_seed, dropped_by_seed = [], [], {}, {}
    for s in seeds:
        p = dict(params)
        p["seed"] = s
        kw = {}
        if val_set is not None and early_rounds > 0:
            kw["valid_sets"] = [lgb.Dataset(X_val, label=y_val, feature_name=feature_names)]
            cb = [lgb.early_stopping(early_rounds, verbose=False)]
        else:
            cb = []
        m = lgb.train(p, train_set, num_boost_round=n_estimators, callbacks=cb, **kw)
        bi = int(getattr(m, "best_iteration", 0) or 0) if val_set is not None else 0
        if val_set is not None and bi <= min_best_iter:
            dropped_by_seed[s] = bi
            continue
        models.append(m)
        kept_seeds.append(s)
        best_iter_by_seed[s] = bi
    if not models:
        diag = {"status": "empty", "reason": "degraded", "min_best_iter": min_best_iter,
                "seeds": seeds, "best_iter_by_seed": dropped_by_seed}
        if sel_diag:
            diag["feature_selection"] = sel_diag
        return None, feature_names, diag  # 全部成员退化→空仓（同单种子规则）
    med_bi = int(np.median(list(best_iter_by_seed.values()))) if val_set is not None else 0
    diag = {"status": "ok", "best_iter": med_bi, "n_estimators": n_estimators,
            "has_val_set": val_set is not None,
            "n_train": len(X_tr), "n_val": (len(X_val) if val_set is not None else 0),
            "n_members": len(models), "seeds": kept_seeds,
            "best_iter_by_seed": best_iter_by_seed}
    if dropped_by_seed:
        diag["dropped_by_seed"] = dropped_by_seed
    if sel_diag:
        diag["feature_selection"] = sel_diag
    return _SeedEnsemble(models, kept_seeds), feature_names, diag


def _train_ml_model_hp(X, y, feature_names, X_tr, y_tr, val_set,
                       seeds, params, n_estimators, early_rounds,
                       min_best_iter, hp_groups, ml_cfg, sel_diag=None):
    """M3 超参抖动集成训练器（计划§八 M3，2026-10-26 预登记）。

    集成维度 = 超参组 × 种子（组 A 生产默认 + 组 B 大容量 × seeds [42,7,2026] = 6 成员，
    预测取均值）。每组沿用与现行多种子路径逐字相同的单片验证集早停（M1 的 CV 化已裁，
    不引入）；退化语义与 P1-2 对齐：单成员退化剔除不连坐，组内全部种子退化则剔整组，
    全组退化才空仓。实现上复用 _train_ml_model 已构造的 X_tr/y_tr/val_set/params——
    切分与基础超参与现行路径同源，组差异仅为 groups 里的覆盖字段。

    归因设计（M1 终裁输入）：对照观察项=组间 best_iter 与逐窗成员构成，供判读
    "近期增益来自大容量还是集成平滑"。
    """
    import lightgbm as lgb

    if val_set is not None:
        X_val, y_val = val_set
    models, kept_members, best_iter_by_member, dropped_by_member = [], [], {}, {}
    for gi, group in enumerate(hp_groups):
        p = dict(params)
        p.update(group)
        for s in seeds:
            ps = dict(p)
            ps["seed"] = s
            kw = {}
            if val_set is not None and early_rounds > 0:
                kw["valid_sets"] = [lgb.Dataset(X_val, label=y_val, feature_name=feature_names)]
                cb = [lgb.early_stopping(early_rounds, verbose=False)]
            else:
                cb = []
            m = lgb.train(ps, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names),
                          num_boost_round=n_estimators, callbacks=cb, **kw)
            bi = int(getattr(m, "best_iteration", 0) or 0) if val_set is not None else 0
            member = (gi, s)
            if val_set is not None and bi <= min_best_iter:
                dropped_by_member[f"g{gi}_s{s}"] = bi
                continue
            models.append(m)
            kept_members.append(member)
            best_iter_by_member[f"g{gi}_s{s}"] = bi
    if not models:
        diag = {"status": "empty", "reason": "degraded", "min_best_iter": min_best_iter,
                "seeds": seeds, "best_iter_by_member": dropped_by_member}
        if sel_diag:
            diag["feature_selection"] = sel_diag
        return None, feature_names, diag
    bi_vals = [v for v in best_iter_by_member.values()] if val_set is not None else [0]
    med_bi = int(np.median(bi_vals)) if val_set is not None else 0
    diag = {"status": "ok", "best_iter": med_bi, "n_estimators": n_estimators,
            "has_val_set": val_set is not None,
            "n_train": len(X_tr), "n_val": (len(X_val) if val_set is not None else 0),
            "n_members": len(models), "members": [f"g{g}_s{s}" for g, s in kept_members],
            "seeds": seeds, "hp_groups": [dict(g) for g in hp_groups],
            "best_iter_by_member": best_iter_by_member}
    if dropped_by_member:
        diag["dropped_by_member"] = dropped_by_member
    if sel_diag:
        diag["feature_selection"] = sel_diag
    return _SeedEnsemble(models, [f"g{g}_s{s}" for g, s in kept_members]), feature_names, diag


class _RidgePredictor:
    """ridge 档预测器（P1-3）：包 StandardScaler + RidgeCV，predict 先标准化再线性打分。

    duck-type 对齐 LightGBM Booster 的两个消费面：composite_score_ml 的 model.predict(X)
    （传入原始特征矩阵，标准化只在内部做——训练/预测口径同源）与 walk_forward 窗口
    日志/登记的 model.feature_importance("gain")（取 |系数| 作排序重要性，无 gain 概念）。
    """

    def __init__(self, ridge, scaler):
        self.ridge = ridge
        self.scaler = scaler

    def predict(self, X):
        return self.ridge.predict(self.scaler.transform(X))

    def feature_importance(self, importance_type="gain"):
        return np.abs(self.ridge.coef_)


def _train_ridge_model(train_price_dfs, factors, horizon, ml_cfg, series_caches=None,
                       pool_selector=None):
    """ridge 线性参照档训练器（P1-3，2026-10-21）。

    与 _train_ml_model 共用 _build_train_xy——特征矩阵/标签/NaN 填充口径逐字同源，
    差异仅在拟合器：StandardScaler 标准化 + RidgeCV（alpha 网格由 CV 内部选点，
    scoring=负 MSE）。无早停概念 → 用满全训练段（含 ML 早停所留的末段验证期；
    线性模型无逐轮过拟合问题，用满样本是参照档惯例，与 ML 早停口径的差异属
    档间真实差异、不强行对齐）。sklearn RidgeCV 确定性（无随机源），复现性约束同 ML 档。

    返回 (model, feature_names, diag)；model=None 表示样本不足（空仓窗口，同 ML 语义）。
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    X, y, feature_names, _, _empty_diag = _build_train_xy(
        train_price_dfs, factors, horizon, ml_cfg, series_caches,
        pool_selector=pool_selector)
    if _empty_diag is not None:
        return None, feature_names, _empty_diag  # 样本不足，该窗口退化（同 ML 语义）

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    _alphas_cfg = ml_cfg.get("ridge_alphas")
    alphas = (np.asarray(_alphas_cfg, dtype=float) if _alphas_cfg is not None
              else np.logspace(-3, 3, 13))
    model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
    model.fit(Xs, y)
    diag = {"status": "ok", "alpha": float(model.alpha_), "n_train": len(X),
            "n_features": len(feature_names), "has_val_set": False}
    return _RidgePredictor(model, scaler), feature_names, diag


def run_walk_forward(cfg: dict, universe: str | None = None,
                     train_days: int = 252, test_days: int = 63,
                     step_days: int = 63, horizon: int = 10,
                     verbose: bool = True, attribution: bool = False,
                     read_only: bool = False, collect_exit_pnls: bool = False,
                     collect_exit_events: bool = False,
                     skip_missing: bool = False,
                     valid_min: dict[str, pd.DataFrame] | None = None,
                     persist_nav: str | os.PathLike | None = None) -> dict:
    """
    运行滚动样本外回测。

    train_days: 训练段天数（约1年）
    test_days: 测试段天数（约3个月）
    step_days: 滚动步长
    horizon: IC 计算的未来收益持有期
    attribution: 若 True，旁路记录归因原料（每日各持仓股收益 + 每个调仓日各因子截面打分），
                 存入返回值的 attribution_log，供 attribution.py 做截面回归归因。默认关闭不影响性能
    read_only: True = 数据获取走 cache-only 模式（命中即读、不拉网不写、跳过 migrate）。
        用于场景级进程并行：主进程预热后 worker 设此位，杜绝并发写 parquet 竞态。
        默认 False=旧行为（单场景跑/attribution/verify 不受影响）。见 plans 点3。
    valid_min: 分钟级数据 {symbol: df}，供执行层 open_strength 过滤用。
        None=不启用第二因子过滤（默认，保回归门逐字一致）。
        仅当调用方确认 intraday_filter.enabled 且 open_strength.enabled 时传入；
        本函数内部不再读 config 决定是否拉分钟数据（避免默认路径拉网开销），完全由调用方门控。
    persist_nav: 若非 None，把样本外逐日 nav + 基准 nav 落盘到该 parquet 路径
        （列: date/nav/bench_nav，升序）。用于回撤控制叠加层（CPPI/vol-target）实验——
        需要主策略真实 nav 序列做底，在它之上仿真叠加层而非重跑回测。默认 None=不落盘，
        行为与改动前逐字一致，所有现有调用零影响。
    valid_min: 分钟级数据 {symbol: df}，供执行层 open_strength 过滤用。
        None=不启用第二因子过滤（默认，保回归门逐字一致）。
        仅当调用方确认 intraday_filter.enabled 且 open_strength.enabled 时传入；
        本函数内部不再读 config 决定是否拉分钟数据（避免默认路径拉网开销），完全由调用方门控。
    """
    mf = cfg["multifactor"]
    universe = universe or mf["universe"]
    bt = mf["backtest"]
    wf_cfg = mf.get("walk_forward", {})
    # 提前解析 weighting（原在 line ~386 解析；提前到此处供 ML 宽训练池分支判断），
    # line ~386 处删除重复赋值，保持单一来源。
    weighting = wf_cfg.get("weighting", "static")

    if verbose:
        print(f"获取 {universe} 成分股 ...")
    syms = get_index_constituents(universe, read_only=read_only)
    bench_name = mf.get("benchmark", "csi300")
    bench_syms = get_index_constituents(bench_name, read_only=read_only)

    # ML/ridge 宽训练池隔离：weighting∈{ml,ridge} 且配置 ml_train_universe 时，训练从更广的
    # 可投资池取样本（如 csi300∪500∪1000≈1800 只），但选股仍在 universe（csi1000）内——
    # 回答"宽训练池是否让可投资选股更好"。linear 路径（static/ic）永不进此分支，逐字不变。
    # ridge 档同进此分支（P1-3）：参照档特征矩阵须与 ML 档完全同源（同池同特征）。
    # select_syms = 选股集合（恒为 universe 成分股）；ml_train_pool 仅训练读，不进选股/风控/执行。
    select_syms = set(syms)
    ml_train_pool = None
    if weighting in ("ml", "ridge") and wf_cfg.get("ml_train_universe"):
        pool_universes = wf_cfg["ml_train_universe"]
        if isinstance(pool_universes, str):
            pool_universes = [pool_universes]
        pool_universes = [u for u in pool_universes if u != universe]
        if pool_universes:
            extra_pool = []
            for u in pool_universes:
                extra_pool += get_index_constituents(u, read_only=read_only)
            ml_train_pool = set(extra_pool)
            if verbose:
                print(f"[ML 宽训练池] 额外训练池 {pool_universes} → 新增 {len(ml_train_pool - select_syms)} 只（选股仍限 {universe}）")

    # ---- M6 逐日动态训练池（计划§八 M6，2026-10-30 预登记）----
    # 与上面的 ml_train_universe 不同：那个是**静态**指数成分并集，这个是**逐采样日**按
    # 当日可观测市值/流动性重算的标的集合（见 factors/dynamic_pool.py）。二者可叠加，
    # 但 M6 场景只用动态池。缺省（无 ml_dynamic_pool 块或 enabled=false）= 现行路径逐字不变。
    # 标的清单取"缓存里已有的全部日线标的"——严格 cache-only 下这是唯一可靠的全市场口径，
    # 指数接口返回的是当前成分，与 2018 年的历史截面不一致。
    pool_cfg = (wf_cfg.get("ml_dynamic_pool") or {})
    dynamic_pool_enabled = bool(pool_cfg.get("enabled")) and weighting in ("ml", "ridge")
    pool_universe_syms: list[str] = []
    if dynamic_pool_enabled:
        from jq_framework.data.fetchers.fetcher import list_cached_symbols
        train_start_key = (pd.Timestamp(bt["start"])
                           - pd.Timedelta(days=int(train_days * 1.6))).strftime("%Y-%m-%d")
        pool_universe_syms = list_cached_symbols(start=train_start_key)
        if not pool_universe_syms:
            raise ValueError(
                f"[M6] 动态训练池已启用，但缓存键 stock_*_qfq_{train_start_key}.parquet "
                f"下没有任何标的；请先补齐该键（scripts/backfill_all_a_daily.py "
                f"--start {train_start_key}）或检查 data.cache_dir")
        ml_train_pool = set(pool_universe_syms)
        if verbose:
            print(f"[M6 动态训练池] 候选全市场 {len(pool_universe_syms)} 只，逐日筛选 "
                  f"top_n={pool_cfg.get('top_n')}（选股仍限 {universe}）")

    # ---- M7 扩充可投资域（计划§八 M7，2026-11-02 预登记）----
    # 与上面两个都不同：M6 只换训练域、选股仍限 universe；ml_train_universe 是静态并集。
    # M7 把**选股域、训练域、基准**三处统一到逐日构造的
    #     U(K, t) = csi1000 在册成员(t) ∪ 过门槛的市值最小 K 只非成员(t)
    # 命题：把可投资域本身向下扩到更小市值，在这份扩充了的面板上训练并选股，是否更好。
    # 缺省（无 universe_expansion 块或 enabled=false）= 现行路径逐字不变。
    # 候选标的同样取"缓存里已有的全部日线标的"：严格 cache-only 下这是唯一可靠的全市场
    # 口径，指数接口返回的是当前成分，与历史截面不一致。
    exp_cfg = wf_cfg.get("universe_expansion") or {}
    expansion_enabled = bool(exp_cfg.get("enabled")) and weighting in ("ml", "ridge")
    expansion_syms: list[str] = []
    universe_panel = {}
    if expansion_enabled:
        if dynamic_pool_enabled:
            raise ValueError("universe_expansion 与 ml_dynamic_pool 不可同时启用："
                             "前者把选股/训练/基准统一到扩充池，后者只改训练域，"
                             "同时开启会使二者的口径混淆、读数无法归因")
        from jq_framework.data.fetchers.fetcher import list_cached_symbols
        from jq_framework.factors.universe_expansion import parse_expansion_cfg
        if parse_expansion_cfg(exp_cfg) is None:
            raise ValueError(f"universe_expansion 配置非法（top_n 须为正整数）：{exp_cfg}")
        train_start_key = (pd.Timestamp(bt["start"])
                           - pd.Timedelta(days=int(train_days * 1.6))).strftime("%Y-%m-%d")
        expansion_syms = list_cached_symbols(start=train_start_key)
        if not expansion_syms:
            raise ValueError(
                f"[M7] 扩充可投资域已启用，但缓存键 stock_*_qfq_{train_start_key}.parquet "
                f"下没有任何标的；请先补齐该键或检查 data.cache_dir")
        # 选股/训练/基准共用这份全市场候选；训练域额外并入全部候选（池在采样日再收窄）
        ml_train_pool = set(expansion_syms)
        if verbose:
            print(f"[M7 扩充可投资域] 候选全市场 {len(expansion_syms)} 只，"
                  f"U = {universe} ∪ 市值最小 {exp_cfg.get('top_n')} 只非成员（逐日构造）")

    if verbose:
        print(f"拉取 {len(set(syms + bench_syms))} 只股票日线 ...")
    # 2026-10-12 修复（§10 残余泄漏，哈希探针实证）：set→list 迭代序对字符串符号依赖
    # PYTHONHASHSEED（逐进程随机）→ all_syms 顺序逐进程不同 → get_batch_daily 的键序
    # 修复只保证"按输入序返回"而输入序本身随机 → sym_data/feat_rows/X 行序逐进程不同
    # → LightGBM 浮点梯度累积序变 → 平坦验证曲线处早停翻转（X md5 窗窗不等的根因）。
    # sorted() 固定行序；与历史 run 数值路径不可比（等效换一次抽签，同此前两次修复）。
    all_syms = sorted(set(syms + bench_syms))
    if ml_train_pool:
        all_syms = sorted(set(all_syms) | ml_train_pool)  # 宽池只加载进 valid 供 ML 训练读
    # 训练段需要更早的数据，往前多拉 train_days 天
    start_dt = pd.Timestamp(bt["start"]) - pd.Timedelta(days=int(train_days * 1.6))
    price_dfs = get_batch_daily(all_syms, start=start_dt.strftime("%Y-%m-%d"),
                                progress=verbose, read_only=read_only, skip_missing=skip_missing)

    valid = {s: df for s, df in price_dfs.items() if not df.empty}
    all_dates = sorted(set().union(*[set(df["date"]) for df in valid.values()]))
    dates = [d for d in all_dates if start_dt <= d <= pd.Timestamp(bt["end"])]
    if verbose:
        print(f"总区间: {dates[0].date()} -> {dates[-1].date()}, 共 {len(dates)} 个交易日")

    # 日内因子（open_strength / amplitude）的分钟数据：任一启用才构建（默认关闭传 None，不拉分钟数据）
    # valid_min=None 且有日内因子启用时，本进程用 read_only 口径构建（worker 走
    # cache-only 只读已预热的 parquet，无网络/写；主进程跑则按 read_only=False 旧行为）。
    # 不跨进程序列化 valid_min：每进程本地构建，省 pickle 1000 DataFrame 的开销。
    if valid_min is None:
        _ifilt = mf.get("intraday_filter", {})
        _min_on = _ifilt.get("enabled", False) and (
            _ifilt.get("open_strength", {}).get("enabled", False)
            or _ifilt.get("amplitude", {}).get("enabled", False)
        )
        if _min_on:
            from jq_framework.data.fetchers.fetcher_minute import get_batch_minute
            if verbose:
                print(f"拉取 {len(all_syms)} 只股票分钟数据（日内执行层过滤） ...")
            min_dfs = get_batch_minute(all_syms,
                                       freq=cfg.get("data", {}).get("minute_freq", "5"),
                                       adjust=cfg.get("data", {}).get("minute_adjust", "qfq"),
                                       progress=verbose, read_only=read_only)
            valid_min = {s: df for s, df in min_dfs.items() if not df.empty}
            if verbose:
                print(f"  分钟数据就绪: {len(valid_min)}/{len(all_syms)} 只")

    # 加载因子
    factors = [(load_factor(spec), spec.get("weight", 1.0))
               for spec in mf["factors"].values()]

    # ML/ridge 增强特征注入（业界范式：算子库生成海量特征喂 ML，Qlib Alpha158 路径1）：
    # 仅 weighting∈{ml,ridge} 且显式配置 augmented_factors 时，额外挂载注入源（如 alpha158 的
    # 158 个算子特征）进训练特征池。不写进生产 multifactor.factors（避免污染 weighting=static
    # 线性路径——静态下注入源会白算/非预期影响合成），故走 walk_forward.ml 层 opt-in 注入。
    # ridge 档同进（P1-3）：参照档特征池须与 ML 档完全同源。
    # 注入源 spec 复用 load_factor 契约（module/class/params），augmented_factors 为 spec dict
    # （key=特征名基，每个展开为对应实例），或 [module] 简写（用默认 params 全 158 特征）。
    # 手动标志仅记录不保护：注入特征与生产 factors 一视同仁进统一筛选。
    ml_cfg = wf_cfg.get("ml", {}) or {}
    aug = ml_cfg.get("augmented_factors") or {}
    if weighting in ("ml", "ridge") and aug:
        aug_specs = aug if isinstance(aug, dict) else {"alpha158": {"module": "factors.alpha158"}}
        for base, spec in aug_specs.items():
            spec = spec if isinstance(spec, dict) else {}
            module = spec.get("module", base)
            from importlib import import_module
            mod = import_module(module)
            # 默认展开全特征；显式 params.feature 只挂一个（模块按 feature 参数化）
            feats = getattr(mod, "FEATURE_NAMES", [base] if base else [])
            for fn in feats:
                fspec = {"module": module, "class": spec.get("class", "Alpha158Factor"),
                         "params": {**(spec.get("params", {})), "feature": fn},
                         "weight": spec.get("weight", 1.0),
                         "manual": False}
                factors.append((load_factor(fspec), fspec["weight"]))

    # 行业中性化注入（补齐：walk-forward 原本缺失，与 portfolio_runner 对齐）
    industry_neutral = mf.get("industry_neutral", False)
    if industry_neutral:
        if verbose:
            print("拉取申万一级行业映射（用于行业中性化）...")
        try:
            industry_map = get_stock_industry_map(read_only=read_only)
            for factor, _ in factors:
                factor.industry_neutral = True
                factor.set_industry_map(industry_map)
        except Exception as e:
            if verbose:
                print(f"  行业映射拉取失败，跳过中性化: {e}")

    # 基本面因子数据注入（若有任何 is_fundamental 因子）
    has_fund = any(getattr(f, "is_fundamental", False) for f, _ in factors)
    if has_fund:
        if verbose:
            print("拉取基本面数据（估值+财务）注入基本面因子 ...")
        try:
            fund_data = build_fundamental_data(all_syms, progress=verbose, read_only=read_only)
            for factor, _ in factors:
                factor.set_fundamental_data(fund_data)
            if verbose:
                ok = sum(1 for v in fund_data.values() if v is not None and not v.empty)
                print(f"  基本面数据: {ok}/{len(all_syms)} 只股票就绪")
        except Exception as e:
            if verbose:
                print(f"  基本面数据拉取失败，基本面因子将返回 None: {e}")

    # 风控
    risk_cfg = mf.get("risk", {})
    risk = RiskManager(
        stop_loss=risk_cfg.get("stop_loss", 0.10),
        take_profit=risk_cfg.get("take_profit", 0.30),
        target_vol=risk_cfg.get("target_vol", 0.15),
        lookback=risk_cfg.get("lookback", 20),
        max_leverage=risk_cfg.get("max_leverage", 1.5),
        stop_loss_mode=risk_cfg.get("stop_loss_mode", "fixed"),
        stop_vol_k=risk_cfg.get("stop_vol_k", 1.0),
        stop_vol_min_days=risk_cfg.get("stop_vol_min_days", 10),
        trail_pct=risk_cfg.get("trail_pct"),
        trail_activate=risk_cfg.get("trail_activate"),
        sl_trail_pct=risk_cfg.get("sl_trail_pct"),
        sl_trail_activate=risk_cfg.get("sl_trail_activate"),
    )

    # 大盘择时（RSRS 仓位乘数，默认关闭→no-op 恒 1.0 不破回归门）
    market_timer = MarketTimer(mf.get("market_timing", {}))

    # CPPI 仓位层缩放（默认关闭→no-op 恒 1.0 不破回归门）。
    # 与 RSRS 同属总仓位层但语义相反：RSRS 预计算查表（依赖外生指数）只在调仓日缩权重；
    # CPPI 逐日 update(nav) 实时算（依赖内生 nav）在 port_ret 层施加，非调仓日也缩仓。
    # 状态(hwm/v0)跨窗口延续——本实例在 run_walk_forward 内创建一次，nav 跨窗口累积不重置，
    # CPPI 的 hwm 也跨窗口延续（否则每窗口 hwm 重置破坏 HWM 锁语义）。
    cppi_timer = CPPITimer(mf.get("cppi", {}))

    # 组合层波动率目标缩放（方案B，默认关闭→no-op 恒 1.0 不破回归门）。
    # 与 CPPI 同属总仓位层 port_ret 缩放，但机制相反：CPPI 看净值水平（cushion）反应性保底，
    # vol-target 看已实现波动率（σ_target/σ_realized）预防性控风险水平。两者可串联取 min。
    # 状态(收益序列 deque)跨窗口延续，与 cppi_timer 同实例生命周期。
    vol_target_timer = VolTargetTimer(mf.get("vol_target", {}))

    # 成本
    cost_cfg = bt.get("cost", {})
    cost_buy = cost_cfg.get("commission", 0.00025) + cost_cfg.get("slippage", 0.001)
    cost_sell = (cost_cfg.get("commission", 0.00025) + cost_cfg.get("slippage", 0.001)
                 + cost_cfg.get("stamp_duty", 0.0005))

    # 基准：M7 下必须等 U(t) 面板构造完成后才能算（见下方 M7 段之后的赋值）。
    # 非 M7 路径在此立即计算，逐字不变。
    if not expansion_enabled:
        bench_df = get_index_daily(bench_syms, {s: valid.get(s, pd.DataFrame()) for s in bench_syms},
                                   start=dates[0].strftime("%Y-%m-%d"))
        bench_ret = (bench_df.set_index("date")["nav"].pct_change()
                     if not bench_df.empty else pd.Series(dtype=float)).fillna(0.0)

    # 大盘择时乘数预算：用真实指数日线（需 high/low，合成基准不可用）一次性算好全部日期
    # 的 RSRS 仓位乘数 → 查表字典。默认关闭时 no-op 不拉数据。start 早于回测起点以保证
    # z-score 标准化窗口 M=600 有足够历史预热（避免回测前期乘数恒满仓）。
    # 起点取 max(回测起点前推 M×2 日历日, 2009-01-01)：M≈600交易日≈2.4年，前推 M×2 日历日
    # 留足 RSRS 的 N+M 滚动回归预热余量；2009 为 csi300 真实指数可拉最早（再早无 high/low）。
    # 旧场景(回测起点≥2015)时前推后仍早于2015，行为不变；长历史(2012起)时不再因 hardcoded 2015
    # 导致 2012-2014 段预热不足→查表 miss→恒满仓的"假择时"。
    if market_timer.enabled:
        warmup_days = market_timer.m * 2
        idx_start = (dates[0] - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        idx_start = max(idx_start, "2009-01-01")
        idx_df = get_index_real_daily(bench_name, start=idx_start,
                                      end=dates[-1].strftime("%Y-%m-%d"), read_only=read_only)
        market_timer.build(idx_df)
        if verbose:
            cov = market_timer.coverage()
            print(f"  RSRS 大盘择时: 覆盖{cov[0]}天 均仓{cov[1]:.2f} 满仓{cov[3]*100:.0f}%")

    windows = _segment_dates(dates, train_days, test_days, step_days)
    if verbose:
        print(f"切分: 训练{train_days}天 + 测试{test_days}天, 步长{step_days}天 → {len(windows)} 个窗口")

    top_n = mf["select"]["top_n"]
    # 换手缓冲：优先取分池覆盖（buffer_by_universe[当前universe]，A/B 结论 csi500→8/csi1000→16，
    # 见 [[turnover-buffer-buf8-breakeven]]），否则回退全局 turnover_buffer。默认未配=旧行为零回归。
    _buf_override = mf["select"].get("buffer_by_universe") or {}
    buffer = _buf_override.get(universe, mf["select"].get("turnover_buffer", 0))
    if verbose:
        print(f"  换手缓冲     : buffer={buffer} (universe={universe}, 全局turnover_buffer="
              f"{mf['select'].get('turnover_buffer', 0)}, 分池覆盖 {_buf_override})")
    min_ic_abs = wf_cfg.get("min_ic_abs", 0.01)
    ic_freq = wf_cfg.get("ic_freq", 5)
    clamp = (wf_cfg.get("ic_clamp_floor", 0.25), wf_cfg.get("ic_clamp_cap", 1.5))
    # 调仓频率：优先取分池覆盖（rebalance_by_universe[当前universe]，A/B 结论 csi500→21/csi1000→42，
    # 见 [[rebalance-frequency-csi1000-breakthrough]]/[[rebalance-frequency-cross-pool-csi500]]），
    # 否则回退全局 rebalance_days。默认未配=旧行为零回归。
    _rebal_override = mf["select"].get("rebalance_by_universe") or {}
    rebalance_days = max(1, int(_rebal_override.get(
        universe, mf["select"].get("rebalance_days", 1))))
    if verbose:
        print(f"  调仓频率     : rebalance_days={rebalance_days} (universe={universe}, 全局="
              f"{mf['select'].get('rebalance_days', 1)}, 分池覆盖 {_rebal_override})")
    # 两层合成配置（None/flat=逐字复现旧 flat 逻辑；twolevel=族内正交+族间风险预算）
    # neutralize_size: opt-in size 残差化（剥离最终 score 对 size 的纯线性暴露），默认 false 零回归
    combine_cfg = {k: mf.get(k) for k in ("combine", "families", "orthogonalize", "neutralize_size")}

    # 簇B时序分层：layers.enabled 缺省/false → (False, None)，以下 legacy 单层路径逐字不变。
    # 分层时按因子 layer 标注分组、每层独立决定参数（见 _resolve_layers）；各层仍按 family 两level
    # 合成（combine/families/orthogonalize 不动，仅喂各自因子子集）。weighting:"ml"/"ridge" 与分层
    # 冲突——ML/ridge 合成走"全因子统一特征矩阵→单模型打分"，与分层"各层独立选股聚合"是两种
    # 互斥的合成架构，训练接口也只认全 factor 列表，故显式拒绝（分层实验用 static）。
    layers_enabled, layer_specs = _resolve_layers(mf, universe)
    if layers_enabled and weighting in ("ml", "ridge"):
        raise ValueError("簇B时序分层(layers.enabled=true) 与 weighting=ml/ridge 冲突：分层实验用 static，"
                         "ML/ridge 单模型合成已替代分层轴，拒绝组合避免误用")

    # ML/ridge 训练侧因子序列缓存：一次性从全历史 valid 预算，跨所有窗口复用。
    # 每窗口 _train_ml_model/_train_ridge_model 的 train_dfs 是"该窗口前缀"（新 df 对象/新首尾
    # 日期），窗口内预算会让 Alpha158 每窗口重算全 158 列 + 内容缓存跨窗口无限累积
    # （csi500 单窗~数分钟、RSS→4.5GB+）。calc_cross_section 按日期索引缓存、只读训练前缀内日期
    # → 全历史缓存结果逐字相同（多出未来历史从不被索引）。仅 weighting∈{ml,ridge} 预算
    # （linear 路径零影响）；未实现 calc_series 的因子缓存为空 dict，calc_cross_section 自动回退
    # 逐点，口径不变。
    ml_train_caches = None
    if weighting in ("ml", "ridge"):
        ml_train_caches = _build_series_caches(valid, factors)
        if verbose:
            print(f"[{weighting.upper()}] 全历史训练序列缓存预算完成（{len(ml_train_caches)} 因子, 跨窗口复用）")

    # ---- M6：构造逐日池选择器（缺省 disabled → None，_build_train_xy 走现行路径）----
    # 市值/流动性数据一次性预算后跨窗口复用：市值读 fin_value_em 缓存（严格 cache-only 下
    # 只读快照），成交额从 valid 的日线现算。开销实测可忽略（全 A 市值文件 5218 个约 60s）。
    pool_selector = None
    if dynamic_pool_enabled:
        from jq_framework.factors.dynamic_pool import build_pool_selector
        from jq_framework.data.fetchers.fetcher_fundamental import get_valuation
        if verbose:
            print(f"[M6 动态训练池] 载入 {len(pool_universe_syms)} 只的市值序列 ...")
        market_caps = {}
        for _s in pool_universe_syms:
            try:
                _v = get_valuation(_s, read_only=read_only)
                if _v is not None and not _v.empty:
                    market_caps[_s] = _v
            except Exception:
                continue
        if verbose:
            print(f"[M6 动态训练池] 市值可用 {len(market_caps)}/{len(pool_universe_syms)} 只")
        pool_selector = build_pool_selector(market_caps, valid, pool_cfg,
                                            fallback_symbols=syms)
        # 池选择器不再需要原始市值帧，释放内存（全 A 市值帧约 1GB）
        del market_caps

    # ---- M7：构造逐日扩充池 U(K, t)（缺省 disabled → 空，走现行路径逐字不变）----
    # 与 M6 的区别：M6 的选择器接收 candidates 再返回子集（训练侧用），M7 直接预算出
    # {date: set} 面板，因为**选股域与基准**也要按同一份 U(t) 变化，不能只在训练侧生效。
    # 预算是 dates 全体（含训练段），因为训练采样日也落在 dates 上；基准只在测试段取用。
    if expansion_enabled:
        from jq_framework.factors.universe_expansion import build_universe_panel, parse_expansion_cfg
        from jq_framework.data.fetchers.fetcher_fundamental import get_valuation
        if verbose:
            print(f"[M7 扩充可投资域] 载入 {len(expansion_syms)} 只的市值序列 ...")
        market_caps = {}
        for _s in expansion_syms:
            try:
                _v = get_valuation(_s, read_only=read_only)
                if _v is not None and not _v.empty:
                    market_caps[_s] = _v
            except Exception:
                continue
        if verbose:
            print(f"[M7 扩充可投资域] 市值可用 {len(market_caps)}/{len(expansion_syms)} 只，"
                  f"逐日构造 U ...")
        # 候选日线取 valid（= 已加载的全市场日线）；成本门槛与市值门槛都在面板内施加
        valid_for_panel = {s: valid[s] for s in expansion_syms if s in valid}
        universe_panel = build_universe_panel(
            market_caps, valid_for_panel, select_syms, dates, exp_cfg)
        del market_caps
        if verbose:
            sizes = [len(v) for v in universe_panel.values() if v]
            if sizes:
                print(f"[M7 扩充可投资域] U 规模 {min(sizes)} ~ {max(sizes)} 只"
                      f"（中位 {int(np.median(sizes))}）；选股/训练/基准同源")
            else:
                print("[M7 扩充可投资域][WARN] U 全为空集，请检查市值数据与日期范围")

        # 训练域复用 M6 的 pool_selector 接口（_build_train_xy 内按采样日调用）。
        # 与 M6 的区别：M6 的选择器做"市值最小 K 只"的**替换**，M7 返回的是**并集** U(t)
        # ——含全部在册成员 + 过门槛的新增。二者共用接口是为了让池过滤在截面打分之前生效
        # （预登记第七节(b)：rank 标准化的标定基准 = 训练池本身）。
        # 缺失日期（如采样日不在 dates 上）返回 None = 不施加过滤，与现有退化语义一致。
        def _m7_selector(_date, _candidates, _panel=universe_panel):
            return _panel.get(pd.Timestamp(_date))
        pool_selector = _m7_selector

        # 基准：当日 U(t) 的等权组合，成员**逐日变化**。不得退化成静态符号表——
        # 用"全部历史出现过的 U 成员并集"会把后期才入池的标的提前注入早期基准，
        # 等价于给基准塞入它本不该有的成分（预登记§七）。
        # 位置必须在 U 面板构造完成之后：首版曾把它放在前面，读到空 dict，基准全 0。
        from jq_framework.factors.universe_expansion import panel_to_returns
        bench_ret = panel_to_returns(universe_panel, valid, dates[0])
        if verbose:
            nz = int((bench_ret != 0).sum()) if len(bench_ret) else 0
            print(f"[M7 扩充可投资域] 基准 = U(t) 等权，覆盖 {len(bench_ret)} 个交易日"
                  f"（非零收益 {nz} 天）")
        pool_selector = _m7_selector

    nav = 1.0
    bench_nav = 1.0
    nav_series = []
    bench_series = []
    window_reports = []
    holdings: dict[str, dict] = {}
    pending_trade: list[str] | None = None
    days_since_rebal = 0  # 距上次重算信号的天数，跨窗口延续（rebalance_days 控制换仓频率）
    # 簇B分层：layers.enabled=false 时此 dict 恒空，零副作用；enabled 时每层一套状态
    # （lstate[name] = {holdings, pending_trade, days_since_rebal, directed, series_cache}）。
    # 状态跨窗口延续（同 legacy），directed/series_cache 每窗口由 Edit-3 块刷新。聚合走
    # A 分层子组合 w_s=Σ_l δ(l,s)(1/n_layers)/N_l，见测试段分层分支。
    if layers_enabled:
        lstate: dict[str, dict] = {ln: {"holdings": {}, "pending_trade": None,
                                        "days_since_rebal": 0,
                                        "directed": [], "series_cache": None}
                                   for ln in layer_specs}
    else:
        lstate = {}
    total_cost = 0.0
    turnovers = []  # 每次调仓的单边换手率采集（compute_turnover），供 avg_turnover 报告
    stop_count = {"stop_loss": 0, "take_profit": 0}
    # 实现 PnL 采集（attribution=True 时填充，默认关→零回归）：逐笔记录每笔卖出的实现盈亏
    # （open/entry - 1），区分主动止盈/止损 vs 被动换出。供卖出归因算胜率/盈亏比。
    exit_pnls = {"stop_loss": [], "take_profit": [], "passive": []}
    # 卖出事件明细（collect_exit_events=True 时填充，默认关→零回归）：逐笔记录已了结
    # (symbol, sell_date, reason, sell_price, pnl)，供"止盈/止损决策后验"用后市价判断对错。
    exit_events = []
    rejected_buys = 0   # 涨停开盘锁死被跳过的买入（诊断，验证清洗 delta 后可删）
    rejected_sells = 0  # 跌停开盘锁死未能卖出、保留持仓次日重试（诊断）
    vol_filtered = 0     # 放量执行层过滤净剔除数（诊断）
    vol_backfilled = 0   # 放量过滤兜底补回数（诊断）
    os_filtered = 0      # 开盘强度执行层过滤净剔除数（诊断，第二日内因子）
    os_backfilled = 0    # 开盘强度过滤兜底补回数（诊断）
    amp_filtered = 0     # 振幅执行层过滤净剔除数（诊断，第三日内因子，IC=-0.108 显著为负）
    amp_backfilled = 0   # 振幅过滤兜底补回数（诊断）
    direction_log = []  # 记录每窗口各因子方向，供分析
    # 归因原料（attribution=True 时填充）：每日各持仓股收益 + 每个调仓日各因子截面打分
    attribution_log = {"daily_returns": [], "selection_scores": []}

    def get_open(sym, date):
        df = valid.get(sym)
        if df is None:
            return None
        row = df[df["date"] == date]
        return row["open"].iloc[0] if not row.empty else None

    for wi, (train_dates, test_dates) in enumerate(windows):
        # --- 训练：在训练段算各因子方向（linear）/ 训练 ML 模型（ml）---
        train_end = train_dates[-1]
        # 逐窗基准起点快照（防重复原则1：逐窗落盘）。训练段不推进 bench_nav，
        # 窗口首日的累计基准净值即上一窗末值——循环顶捕获一次，下方三个
        # window_reports 追加点共用（含 all_factors_insignificant 空仓窗）。
        w_bench_start = bench_nav
        train_dfs = {s: df[df["date"] <= train_end] for s, df in valid.items()
                     if (df["date"] <= train_end).any()}
        ml_model = None  # ML/ridge 模式下承载该窗口训练好的模型；linear 路径恒 None
        feature_names = None
        if weighting in ("ml", "ridge"):
            ml_cfg = wf_cfg.get("ml", {})
            # P1-3：ridge 参照档与 ML 档同源特征矩阵（_build_train_xy），仅换拟合器
            _trainer = _train_ridge_model if weighting == "ridge" else _train_ml_model
            ml_model, feature_names, _ml_diag = _trainer(train_dfs, factors, horizon, ml_cfg,
                                                         series_caches=ml_train_caches,
                                                         pool_selector=pool_selector)
            # ML/ridge 模式：directed 退化为纯因子列表（weight 不参与打分，仅供归因 hook 遍历 factor.name）
            directed = list(factors)
            # 模型档诊断日志（direction_log 的 sign/ic_mean/t/icir 在 ML/ridge 模式无意义，记 importance 代替）。
            # [[ml-best-iter-degradation-blindspot]] 修复：direction_log 记录每窗口的 ml_status 与
            # 细粒度 diag（empty/degraded/ok + has_val_set/best_iter），空仓与无验证集窗口全部可见，
            # 不再被压成单一 insufficient_samples 或隐藏成 best_iter=0。
            entry = {"window": wi, "train_end": train_end, "ml_status": _ml_diag["status"],
                     "ml_reason": _ml_diag.get("reason"),
                     "ml_best_iter": _ml_diag.get("best_iter"),
                     "ml_has_val_set": _ml_diag.get("has_val_set"),
                     "ml_n_train": _ml_diag.get("n_train"),
                     # 特征池构成（手动标志仅记录不保护）：生产 factors = manual，注入增强 = aug
                     "ml_n_manual": sum(1 for f, _ in directed
                                        if not str(getattr(f, "name", "")).startswith("alpha158")),
                     "ml_n_aug": sum(1 for f, _ in directed
                                     if str(getattr(f, "name", "")).startswith("alpha158"))}
            if ml_model is not None:
                entry["ml_feature_importance"] = dict(
                    zip(feature_names, ml_model.feature_importance(importance_type="gain").tolist()))
            if weighting == "ridge":
                entry["ridge_alpha"] = _ml_diag.get("alpha")  # CV 选中的正则强度（P1-3 诊断）
            direction_log.append(entry)
        else:
            trained = _train_factor_directions(train_dfs, factors, horizon, ic_freq, min_ic_abs,
                                               weighting=weighting, clamp=clamp)
            # 构建带方向的因子列表（剔除 sign==0 的）。ic 模式下 w 已含 sign*幅度
            directed = [(f, w) for f, w, s, _, _, _ in trained if s != 0]
            dirs = {f.name: {"sign": s, "ic_mean": im, "t": t, "icir": iir}
                    for f, w, s, im, t, iir in trained}
            direction_log.append({"window": wi, "train_end": train_end, **{k: v for k, v in dirs.items()}})

        if not directed or (weighting in ("ml", "ridge") and ml_model is None):
            # 所有因子都不显著（linear）/ 样本不足（ml/ridge）：该测试段空仓
            for date in test_dates:
                br = bench_ret.get(date, 0.0)
                if br is None or (isinstance(br, float) and np.isnan(br)):
                    br = 0.0
                bench_nav *= (1 + br)
                nav_series.append({"date": date, "nav": nav})
                bench_series.append({"date": date, "nav": bench_nav})
            window_reports.append({"window": wi, "train_end": train_end,
                                   "window_start": str(test_dates[0].date()),
                                   "window_end": str(test_dates[-1].date()),
                                   "n_days": len(test_dates),
                                   # 空仓窗：策略 nav 不动，窗口收益恒 0；基准照常推进
                                   "window_return": 0.0,
                                   "bench_window_return": bench_nav / w_bench_start - 1,
                                   "status": "all_factors_insignificant"})
            continue

        # 簇B分层：按因子 layer 标注把训练后的 directed 分片到各层（静态训练只算方向，权重来自
        # config，跨层无冲突；结果 dict 与单层路径各出各的测试段）。层容 + series 缓存子集化
        # （series_caches 键 factor.name，按名字子集，无额外内存预算，见 [[backtest-perf-optimization]]）。
        # 层状态(lstate)跨窗口延续：holdings 持有跨窗口，directed/series_cache 每窗口按新训练刷新。
        if layers_enabled:
            for _ln, _st in lstate.items():
                _st["directed"] = []
                _st["series_cache"] = None
            _laid = {}
            for _f, _w in directed:
                _laid.setdefault(getattr(_f, "layer", None) or "slow", []).append((_f, _w))
            for _ln, _st in lstate.items():
                _st["directed"] = _laid.get(_ln, [])
            # 空层（全部因子走一组）不 emit：各层仍需独立选股，保证每层在总 N 里占份。
            _diag_layers = {_ln: len(_st["directed"]) for _ln, _st in lstate.items()}
            if verbose:
                print(f"  窗口{wi} 分层因子数 (slow/medium/fast): "
                      f"{_diag_layers.get('slow', 0)}/{_diag_layers.get('medium', 0)}/"
                      f"{_diag_layers.get('fast', 0)} 层规格 {layer_specs}")

        # 预算测试段因子序列缓存（linear + ml 共用）：把每调仓日×每股票从头算 rolling
        # 降为每窗口每股票算一次 calc_series、调仓日索引。ML 路径同样受益——全市场 1800 只
        # × 18 因子下预测侧逐点 calc 是主瓶颈（[[ml-training-fixes-gap-nan-degradation]] 后续
        # 性能修复）。calc_series 尾 == calc（烟测验证），口径逐字不变；未实现 calc_series 的
        # 因子缓存为空 dict，calc_cross_section 自动回退逐点路径。
        series_caches = _build_series_caches(valid, directed)
        # 分层分支复用同一份全局缓存（键 factor.name，composite_score 按名 lookup，
        # 任意层因子子集都命中，无额外内存预算）——替代 lstate reset 里置 None 的占位。
        if layers_enabled:
            for _ln in lstate:
                lstate[_ln]["series_cache"] = series_caches

        if verbose:
            if weighting == "ml":
                imp = dict(zip(feature_names, ml_model.feature_importance(importance_type="gain").tolist()))
                top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
                imp_str = ", ".join(f"{nm}:{v:.0f}" for nm, v in top)
                _vs = "有验证集" if _ml_diag.get("has_val_set") else "无验证集用满轮"
                _cv = _ml_diag.get("cv") or {}
                _cv_str = (f" cv[{_cv.get('chosen')} score={_cv.get('score'):.6f} "
                           f"rounds={_cv.get('rounds')}]") if _cv else ""
                print(f"  窗口{wi} 训练截止{train_end.date()} [ML]: best_iter={_ml_diag.get('best_iter')} "
                      f"({_vs}){_cv_str} top特征 {imp_str}")
            elif weighting == "ridge":
                print(f"  窗口{wi} 训练截止{train_end.date()} [ridge]: "
                      f"alpha={_ml_diag.get('alpha'):.3g} n_train={_ml_diag.get('n_train')}")
            else:
                signs = ", ".join(f"{f.name}:{'+' if s>0 else ('-' if s<0 else '0')}"
                                  for f, _, s, _, _, _ in trained)
                print(f"  窗口{wi} 训练截止{train_end.date()}: {signs}")
                if weighting == "ic" and directed:
                    ws = ", ".join(f"{f.name}:{w:+.2f}" for f, w in directed)
                    print(f"           IC 权重: {ws}")

        # --- 测试：在测试段用训练出的方向选股回测 ---
        w_nav_start = nav
        if layers_enabled:
            # ======== 簇B分层路径（独立分支，legacy 单层不动） ========
            # 聚合 A 分层子组合：w_s = Σ_l δ(l,s)×(1/n_layers)/N_l，
            # 每层独立等权子组合、跨层重叠权重相加，总毛敞口恒~1。
            # 各层独立调仓周期（slow/medium/fast），某层刷新只动自己的 pending_trade，
            # 他层 holdings 冻结 → 聚合 diff 成本对冻结层≈0，实现真实低换手（问题②解药）。
            # 与 legacy 差异（实验记录）：① 层内不做个股 vol-weight（等权，保持基权1/N_l干净），
            # 风险缩放统一走顶层 port_ret（RSRS/CPPI/vol-target，默认全关→no-op）；② RSRS 改顶层。
            n_layers = len(lstate)

            def _layer_positions():
                pos = {}
                for _lnn, _stt in lstate.items():
                    _ld = _stt["holdings"]
                    if not _ld:
                        continue
                    _base = (1.0 / n_layers) / len(_ld)
                    for _s in _ld:
                        pos[_s] = pos.get(_s, 0.0) + _base
                return pos

            for date in test_dates:
                day_cost = 0.0
                # (a) 止盈止损 —— 逐层：止出的该层剩余按等权归一（保持层毛敞口=1/n_layers）
                for ln, st in lstate.items():
                    if not st["holdings"]:
                        continue
                    to_sell = risk.filter_stops(st["holdings"], date, valid)
                    sellable = [s for s in to_sell if is_tradeable(s, date, valid, "sell")]
                    rejected_sells += len(to_sell) - len(sellable)
                    to_sell = sellable
                    if to_sell:
                        for s in to_sell:
                            reason = risk.check_stop_for_date(s, date, valid, st["holdings"][s]["entry_price"],
                                                              st["holdings"][s].get("entry_date"))
                            stop_count[reason] = stop_count.get(reason, 0) + 1
                        sell_w = sum(st["holdings"][s]["weight"] for s in to_sell) / n_layers
                        day_cost += sell_w * cost_sell
                        total_cost += sell_w * cost_sell
                        st["holdings"] = {s: i for s, i in st["holdings"].items() if s not in to_sell}
                        if st["holdings"]:
                            per = 1.0 / len(st["holdings"])
                            for s in st["holdings"]:
                                st["holdings"][s]["weight"] = per
                # (b) 执行 —— 聚合 diff 成本（慢层冻结→重平衡≈0）
                pending_layers = [ln for ln, st in lstate.items() if st["pending_trade"] is not None]
                if pending_layers:
                    old_pos = _layer_positions()
                    temp_hold = {}
                    for ln in pending_layers:
                        st = lstate[ln]
                        syms = st["pending_trade"]
                        per = 1.0 / len(syms) if syms else 0.0
                        nh = {}
                        for s in syms:
                            op = get_open(s, date)
                            if op is None:
                                continue
                            if not is_tradeable(s, date, valid, "buy"):
                                rejected_buys += 1
                                continue
                            old = st["holdings"].get(s)
                            if old:
                                nh[s] = {"weight": per, "entry_price": old["entry_price"],
                                         "entry_date": old["entry_date"]}
                            else:
                                nh[s] = {"weight": per, "entry_price": op, "entry_date": date}
                        temp_hold[ln] = nh

                    def _pos_with():
                        pos = {}
                        for _lnn, _stt in lstate.items():
                            _ld = temp_hold[_lnn] if _lnn in temp_hold else _stt["holdings"]
                            if not _ld:
                                continue
                            _bb = (1.0 / n_layers) / len(_ld)
                            for _s in _ld:
                                pos[_s] = pos.get(_s, 0.0) + _bb
                        return pos

                    new_pos = _pos_with()
                    buy_amt = sum(new_pos[s] for s in new_pos if s not in old_pos)
                    sell_amt = sum(old_pos[s] for s in old_pos if s not in new_pos)
                    rebal = sum(abs(new_pos.get(s, 0) - old_pos.get(s, 0))
                                for s in set(new_pos) | set(old_pos))
                    trade_cost = (buy_amt * cost_buy + sell_amt * cost_sell
                                  + rebal * (cost_buy + cost_sell) / 2) / 2
                    day_cost += trade_cost
                    total_cost += trade_cost
                    turnovers.append(compute_turnover(list(new_pos.keys()), list(old_pos.keys())))
                    for ln in pending_layers:
                        lstate[ln]["holdings"] = temp_hold[ln]
                        lstate[ln]["pending_trade"] = None
                # (c) 当日收益 + 顶层总仓位缩放（RSRS/CPPI/vol-target，默认全关→恒1.0）
                port_ret = 0.0
                for s, w in _layer_positions().items():
                    df = valid.get(s)
                    if df is None:
                        continue
                    row = df[df["date"] == date]
                    if not row.empty and len(df[df["date"] <= date]) >= 2:
                        prev_close = df[df["date"] < date]["close"].iloc[-1]
                        sym_ret = row["close"].iloc[0] / prev_close - 1
                        port_ret += w * sym_ret
                        if attribution:
                            attribution_log["daily_returns"].append(
                                {"date": date, "window": wi, "symbol": s, "weight": w, "ret": sym_ret})
                _rsrs = market_timer.get_multiplier(date) if market_timer.enabled else 1.0
                _cppi_w = cppi_timer.update(nav) if cppi_timer.enabled else 1.0
                _vt_w = vol_target_timer.update(nav) if vol_target_timer.enabled else 1.0
                if market_timer.enabled or cppi_timer.enabled or vol_target_timer.enabled:
                    port_ret *= min(_rsrs, _cppi_w, _vt_w)
                nav *= (1 + port_ret) * (1 - day_cost)
                br = bench_ret.get(date, 0.0)
                if br is None or (isinstance(br, float) and np.isnan(br)):
                    br = 0.0
                bench_nav *= (1 + br)
                nav_series.append({"date": date, "nav": nav})
                bench_series.append({"date": date, "nav": bench_nav})
                # (d) 收盘逐层产生信号（各层独立 rebalance_days/走各自 top_n+buffer）
                _need_current = True
                for ln, st in lstate.items():
                    spec = layer_specs[ln]
                    if st["days_since_rebal"] >= spec["rebalance_days"] or not st["holdings"]:
                        if _need_current:
                            current = {s: df[df["date"] <= date] for s, df in valid.items()
                                       if (df["date"] <= date).any()}
                            _need_current = False
                        if not st["directed"]:
                            # 该层所有因子该窗口不显著：该层空仓（贡献0），不挂 pending
                            st["holdings"] = {}
                            st["pending_trade"] = None
                            st["days_since_rebal"] = 0
                            continue
                        out = composite_score(current, st["directed"], combine_cfg=combine_cfg,
                                              return_factors_cs=attribution,
                                              series_caches=st["series_cache"])
                        if attribution:
                            score, cs_map = out
                        else:
                            score = out
                            cs_map = {}
                        new_syms = select_portfolio(score, spec["top_n"],
                                                    list(st["holdings"].keys()), spec["buffer"])
                        ifilt = mf.get("intraday_filter", {})
                        vs_cfg = ifilt.get("volume_surge", {}) if ifilt.get("enabled") else {}
                        if vs_cfg.get("enabled", True) and vs_cfg:
                            _pending, vstats = filter_by_volume(
                                new_syms, date, valid, spec["top_n"],
                                threshold=vs_cfg.get("threshold", 1.0),
                                window=vs_cfg.get("window", 20),
                                min_keep_ratio=vs_cfg.get("min_keep_ratio", 0.8))
                            vol_filtered += vstats["filtered"]
                            vol_backfilled += vstats["backfilled"]
                        else:
                            _pending = new_syms
                        os_cfg = ifilt.get("open_strength", {}) if ifilt.get("enabled") else {}
                        if os_cfg.get("enabled", False) and os_cfg and valid_min is not None:
                            _pending, ostats = filter_by_open_strength(
                                _pending, date, valid_min, spec["top_n"],
                                threshold=os_cfg.get("threshold", 0.0),
                                window=os_cfg.get("window", 48),
                                min_keep_ratio=os_cfg.get("min_keep_ratio", 0.8))
                            os_filtered += ostats["filtered"]
                            os_backfilled += ostats["backfilled"]
                        amp_cfg = ifilt.get("amplitude", {}) if ifilt.get("enabled") else {}
                        if amp_cfg.get("enabled", False) and amp_cfg and valid_min is not None:
                            _pending, astats = filter_by_amplitude(
                                _pending, date, valid_min, spec["top_n"],
                                threshold=amp_cfg.get("threshold", 0.0),
                                window=amp_cfg.get("window", 48),
                                min_keep_ratio=amp_cfg.get("min_keep_ratio", 0.8))
                            amp_filtered += astats["filtered"]
                            amp_backfilled += astats["backfilled"]
                        st["pending_trade"] = _pending
                        st["days_since_rebal"] = 0
                        if attribution and new_syms:
                            for factor, w in st["directed"]:
                                cs = cs_map.get(factor.name, pd.Series(dtype=float))
                                if cs.empty:
                                    continue
                                attribution_log["selection_scores"].append(
                                    {"date": date, "window": wi, "factor": factor.name,
                                     "weight": w, "selected_scores": cs.reindex(new_syms).to_dict()})
                    else:
                        st["pending_trade"] = None
                        st["days_since_rebal"] += 1
            window_reports.append({"window": wi, "train_end": train_end,
                                   "window_start": str(test_dates[0].date()),
                                   "window_end": str(test_dates[-1].date()),
                                   "n_days": len(test_dates),
                                   "window_return": nav / w_nav_start - 1,
                                   "bench_window_return": bench_nav / w_bench_start - 1,
                                   "status": "ok", "layered": True})
            continue
        w_nav_start = nav
        for date in test_dates:
            day_cost = 0.0
            # (a) 止盈止损检查
            if holdings:
                to_sell = risk.filter_stops(holdings, date, valid)
                sellable = [s for s in to_sell if is_tradeable(s, date, valid, "sell")]
                rejected_sells += len(to_sell) - len(sellable)  # 跌停锁死：卖不掉，保留持仓次日重试
                to_sell = sellable
                if to_sell:
                    for s in to_sell:
                        reason = risk.check_stop_for_date(s, date, valid, holdings[s]["entry_price"],
                                                          holdings[s].get("entry_date"))
                        stop_count[reason] = stop_count.get(reason, 0) + 1
                        if collect_exit_pnls:
                            _p = get_open(s, date)
                            _ep = holdings[s]["entry_price"]
                            exit_pnls[reason].append(_p / _ep - 1 if _p and _ep > 0 else float('nan'))
                            if collect_exit_events and _p and _ep > 0:
                                exit_events.append(
                                    {"symbol": s, "sell_date": date, "reason": reason,
                                     "sell_price": _p, "pnl": _p / _ep - 1})
                    sell_w = sum(holdings[s]["weight"] for s in to_sell)
                    day_cost += sell_w * cost_sell
                    total_cost += sell_w * cost_sell
                    holdings = {s: info for s, info in holdings.items() if s not in to_sell}
                    if holdings:
                        scale = 1.0 / sum(info["weight"] for info in holdings.values())
                        for s in holdings:
                            holdings[s]["weight"] *= scale
            # (b) 执行待成交信号
            if pending_trade is not None:
                new_syms = pending_trade
                target_w = 1.0 / len(new_syms) if new_syms else 0
                new_weights = {s: target_w for s in new_syms}
                if risk.target_vol is not None and new_syms:
                    mults = {s: risk.vol_weight_multiplier(s, date, valid) for s in new_syms}
                    total_mult = sum(mults.values()) or 1.0
                    avg_mult = total_mult / len(new_syms)
                    new_weights = {s: target_w * mults[s] / avg_mult for s in new_syms}
                # 大盘择时仓位乘数（RSRS）：对整个组合统一缩放总仓位（满仓→底仓）。
                # 在个股波动率加权（相对权重层）之后、交易成本计算之前施加（总仓位层），
                # 串联非并联。默认关闭时 get_multiplier 恒 1.0 无开销。
                if market_timer.enabled and new_syms:
                    market_mult = market_timer.get_multiplier(date)
                    new_weights = {s: w * market_mult for s, w in new_weights.items()}
                old_w = {s: info["weight"] for s, info in holdings.items()}
                buy_amt = sum(new_weights[s] for s in new_syms if s not in old_w)
                sell_amt = sum(old_w[s] for s in old_w if s not in new_weights)
                if collect_exit_pnls:
                    for s in old_w:
                        if s not in new_weights and s in holdings:
                            _p = get_open(s, date)
                            _ep = holdings[s]["entry_price"]
                            exit_pnls["passive"].append(_p / _ep - 1 if _p and _ep > 0 else float('nan'))
                            if collect_exit_events and _p and _ep > 0:
                                exit_events.append(
                                    {"symbol": s, "sell_date": date, "reason": "passive",
                                     "sell_price": _p, "pnl": _p / _ep - 1})
                rebal = sum(abs(new_weights.get(s, 0) - old_w.get(s, 0))
                            for s in set(new_weights) | set(old_w))
                trade_cost = (buy_amt * cost_buy + sell_amt * cost_sell
                              + rebal * (cost_buy + cost_sell) / 2) / 2
                day_cost += trade_cost
                total_cost += trade_cost
                # 单边换手率采集（与 portfolio_runner 口径一致：目标篮 vs 上一期持仓）
                turnovers.append(compute_turnover(new_syms, list(old_w.keys())))
                new_holdings = {}
                for s in new_syms:
                    op = get_open(s, date)
                    if op is None:
                        continue
                    if not is_tradeable(s, date, valid, "buy"):  # 涨停开盘锁死：买不到，跳过该股
                        rejected_buys += 1
                        continue
                    if s in holdings:
                        new_holdings[s] = {"weight": new_weights[s],
                                           "entry_price": holdings[s]["entry_price"],
                                           "entry_date": holdings[s]["entry_date"]}
                    else:
                        new_holdings[s] = {"weight": new_weights[s],
                                           "entry_price": op, "entry_date": date}
                holdings = new_holdings
                pending_trade = None
            # (c) 当日收益
            port_ret = 0.0
            if holdings:
                for s, info in holdings.items():
                    df = valid.get(s)
                    if df is None:
                        continue
                    row = df[df["date"] == date]
                    if not row.empty and len(df[df["date"] <= date]) >= 2:
                        prev_close = df[df["date"] < date]["close"].iloc[-1]
                        sym_ret = row["close"].iloc[0] / prev_close - 1
                        port_ret += info["weight"] * sym_ret
                        # 归因 hook B：记录每只持仓股当日收益与权重
                        if attribution:
                            attribution_log["daily_returns"].append(
                                {"date": date, "window": wi, "symbol": s,
                                 "weight": info["weight"], "ret": sym_ret})
            # CPPI 总仓位缩放：用 t-1 收盘 nav 算 t 日仓位系数 w，缩当日组合收益。
            # 在 port_ret 层施加（非权重层）：w 缩总仓位，现金占比 (1-w) 收益为 0，
            # 个股相对权重不变。与 RSRS 的根本差异——CPPI 依赖内生 nav 无法预计算，
            # 必须逐日实时算且非调仓日也缩仓（否则 21 天里 20 天仓位不受控）。
            # 无未来函数：nav 在此处是 t-1 收盘值（前一日循环末已更新），update 用 t-1
            # 算 w_t 喂 t 日 port_ret，与 overlay_cppi.py 的 V[t-1] 口径一致（已校准）。
            # 默认关闭→update 恒 1.0 零开销，行为逐字不变。
            # 仓位层总仓位缩放（CPPI + vol-target，同在 port_ret 层逐日施加）。
            # 两者机制相反：CPPI 看净值水平（cushion）反应性保底，vol-target 看已实现波动率
            # （σ_target/σ_realized）预防性控风险水平。同时开时取 min（较严减仓胜出，不叠加）。
            # 各自默认关闭→update 恒 1.0；只一个开时即乘该 w。无未来函数：w 用 t-1 nav 算。
            _cppi_w = cppi_timer.update(nav) if cppi_timer.enabled else 1.0
            _vt_w = vol_target_timer.update(nav) if vol_target_timer.enabled else 1.0
            if cppi_timer.enabled or vol_target_timer.enabled:
                port_ret *= min(_cppi_w, _vt_w)
            nav *= (1 + port_ret) * (1 - day_cost)
            br = bench_ret.get(date, 0.0)
            if br is None or (isinstance(br, float) and np.isnan(br)):
                br = 0.0
            bench_nav *= (1 + br)
            nav_series.append({"date": date, "nav": nav})
            bench_series.append({"date": date, "nav": bench_nav})
            # (d) 收盘产生信号（rebalance_days 控制换仓频率：仅到调仓日或空仓时重算，否则保持持仓）
            if days_since_rebal >= rebalance_days or not holdings:
                current = {s: df[df["date"] <= date] for s, df in valid.items()
                           if (df["date"] <= date).any()}
                # 漏斗第1层硬门：打分前清洗宇宙（config 门控，默认关→零回归）。
                # 只在 static 合成打分路径应用（ML/ridge 打分域走下方统一 select_syms 口径，见下）。
                # ST/流动性/停牌/新股：min_liquidity_turnover/max_suspension_days/min_listing_days
                # 默认全空 → 恒返回原 current，逐字不变（[[funnel-top10-rewrite]] 第1刀）。
                if weighting not in ("ml", "ridge"):
                    hg = (mf.get("hard_gate") or {})
                    if hg.get("enabled") and any(
                            hg.get(k) for k in ("min_liquidity_turnover",
                                                "max_suspension_days", "min_listing_days")):
                        current = _apply_hard_gate(current, date, hg)
                # 归因 hook 复用：调仓日把合成函数内部已算的各因子截面打分一并取回，
                # 避免下面归因段对每因子重算 calc_cross_section（同一 current、同一批因子，二算纯浪费）。
                # return_factors_cs 与 attribution 同步：attribution=True 时取回 cs_map 喂归因 hook，
                # False 时合成函数只返 score（无 cs_map），此处补空 dict——cs_map 仅在下方
                # `if attribution and new_syms` 段被读，False 路径不读，空 dict 安全。
                # 打分域口径（2026-09-24 审计#11 修复，扩到线性分支）：选股打分域恒为
                # universe（select_syms），benchmark 成分股绝不进排名。ML 分支 2026-09-04
                # 已修（csi1000 选股 + csi300 基准时 csi300 大盘股混入 rank，实证曾混入
                # 农行/中行/人保），线性分支当时漏修——universe 与 benchmark 不相交的池
                # （如 csi500 + csi300 基准）会选出池外大盘股，且信号侧只打分 universe、
                # 两侧口径分裂。csi1000 + 同池基准（compare_ml universe_bench）下
                # union=universe，过滤为 no-op。宽训练池 ml_train_pool 的隔离逻辑已在
                # 上游 select_syms/universe 层处理，此处不再特判。
                # M7：选股域按当日 U(t) 逐日收窄（缺省 expansion_enabled=false → 走
                # 上面的 select_syms 静态过滤，逐字不变）。**必须替换而非叠加**：
                # select_syms 是 csi1000 成员集，若先按它过滤再与 U 求交，扩充进来的
                # 非成员会被全部滤掉，M7 就退化成了基线。U(t) 已含当日全部在册成员，
                # 因此直接以 U(t) 为准。必须用**当日**的 U(t)，不能用历史并集，否则
                # 早期调仓日会选到当日尚未入池的标的。
                if expansion_enabled:
                    _u = universe_panel.get(date)
                    current = {s: d for s, d in current.items() if _u is None or s in _u}
                else:
                    current = {s: d for s, d in current.items() if s in select_syms}
                if weighting in ("ml", "ridge"):
                    out = composite_score_ml(current, directed, ml_model, feature_names,
                                             return_factors_cs=attribution,
                                             series_caches=series_caches)
                else:
                    out = composite_score(current, directed, combine_cfg=combine_cfg,
                                          return_factors_cs=attribution,
                                          series_caches=series_caches)
                if attribution:
                    score, cs_map = out
                else:
                    score = out
                    cs_map = {}
                new_syms = select_portfolio(score, top_n, list(holdings.keys()), buffer)
                # 执行层过滤：链式 volume_surge → open_strength（T 日收盘信号 → T+1 开盘成交）。
                # 只改 pending_trade 不改 new_syms：归因 hook A（下方）用 new_syms 记录各因子
                # 对"次日将持仓"股票的截面打分，保持归因记录纯选股层，执行层增益单独看计数器。
                ifilt = mf.get("intraday_filter", {})
                vs_cfg = ifilt.get("volume_surge", {}) if ifilt.get("enabled") else {}
                if vs_cfg.get("enabled", True) and vs_cfg:
                    _pending, vstats = filter_by_volume(
                        new_syms, date, valid, top_n,
                        threshold=vs_cfg.get("threshold", 1.0),
                        window=vs_cfg.get("window", 20),
                        min_keep_ratio=vs_cfg.get("min_keep_ratio", 0.8),
                    )
                    vol_filtered += vstats["filtered"]
                    vol_backfilled += vstats["backfilled"]
                else:
                    _pending = new_syms
                # 第二因子（链式）：open_strength 用分钟 valid_min，输入是上一因子输出 _pending
                os_cfg = ifilt.get("open_strength", {}) if ifilt.get("enabled") else {}
                if os_cfg.get("enabled", False) and os_cfg and valid_min is not None:
                    _pending, ostats = filter_by_open_strength(
                        _pending, date, valid_min, top_n,
                        threshold=os_cfg.get("threshold", 0.0),
                        window=os_cfg.get("window", 48),
                        min_keep_ratio=os_cfg.get("min_keep_ratio", 0.8),
                    )
                    os_filtered += ostats["filtered"]
                    os_backfilled += ostats["backfilled"]
                # 第三因子（链式）：amplitude 用分钟 valid_min，输入是上一因子输出 _pending。
                # IC=-0.108 显著为负（振幅大→未来差），信号取负→保低振幅滤高振幅。
                amp_cfg = ifilt.get("amplitude", {}) if ifilt.get("enabled") else {}
                if amp_cfg.get("enabled", False) and amp_cfg and valid_min is not None:
                    _pending, astats = filter_by_amplitude(
                        _pending, date, valid_min, top_n,
                        threshold=amp_cfg.get("threshold", 0.0),
                        window=amp_cfg.get("window", 48),
                        min_keep_ratio=amp_cfg.get("min_keep_ratio", 0.8),
                    )
                    amp_filtered += astats["filtered"]
                    amp_backfilled += astats["backfilled"]
                pending_trade = _pending
                days_since_rebal = 0
                # 归因 hook A：调仓日记录各 directed 因子对"次日将持仓"股票的截面打分
                # （new_syms 即次日开盘将建的持仓；cs 取自 composite_score 返回，与现场重算逐字相同）
                if attribution and new_syms:
                    for factor, w in directed:
                        cs = cs_map.get(factor.name, pd.Series(dtype=float))
                        if cs.empty:
                            continue
                        attribution_log["selection_scores"].append(
                            {"date": date, "window": wi, "factor": factor.name,
                             "weight": w, "selected_scores": cs.reindex(new_syms).to_dict()})
            else:
                pending_trade = None  # 非调仓日不换仓，保持现持仓
                days_since_rebal += 1

        window_reports.append({"window": wi, "train_end": train_end,
                               "window_start": str(test_dates[0].date()),
                               "window_end": str(test_dates[-1].date()),
                               "n_days": len(test_dates),
                               "window_return": nav / w_nav_start - 1,
                               "bench_window_return": bench_nav / w_bench_start - 1,
                               "status": "ok"})

    result = _stats(nav_series, bench_series, verbose)
    result["total_cost"] = total_cost
    result["avg_turnover"] = np.mean(turnovers) if turnovers else 0.0
    result["stop_count"] = stop_count
    result["rejected_buys"] = rejected_buys
    result["rejected_sells"] = rejected_sells
    result["vol_filtered"] = vol_filtered
    result["vol_backfilled"] = vol_backfilled
    result["os_filtered"] = os_filtered
    result["os_backfilled"] = os_backfilled
    result["amp_filtered"] = amp_filtered
    result["amp_backfilled"] = amp_backfilled
    result["n_windows"] = len(windows)
    result["window_reports"] = window_reports
    result["direction_log"] = direction_log
    if attribution:
        result["attribution_log"] = attribution_log
    if collect_exit_pnls:
        result["exit_pnls"] = exit_pnls
    if collect_exit_events:
        result["exit_events"] = exit_events
    if verbose:
        print(f"  累计交易成本 : {total_cost:.2%}")
        print(f"  平均换手率   : {result['avg_turnover']:.2%} (单边/日)")
        print(f"  止盈止损触发 : 止损{stop_count['stop_loss']}次 止盈{stop_count['take_profit']}次")
        print(f"  涨跌停拒绝   : 买入{rejected_buys}次 卖出{rejected_sells}次")
        if vol_filtered or vol_backfilled:
            print(f"  放量过滤     : 净剔除{vol_filtered}次 兜底补回{vol_backfilled}次")
        if os_filtered or os_backfilled:
            print(f"  开盘强度过滤 : 净剔除{os_filtered}次 兜底补回{os_backfilled}次")
        if amp_filtered or amp_backfilled:
            print(f"  振幅过滤     : 净剔除{amp_filtered}次 兜底补回{amp_backfilled}次")
        print(f"  样本外窗口数 : {len(windows)}")
        if cppi_timer.enabled:
            cov = cppi_timer.coverage()
            print(f"  CPPI 仓位层 : 调用{cov[0]}次 均仓{cov[1]:.2f} "
                  f"最低{cov[2]:.2f} 最高{cov[3]:.2f}")
        if vol_target_timer.enabled:
            cov = vol_target_timer.coverage()
            print(f"  Vol-Target   : 调用{cov[0]}次 均仓{cov[1]:.2f} "
                  f"最低{cov[2]:.2f} 最高{cov[3]:.2f} "
                  f"(σ_target={vol_target_timer.sigma_target} W={vol_target_timer.window})")

    # 可选落盘 nav 序列（回撤控制叠加层实验用底）。默认 persist_nav=None 完全跳过，零回归风险。
    if persist_nav is not None and nav_series:
        try:
            nav_df = pd.DataFrame(nav_series).set_index("date")
            bench_df = pd.DataFrame(bench_series).set_index("date")
            out = nav_df.rename(columns={"nav": "nav"}).join(
                bench_df.rename(columns={"nav": "bench_nav"}), how="left")
            out = out.sort_index().reset_index()
            out["bench_nav"] = out["bench_nav"].ffill().bfill()
            path = Path(persist_nav)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path, index=False)
            if verbose:
                print(f"  nav 落盘     : {path}（{len(out)} 天）")
        except Exception as e:  # 落盘失败不影响回测结果（诊断用，不抛）
            if verbose:
                print(f"  nav 落盘失败 : {e}")
    return result


def _stats(nav_series, bench_series, verbose):
    nav_df = pd.DataFrame(nav_series).set_index("date")
    bench_df = pd.DataFrame(bench_series).set_index("date")
    if nav_df.empty:
        return {"total_return": 0, "ann_return": 0, "max_drawdown": 0, "sharpe": 0,
                "excess_annual": 0, "n_days": 0}
    daily_ret = nav_df["nav"].pct_change().dropna()
    bench_ret = bench_df["nav"].pct_change().dropna()
    n_years = len(nav_df) / 252
    total_ret = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    bench_total = bench_df["nav"].iloc[-1] / bench_df["nav"].iloc[0] - 1
    bench_ann = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    peak = nav_df["nav"].cummax()
    max_dd = (nav_df["nav"] / peak - 1).min()
    sharpe = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    bench_sharpe = (bench_ret.mean() * 252 - 0.02) / (bench_ret.std() * np.sqrt(252)) if bench_ret.std() > 0 else 0
    result = {"total_return": total_ret, "ann_return": ann_ret,
              "bench_total": bench_total, "bench_ann": bench_ann,
              "excess_annual": ann_ret - bench_ann, "max_drawdown": max_dd,
              "sharpe": sharpe, "bench_sharpe": bench_sharpe, "n_days": len(nav_df)}
    if verbose:
        print("\n" + "=" * 50)
        print("  滚动样本外回测（walk-forward）绩效")
        print("=" * 50)
        print(f"  样本外天数   : {result['n_days']}")
        print(f"  累计收益     : 组合 {total_ret:+.2%}  | 基准 {bench_total:+.2%}")
        print(f"  年化收益     : 组合 {ann_ret:+.2%}  | 基准 {bench_ann:+.2%}")
        print(f"  超额年化     : {ann_ret - bench_ann:+.2%}")
        print(f"  最大回撤     : {max_dd:.2%}")
        print(f"  夏普比率     : 组合 {sharpe:.2f}  | 基准 {bench_sharpe:.2f}")
    return result


def main():
    # stdout 行缓冲化：重定向到文件时默认按 8KB 块缓冲，进度日志滞后整块
    # （2026-10-20 巡检曾因此把健康进程误判为无响应）。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    # 命令行直跑兜底内存护栏（RLIMIT_AS 12G）：对比 compare_ml.py 场景级 worker 各自受限，
    # 本入口对单进程也设上限，防 walk-forward 直跑时全市场×Alpha158 单进程就吃爆内存拖死整机。
    # 仅影响自身进程；设失败静默跳过（尽力防护，不阻断）。
    try:
        import resource
        _limit = 12 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
        print(f"[护栏] 本进程 RLIMIT_AS={_limit/2**30:.0f}G，超限崩进程不拖系统")
    except (ValueError, OSError, ImportError):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_walk_forward(cfg, universe=args.universe, train_days=args.train_days,
                     test_days=args.test_days, step_days=args.step_days,
                     horizon=args.horizon)


if __name__ == "__main__":
    main()
