"""
期货多标的时间序列组合回测（TSMOM + Carry 组合的样本外验证）。

定位（区别于 A 股两条回测线）：
  - backtest/runner.py         单标的 backtrader 回测（含期货 adapter 下单）
  - backtest/walk_forward.py   股票横截面多因子 walk-forward（排序选股）
  - 本文件                  期货多标的时间序列组合（每品种出方向→组合→风控）

核心：TSMOM/Carry 是"无参数时序信号"（只用自身过去数据预测自身未来方向），
不需要训练段拟合参数。因此本回测对全部历史逐日滚动算信号，天然满足样本外
（t 日信号只用 t-1 及以前数据）。复用 walk_forward._segment_dates/_stats 做
窗口切分与绩效统计（只 import 不改）。

流程（逐日 t）：
  1. 每品种 i：用截至 t 的 bars 调 on_bar → 方向 dir_i ∈ {+1,-1,0}
  2. 组合原始权重：
     等权           w_i = 1/N
     波动率平价     w_i ∝ 1/σ_i（σ_i 用过去 window 日日收益 std, 归一化到总权重=杠杆预算）
  3. FuturesRiskManager.apply(w_i, margin_i) → 缩放后带符号权重 w_i*
     （vol-target → 杠杆 → 单品种保证金 → 保证金预算）
  4. 组合日收益 = Σ w_i* · r_i(t)（r_i = settle(t)/settle(t-1)-1）
  5. 累计 NAV：调仓日按上期已执行权重计单边成本（佣金+滑点，无印花税）
  6. 逐品种归因 + 可选逐因子归因（TSMOM-only / Carry-only，--attribution）

保证金口径 = 约束 + 不计息：组合保证金占用 ≤ capital_utilization（超限风控内整体缩仓），
闲置现金不计息（不影响 nav 增益）。

CLI:
  python -m backtest.futures_walk_forward --config config.yaml \
      --strategy tsmom_cta --contract X0 [--rebalance-days 1]
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jq_framework.backtest.walk_forward import _segment_dates, _stats
from jq_framework.data.fetchers.fetcher_futures import (
    build_contract_series,
    get_futures_main_continuous,
    get_futures_rules,
    get_roll_yield_series,
    normalize_built_frame,
    resolve_contract,
)
from jq_framework.portfolio.futures_sizing import futures_trade_cost
from jq_framework.risk.futures_risk import FuturesRiskManager

# carry 类策略需注入展期收益率；此处按策略模块/类名识别是否 carry 类
# TripleFactorStrategy 内部含 CarryStrategy，同样需注入 carry。
_CARRY_CLASSES = {"CarryStrategy", "TsmomCarryStrategy", "TripleFactorStrategy"}


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def build_strategy(strat_cfg: dict):
    module = importlib.import_module(strat_cfg["module"])
    cls = getattr(module, strat_cfg["class"])
    return cls(params=strat_cfg.get("params", {})), cls.__name__


def _need_carry(class_name: str) -> bool:
    return class_name in _CARRY_CLASSES


def _roi_series(df: pd.DataFrame) -> pd.Series:
    """品种日收益序列（index=date）。用 settle(结算价) 计算 t-1→t，首行 NaN。

    口径：TSMOM 方向信号用 settle（_pick_price 优先 settle），若收益路径用 close
    会与信号/保证金盯市口径脱节。故收益也切到 settle 版；settle 缺失/全 0 时
    回退 close（脏数据防御，与 _pick_price 同思路）。
    """
    if "settle" in df.columns:
        s = pd.to_numeric(df["settle"], errors="coerce")
        if s.notna().any() and (s.dropna() > 0).any():
            # S2 脏值防护：零星 settle==0/负值(脏数据) 直接进 pct_change 会产出 -100%
            # 假收益穿透生产 P&L（且 pct_change 遇 NaN 前值会把整段尾部也变 NaN）。
            # 把 <=0 行**前向填充为上一个有效结算价** → 该行收益=0（视为无成交/不赌），
            # 相邻行不受污染、尾部保持完好（close 回退路径同样处理）。
            s = s.mask(s <= 0).ffill()
            return pd.Series(s.values, index=pd.to_datetime(df["date"])).pct_change()
    close = pd.to_numeric(df["close"], errors="coerce") if "close" in df.columns else None
    if close is not None:
        close = close.mask(close <= 0).ffill()
        return pd.Series(close.values, index=pd.to_datetime(df["date"])).pct_change()
    return df.set_index("date")["close"].pct_change()


def _realized_vol(roi: pd.Series, window: int) -> pd.Series:
    """过去 window 日日收益的年化已实现波动率（滚动 std×√252）。"""
    return roi.rolling(window, min_periods=max(3, window // 4)).std() * np.sqrt(252)


def _make_weights(raw_dirs: dict[str, float], method: str,
                  vols: dict[str, float], budget: float,
                  min_vol: float) -> dict[str, float]:
    """组合原始权重。method: "equal" / "vol_parity"。budget=杠杆预算(目标总|w|)。"""
    active = {k: v for k, v in raw_dirs.items() if v != 0}
    if not active:
        return {}
    if method == "vol_parity":
        # w_i ∝ 1/σ_i，归一化使总|w|=budget（信号方向带符号）
        inv = {k: 1.0 / max(vols.get(k, min_vol), min_vol) for k in active}
        s = sum(inv.values())
        w = {k: budget * (inv[k] / s) * (1 if raw_dirs[k] > 0 else -1) for k in active}
        return w
    # 等权：每品种 ±budget/N（方向由信号定）
    n = len(active)
    return {k: (budget / n) * (1 if raw_dirs[k] > 0 else -1) for k in active}


def _decorr_vol_parity_weights(raw_dirs: dict[str, float], vols: dict[str, float],
                               budget: float, roi_rets: dict[str, pd.Series],
                               window: int = 60, min_vol: float = 0.05,
                               ridge: float = 1e-6) -> dict[str, float]:
    """相关感知的反波动权重：先用收益协方差白化(去相关)，再逆波动风险平价。

    参考 A股 composite._whiten(eigh 对 X'X 取 -1/2)——只借范式，不 import A股文件。
    roi_rets: {var: 收益Series(index=date)}，取最近 window 日构成设计矩阵 X。
    高相关簇(黑色系 RB/I/JM/J)被白化摊平，低相关品种获更多预算——
    正面回应 docs §7 自承"有效独立下注仅 3-4 个"。

    旁路可选（默认 weight_method 仍 vol_parity，62 测试/默认绩效零回归）。
    """
    active = [k for k in raw_dirs if raw_dirs[k] != 0]
    if not active:
        return {}
    df = pd.DataFrame({k: roi_rets[k].tail(window) for k in active})
    X = df.to_numpy(dtype=float)
    X = X[~np.isnan(X).any(axis=1)]                        # 剔 NaN 行
    if X.shape[0] < max(10, len(active)):                  # 样本不足 → 回退 vol_parity
        return _make_weights(raw_dirs, "vol_parity", vols, budget, min_vol)
    XtX = X.T @ X + ridge * np.eye(len(active))            # ridge 防病态
    eigvals, eigvecs = np.linalg.eigh(XtX)
    eigvals = np.clip(eigvals, 1e-12, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    X_orth = X @ inv_sqrt                                  # 白化：相关簇摊平
    orig_std = np.nanstd(X, axis=0)
    for j in range(X_orth.shape[1]):
        s = np.nanstd(X_orth[:, j])
        if orig_std[j] < 1e-12 or s < 1e-12:
            continue
        X_orth[:, j] *= orig_std[j] / s                    # 保波动量纲
        c = np.corrcoef(X_orth[:, j], X[:, j])[0, 1]
        if np.isnan(c):
            c = float(np.dot(X_orth[:, j], X[:, j]))
        if c < 0:
            X_orth[:, j] *= -1                             # 保号对齐原方向
    post_vol = np.nanstd(X_orth, axis=0) + min_vol
    inv = 1.0 / post_vol
    w = budget * (inv / inv.sum())
    return {k: w[i] * (1 if raw_dirs[k] > 0 else -1)
            for i, k in enumerate(active)}


# 产业类别映射（与 config.yaml futures.symbols 注释的 5 大类一致）。
# 用于类别内权重硬封顶（--weight-method volparity_capped）：同类别品种行情高度同步
# （黑色系 RB/I/JM/J 为钢-煤-焦产业链），名义 17 品种的实际有效下注远少于 17。
# 建立在此映射上的 cap 是"已知产业知识"的硬约束，非收益协方差的统计估计——
# 不引入 decorr_vol_parity 白化那样样本内噪声（§11.1 OOS 被否）。
CATEGORY = {
    "RB": "黑色", "I": "黑色", "JM": "黑色", "J": "黑色",
    "CU": "有色", "AL": "有色", "ZN": "有色", "NI": "有色",
    "AU": "贵金属", "AG": "贵金属",
    "TA": "能化", "MA": "能化", "SC": "能化",
    "M": "农产品", "Y": "农产品", "P": "农产品", "CF": "农产品",
}


def sector_shares(weights_row: dict[str, float]) -> dict[str, float]:
    """单日类别权重份额：Σ类别内 |w| / 总 |w|。纯函数（第六步）。

    组合级等比缩放（vol-target/DD/杠杆）不改变份额，故用风险缩放后的权重
    算份额与缩放前等价。空仓日（总 |w| ≤ 0）返回空 dict。
    证据包（futures_sector_evidence）与每日信号软形态共用此单一定义。
    """
    tot = sum(abs(v) for v in weights_row.values())
    if tot <= 0:
        return {}
    out: dict[str, float] = {}
    for k, v in weights_row.items():
        cat = CATEGORY.get(k, "其他")
        out[cat] = out.get(cat, 0.0) + abs(v)
    return {c: s / tot for c, s in out.items()}


def effective_bets(shares: dict[str, float]) -> float:
    """有效独立下注数 = 1/HHI（类别份额平方和的倒数）。纯函数（第六步）。

    5 类均分 → 5.0，单类独占 → 1.0——板块集中度的标准度量。
    """
    s2 = sum(s * s for s in shares.values())
    return 1.0 / s2 if s2 > 0 else 0.0


def extreme_day_mask(roi: pd.Series, quantile: float = 0.99, window: int = 504,
                     min_obs: int = 126) -> pd.Series:
    """极端日标注（中短线 Phase 0a）：|roi| 超该品种过去 window 日（t-1 及以前）
    |roi| 的 quantile 分位 → True。纯函数。

    数据可用性结论（2026-10-15 查证）：futures_rules_latest.parquet 仅
    [variety, margin_ratio, multiplier]，**无涨跌停字段** → 用因果收益分位代理。
    代理是停板/大跳空的保守上界：凡涨跌停或跳空日必被标，另含普通大波动日——
    作敏感性口径偏保守（多剔）而非偏乐观。
    因果性：阈值 shift(1).rolling → 当日极端不抬高自身门槛；热身期有效观测
    < min_obs（约半年）不标注（宁漏勿猜，晚上市品种前段不参与统计）。
    """
    a = roi.abs()
    thr = a.shift(1).rolling(window, min_periods=min_obs).quantile(quantile)
    return (a > thr) & thr.notna()


def _extreme_day_stats(merged: dict, roi: dict[str, pd.Series],
                       quantile: float = 0.99, window: int = 504,
                       min_obs: int = 126) -> dict:
    """组合级极端日统计（Phase 0a），追加进 stats 的键。纯函数。

    极端日 = 任一持仓品种（|w|>0）当日被 extreme_day_mask 标记。
    口径：组合收益用 port_ret_series（仓位 P&L、成本前——三次成本档下仓位
    路径相同，极端日标注与成本无关）；敏感性 = 极端日收益置 0 后复利年化
    （252 口径，与 _stats 年化一致，不含成本）。总收益近 0 时 ret_share
    置 NaN。空 NAV → 全零/NaN 键，不抛错。
    """
    nav_series = merged.get("nav_series") or []
    port_rets = merged.get("port_ret_series") or []
    weights_log = merged.get("weights_log") or {}
    n = len(nav_series)
    if n == 0 or len(port_rets) < n:
        return {"extreme_days": 0, "extreme_day_ratio": 0.0,
                "extreme_ret_share": float("nan"),
                "ann_return_ex_extreme": 0.0}
    masks = {var: extreme_day_mask(roi[var], quantile, window, min_obs)
             for var in weights_log if var in roi}
    combo = []
    for i, row in enumerate(nav_series):
        hit = False
        for var, m in masks.items():
            w = weights_log.get(var, [])
            if i < len(w) and abs(w[i]) > 0.0 and bool(m.get(row["date"], False)):
                hit = True
                break
        combo.append(hit)
    rets = [float(r) for r in port_rets[:n]]
    n_ex = sum(combo)
    total = sum(rets)
    ex_ret = 0.0
    ex_nav = 1.0
    for r, m in zip(rets, combo):
        if m:
            ex_ret += r
        else:
            ex_nav *= (1.0 + r)
    return {"extreme_days": n_ex,
            "extreme_day_ratio": n_ex / n,
            "extreme_ret_share": (ex_ret / total) if abs(total) > 1e-12 else float("nan"),
            "ann_return_ex_extreme": (ex_nav ** (252.0 / n) - 1.0) if ex_nav > 0
                                     else float("nan")}


def hold_days_stats(weights_log: dict[str, list[float]]) -> dict:
    """同号串持仓统计（中短线 Phase 0d）。纯函数。

    每品种权重序列的"连续同号串"（0 仓断串、变号即前串结束开新串）：
    串长 = 每日重判框架下同向判定连续天数 ≈ 单次持仓天数。返回全品种
    **全部串**（含被窗口截断的尾串）的等权平均 / 最长 / 串数。
    截断偏短如实记录：样本末仍持仓的尾串被截断（均值略偏短）。
    分窗语义：段首 bars 含 train 全历史 → 方向自段首首记录日即延续，
    **同号串跨段自然连长**（段首只有重执行的换手成本、无强制空仓日；
    唯一强制空仓日 = 全样本首记录日，0 根 bar）。故串长度量的是拼接
    OOS 全轴上的连续判定串——长线（250 日动量串长数月）与短线候选
    （1c gate"≤10 交易日"）在同一口径下可直接对比。
    """
    runs: list[int] = []
    for ws in weights_log.values():
        cur_len, cur_sign = 0, 0
        for w in ws:
            s = 1 if w > 0 else (-1 if w < 0 else 0)
            if s == 0:                      # 空仓：断串
                if cur_len > 0:
                    runs.append(cur_len)
                cur_len, cur_sign = 0, 0
            elif s == cur_sign:             # 同号延续
                cur_len += 1
            else:                           # 新方向（含首日）：前串结算
                if cur_len > 0:
                    runs.append(cur_len)
                cur_sign, cur_len = s, 1
        if cur_len > 0:                     # 尾串（被窗口截断，照计）
            runs.append(cur_len)
    if not runs:
        return {"avg_hold_days": 0.0, "max_hold_days": 0, "n_runs": 0}
    return {"avg_hold_days": sum(runs) / len(runs),
            "max_hold_days": max(runs), "n_runs": len(runs)}


def _vol_parity_capped_weights(raw_dirs: dict[str, float], vols: dict[str, float],
                               budget: float, min_vol: float = 0.05,
                               cap: float = 0.25) -> dict[str, float]:
    """波动率平价权重 + 已知产业类别内权重份额硬封顶（旁路）。

    vol_parity 算完逐品种 |w| 后（w_i ∝ 1/σ_i，归一化到总|w|=budget）：
      1. 按 CATEGORY 把类别内 |w| 加总，除以总 |w| 得类别**份额**；
      2. 份额超 cap 的类别按比例压缩到 cap（释放预算）；
      3. 释放的预算按逆波动重新分给未满类别（迭代至所有类别份额 ≤ cap）。

    意义：把预算从高相关簇（黑色 RB/I/JM/J，§7 自承"有效独立下注仅 3-4 个"）
    强制搬到低相关类别腿（贵金属/有色/农产品），提升真实分散度。
    与 decorr_vol_parity（§11.1 被否）的区别：cap 是产业知识硬约束，不做收益
    协方差统计估计，无白化噪声腿反噬。cap 初值 0.25（5 类均分才 20%）。

    **份额口径（2026-10-15 第六步修正）**：cap 约束类别占总预算的份额，对任意
    budget 良定义（容量 = 类数×cap ≥ 100% 恒可行）。原实现把 cap 当**绝对权重**
    上限——budget=1.0 时两口径等价（单测因此未暴露），但生产 budget=
    max_leverage=1.5 > 5×0.25 时容量不可行，迭代停在中途：类别份额既未封住、
    结果依赖迭代次数。该缺陷版本曾被绩效口径评审否决；正确口径应为危机子样本
    的尾部指标，而非全样本夏普。

    旁路可选：默认 weight_method 仍 vol_parity，默认绩效零回归。
    """
    active = [k for k in raw_dirs if raw_dirs[k] != 0]
    if not active:
        return {}
    # 先按 vol_parity 求初始逐品种权重（带符号）
    base = _make_weights(raw_dirs, "vol_parity", vols, budget, min_vol)
    if not base:
        return base
    by_cat: dict[str, dict[str, float]] = {}
    for k in active:
        cat = CATEGORY.get(k, "其他")
        by_cat.setdefault(cat, {})[k] = base[k]
    cat_of = {k: c for c, d in by_cat.items() for k in d}
    weights = {k: v for d in by_cat.values() for k, v in d.items()}
    # 迭代压缩：类别份额（|w_c|/总|w|）超 cap 压到 cap，释放预算按逆波动补给
    # 严格未满类别。份额口径下容量 = 类数×cap ≥ 1 恒可行，快速收敛。
    for _ in range(10):
        gross = sum(abs(v) for v in weights.values())
        if gross <= 0:
            break
        cat_sum: dict[str, float] = {}
        for k, w in weights.items():
            cat = cat_of[k]
            cat_sum[cat] = cat_sum.get(cat, 0.0) + abs(w)
        cap_abs = cap * gross  # 份额上限换算回绝对权重（gross 守恒 = budget）
        over = {c for c, s in cat_sum.items() if s > cap_abs + 1e-12}
        if not over:
            break
        # 把超限类别压缩到 cap 份额，压缩释放的预算 delta 记下
        released = 0.0
        for c in over:
            factor = cap_abs / cat_sum[c]
            for k in by_cat[c]:
                new = weights[k] * factor
                released += abs(weights[k]) - abs(new)
                weights[k] = new
        if released <= 0:
            break
        # 将释放预算按逆波动补给严格未满类别（保持符号）
        under = [c for c in by_cat if cat_sum[c] < cap_abs - 1e-12]
        if not under:
            # M7 末端守恒：所有类别都触顶、无未满类别可吸收（cap < 1/类数的退化
            # 配置）——按各类逆波动比例回补到所有类别，保持符号，使总 |w| 回到
            # budget。守恒优先于 cap。
            pool = [k for k in weights if abs(weights[k]) > 0]
            invs = {k: 1.0 / max(vols.get(k, min_vol), min_vol) for k in pool}
            tot = sum(invs.values())
            if tot <= 0:
                break
            for k in pool:
                weights[k] += (1 if weights[k] >= 0 else -1) * (
                    released * invs[k] / tot)
            break
        und_inv = {c: sum(1.0 / max(vols.get(k, min_vol), min_vol) for k in by_cat[c])
                   for c in under}
        tot_inv = sum(und_inv.values())
        for c in under:
            add = released * (und_inv[c] / tot_inv)
            # M7 类内回补：按该类别内品种逆波动比例分配（与类别间分配同构），
            # 而非均分——低波动品种获得更多该类预算，避免失真。
            cat_inv = {k: 1.0 / max(vols.get(k, min_vol), min_vol) for k in by_cat[c]}
            ctot = sum(cat_inv.values())
            for k in by_cat[c]:
                sign = 1 if weights[k] >= 0 else -1
                weights[k] += sign * (add * cat_inv[k] / ctot)
    return weights


def _run_date_loop(data: dict[str, pd.DataFrame], roi: dict[str, pd.Series],
                   vol: dict[str, pd.Series], dates: list, strat, class_name: str,
                   need_carry: bool, carry_map: dict[str, pd.DataFrame],
                   margin_ratios: dict[str, float], risk_cfg: dict,
                   commission: float, slippage: float, weight_method: str,
                   rebalance_days: int, attribution: bool,
                   n_vars: int, roll_masks: dict[str, pd.Series] | None = None) -> dict:
    """对 dates 切片跑一遍「信号→组合→风控→累积 NAV」主循环（状态全新）。

    这是 run_futures_portfolio 逐日循环体的**可复用窗口化版本**：dates 是
    任意连续日期子序列（默认=全轴），每次调用从 nav=1.0、空持仓、全新风控
    开始跑。时序信号 on_bar 用"截至当日"的全历史 bars（train 段数据自动
    包含在 window 内），故从 test 段首日起即天然样本外、无预热问题。

    样本外分窗：对每个 test 段独立调本函数（状态互不污染），再在调用方把
    各段 NAV 首尾拼接成「全样本外 NAV」——每段是纯 test，跨窗口不混调仓
    状态（prev_exec_w/风控 vol 缓存都逐段重置）。

    返回聚合结构（与 run_futures_portfolio 原循环产出逐字一致，供 _stats）：
      nav_series / bench_series / port_ret_series / turnover_series /
      cost_series / margin_series / attribution(逐品种累计贡献) /
      weights_log / factor_attr_log(--attribution) / nav_final
    """
    bars_so_far: dict[str, pd.DataFrame] = {var: df.iloc[:0].copy()
                                            for var, df in data.items()}
    risk = FuturesRiskManager(risk_cfg)
    # 基准 budget（组合目标总|w|）：配置 max_leverage 或默认 1.0；无 max_leverage 用 1.0
    risk_budget = risk.max_leverage if risk.max_leverage else 1.0

    nav = 1.0
    nav_series, port_ret_series = [], []
    # 逐品种累计贡献（result["attribution"]）。原命名 attribution 会遮蔽同名布尔参数
    # （第四步测试抓出：遮蔽后 `if attribution:` 恒真，--attribution 开关失效——
    # 凡带 factor_legs 的策略在 attribution=False 跑法里也逐腿 on_bar + 簿记，
    # 白耗算力且 factor_attr_log 意外有值）。改名 var_attr，开关恢复生效。
    var_attr = {var: [] for var in data}
    weights_log = {var: [] for var in data}
    turnover_series, cost_series, margin_series = [], [], []
    prev_exec_w: dict[str, float] = {}
    days_since_rebal = 0
    # --attribution 虚拟净值：因子集从策略的 factor_legs 动态收集（新增因子自动出卡）
    factor_legs = getattr(strat, "factor_legs", {})
    factor_attr = {k: 1.0 for k in factor_legs}
    factor_attr_log: list[dict] = []
    factor_w_abs: dict[str, float] = {var: 0.0 for var in data}
    bench_nav = 1.0
    bench_series = []

    for t in range(1, len(dates)):
        dt = dates[t]
        prev_dt = dates[t - 1]
        is_rebal = (days_since_rebal % rebalance_days == 0)
        for var in data:
            sub = data[var]
            # S1 前视修复：信号只用 dt 之前（t-1 及以前）的 bar 判定 → 落 t-1 收盘，
            # 赚 dt 当天已走完前的未来收益（T+1 口径）。原 `<= dt` 用 t 日当天 settle
            # 判定信号、再去吃 dt 当天已实现收益 = 用 t 日收盘才知道的方向赚 t 日白天
            # 的收益（同日前视）。改 `< dt` 后首日（t=1）bars_so_far 为空 → 持 hold。
            # 归因层与主信号共用本 bars_so_far，一处修改两条路径（主信号+factor_attr）同步修正。
            mask = sub["date"] < dt
            bars_so_far[var] = sub[mask]
        raw_dirs: dict[str, float] = {}
        factor_scores: dict[str, dict] = {}  # --attribution 用：每品种各因子方向分
        # 组合级信号（如横截面动量）需先喂全品种 bars 算相对强弱（hasattr 守护，普通策略无感）
        if hasattr(strat, "set_universe"):
            strat.set_universe(bars_so_far)
        for var in data:
            bars = bars_so_far[var]
            if len(bars) < 2:
                raw_dirs[var] = 0.0
                continue
            cur = strat
            if hasattr(cur, "set_active_var"):
                cur.set_active_var(var)
            # carry 注入必须无条件（含空序列）：策略实例跨品种共享，跳过注入会让
            # 无源品种的 carry 腿保留上一品种注入的序列——吃到邻序品种的展期信号，
            # 且结果依赖 futures.symbols 顺序（第三步证据包实测 0.8pp 年化差）。
            # 空帧注入后 CarryStrategy.on_bar 判 empty → hold，与"无源=腿分 0"语义
            # 一致；无列空帧（异常兜底形态）不切片原样注入。
            if need_carry and var in carry_map:
                cm = carry_map[var]
                dfc = cm[cm["date"] <= dt] if (not cm.empty and "date" in cm.columns) else cm
                cur.inject_carry(dfc) if hasattr(cur, "inject_carry") else None
            try:
                sig = cur.on_bar(bars)
                raw_dirs[var] = {"long": 1.0, "short": -1.0}.get(sig.direction, 0.0)
            except Exception:
                raw_dirs[var] = 0.0
            # 逐因子归因：通用化——遍历策略 factor_legs 各腿方向分（新增因子自动出卡）
            if attribution and factor_legs:
                try:
                    factor_scores[var] = {}
                    for fk, leg in factor_legs.items():
                        if hasattr(leg, "set_active_var"):
                            leg.set_active_var(var)
                        sig = leg.on_bar(bars)
                        factor_scores[var][fk] = {"long": 1.0, "short": -1.0}.get(sig.direction, 0.0)
                except Exception:
                    factor_scores[var] = {fk: 0.0 for fk in factor_legs}
        if is_rebal:
            budget = risk_budget
            vols_now = {var: (vol[var].get(prev_dt) if vol[var].get(prev_dt) else None)
                        for var in data}
            vols_now = {k: (v if v and np.isfinite(v) else 0.0) for k, v in vols_now.items()}
            if weight_method == "decorr_vol_parity":
                weights = _decorr_vol_parity_weights(
                    raw_dirs, vols_now, budget, roi, window=60, min_vol=0.05)
            elif weight_method == "volparity_capped":
                weights = _vol_parity_capped_weights(
                    raw_dirs, vols_now, budget, min_vol=0.05, cap=0.25)
            else:
                weights = _make_weights(raw_dirs, weight_method, vols_now, budget, min_vol=0.05)
            if attribution:
                factor_w_abs = {}
                if weight_method in ("vol_parity", "decorr_vol_parity", "volparity_capped"):
                    inv = {k: 1.0 / max(vols_now.get(k, 0.05), 0.05) for k in data}
                    s = sum(inv.values())
                    factor_w_abs = {k: budget * (inv[k] / s) for k in data}
                else:
                    factor_w_abs = {k: budget / n_vars for k in data}
            prev_weights_for_cost = dict(prev_exec_w)
            w_s, margin_total = risk.apply(weights, margin_ratios, nav_t=nav)
            for k in list(prev_weights_for_cost.keys()):
                if k not in w_s:
                    prev_weights_for_cost[k] = 0.0
            cost = futures_trade_cost(prev_weights_for_cost, w_s, commission, slippage)
            allk = set(prev_exec_w) | set(w_s)
            turnover = sum(abs(w_s.get(k, 0.0) - prev_exec_w.get(k, 0.0)) for k in allk) / 2.0
        else:
            w_s = dict(prev_exec_w)
            margin_total = sum(abs(v) * margin_ratios.get(k, 0.0) for k, v in w_s.items())
            cost = 0.0
            turnover = 0.0
        days_since_rebal += 1
        port_ret = 0.0
        for var in data:
            r = roi[var].get(dt)
            if r is None or not np.isfinite(r) or np.isnan(r):
                r = 0.0
            # 换月日伪 PnL 剔除（仅 built 路径，X0 恒空 → 零影响）：只剔除归因/组合收益，不动信号
            if roll_masks and roll_masks.get(var, None) is not None and roll_masks[var].get(dt, False):
                r = 0.0
            wv = w_s.get(var, 0.0)
            port_ret += wv * r
            var_attr[var].append(wv * r)
            weights_log[var].append(wv)
        nav = nav * (1 + port_ret) * (1 - cost)
        nav_series.append({"date": dt, "nav": nav})
        port_ret_series.append(port_ret)
        turnover_series.append({"date": dt, "turnover": turnover})
        cost_series.append({"date": dt, "cost": cost})
        margin_series.append({"date": dt, "margin": margin_total})
        prev_exec_w = dict(w_s)
        if attribution:
            for var, sc in factor_scores.items():
                r = roi[var].get(dt)
                if r is None or not np.isfinite(r) or np.isnan(r):
                    r = 0.0
                wabs = factor_w_abs.get(var, 0.0)
                for k in factor_legs:
                    factor_attr[k] *= (1 + sc.get(k, 0.0) * wabs * r)
            factor_attr_log.append({"date": dt, "nav": nav,
                                    **{k: factor_attr[k] for k in factor_legs}})
        bench_ret = 0.0
        for var in data:
            r = roi[var].get(dt)
            if r is None or not np.isfinite(r) or np.isnan(r):
                r = 0.0
            if roll_masks and roll_masks.get(var, None) is not None and roll_masks[var].get(dt, False):
                r = 0.0  # 基准与组合同口径剔除换月伪收益
            bench_ret += (budget / n_vars) * r
        bench_nav *= (1 + bench_ret)
        bench_series.append({"date": dt, "nav": bench_nav})

    return {"nav_series": nav_series, "bench_series": bench_series,
            "port_ret_series": port_ret_series, "turnover_series": turnover_series,
            "cost_series": cost_series, "margin_series": margin_series,
            "attribution": var_attr, "weights_log": weights_log,
            "factor_attr_log": factor_attr_log, "nav_final": nav}


def _splice_nav(seg_results: list[dict]) -> dict:
    """把各 test 段结果拼接成全样本外序列（每段 NAV 起点=上一段终值）。

    段内 nav 从 1.0 起累计；拼接时乘上前面所有段的累计乘数 cum，得跨段连续 NAV。
    基准同法但用自己的累计乘数 cum_bench（零择时锚点独立复利，见函数体口径修正注释）。
    归因逐段追加（逐品种贡献可跨段求和）；权重/换手/成本/保证金逐段连排。
    返回与单段 `_run_date_loop` 同构的聚合结构。
    """
    out: dict = {
        "nav_series": [], "bench_series": [], "port_ret_series": [],
        "turnover_series": [], "cost_series": [], "margin_series": [],
        "attribution": None, "weights_log": None, "factor_attr_log": [],
        "nav_final": 1.0,
    }
    cum = 1.0
    cum_bench = 1.0
    attribution, weights_log = None, None
    for seg in seg_results:
        for k in ("nav_series", "bench_series", "port_ret_series",
                  "turnover_series", "cost_series", "margin_series"):
            out[k].extend(seg[k])
        out["factor_attr_log"].extend(seg["factor_attr_log"])
        if attribution is None:
            attribution = {var: list(v) for var, v in seg["attribution"].items()}
            weights_log = {var: list(v) for var, v in seg["weights_log"].items()}
        else:
            for var, lst in seg["attribution"].items():
                attribution[var].extend(lst)
            for var, lst in seg["weights_log"].items():
                weights_log[var].extend(lst)
        # 本段 nav/bench 前乘各自累计乘数，得全样本外连续值。
        # 第一步口径修正（2026-10-14）：基准此前误乘策略的 cum——拼接出的"基准"实为
        # "策略权益 × 段内基准增长"的混合曲线（段边界跳到策略权益），全样本
        # bench_ann / excess_annual 失真。基准是独立的零择时锚点，须按自身段终值复利链。
        seg_navs = [r["nav"] for r in seg["nav_series"]]
        seg_bench = [r["nav"] for r in seg["bench_series"]]
        seg_nav_final = seg_navs[-1] if seg_navs else 1.0
        seg_bench_final = seg_bench[-1] if seg_bench else 1.0
        for r, v in zip(out["nav_series"][-len(seg_navs):], seg_navs):
            r["nav"] = cum * v
        for r, v in zip(out["bench_series"][-len(seg_bench):], seg_bench):
            r["nav"] = cum_bench * v
        cum *= seg_nav_final
        cum_bench *= seg_bench_final
    out["attribution"] = attribution or {}
    out["weights_log"] = weights_log or {}
    out["nav_final"] = cum
    return out


def _per_window_summary(seg_results: list[dict]) -> list[dict]:
    """分窗模式下每测试段的摘要行（第一步绝对绩效基准用）。

    每段：起止日期 / 天数 / 段内策略终值 / 段内基准终值（各自从 1.0 起，段内独立）。
    日期取自 nav_series——attribution 关闭时 factor_attr_log 为空，此前 start/end
    会是 None；归因开启时两者同日期轴，取值不变（向后兼容 health 卡消费）。
    """
    out = []
    for seg in seg_results:
        navs = seg["nav_series"]
        bench = seg["bench_series"]
        out.append({
            "start": navs[0]["date"] if navs else None,
            "end": navs[-1]["date"] if navs else None,
            "n_days": len(navs),
            "nav_final": seg["nav_final"],
            "bench_final": bench[-1]["nav"] if bench else 1.0,
            "factor_attr_log": seg["factor_attr_log"],
        })
    return out


def run_futures_portfolio(cfg: dict, strat_name: str, contract: str | None = None,
                          weight_method: str = "vol_parity",
                          window: int = 60, verbose: bool = True,
                          rebalance_days: int | None = None,
                          attribution: bool = False,
                          train_days: int | None = None,
                          test_days: int | None = None,
                          step_days: int | None = None,
                          read_only: bool = False) -> dict:
    """多标的时间序列组合回测主函数。返回 {nav_df, stats, attribution, diagnostics}。

    rebalance_days: 组合再平衡频率（1=每日）。到调仓日才重算目标权重并计成本，
                    非调仓日沿用上期已执行权重、不计成本。None → 用 config futures.rebalance_days。
    attribution: 逐因子归因开关（TSMOM/Carry/Reversal 各自虚拟净值）。
                 逐日簿记翻倍故默认关，--attribution 启停。
    train_days/test_days/step_days: 显式样本外分窗。三者都给定（或至少 train+test）
        时，用 walk_forward._segment_dates 切窗，各 test 段独立跑后拼接成
        **全样本外 NAV**（每段从纯 test 起点、状态全新——含策略实例逐段重建，
        0c 段首状态重置；train 段只作信号的历史预热，无参数拟合）。
        缺省（都 None）→ 全轴单段，与历史行为逐字一致。
    read_only: cache-only（X0/carry/rules 命中即读、不拉网不写）。绝对基准与
        复核用——carry 缓存过期走的是 2018 起逐日全量重拉（数小时网络），
        复核跑基准不应触发；数据截至缓存末日（如 2026-09-23）。
    """
    fut = cfg.get("futures", {})
    # 第零步契约激活：contract=None → 读 config futures.contract（缺省 X0 零回归）；
    # 显式入参（各 CLI --contract）优先。此前该键全仓无消费者。
    contract = resolve_contract(cfg, contract)
    symbols = fut.get("symbols", ["RB0", "AU0"])
    variety = [s if s.endswith("0") else f"{s}0" for s in symbols]
    varieties = [v[:-1] for v in variety]

    # S3 起点截断：尊重 config futures.start（缺省 None → 不截断，与历史行为一致）
    start = fut.get("start")

    if rebalance_days is None:
        rebalance_days = int(fut.get("rebalance_days", 1))
    rebalance_days = max(1, int(rebalance_days))

    # 主连日线（8 列，含 settle）。contract: "X0"=新浪现成主连 / "built"=自建换月拼接
    data: dict[str, pd.DataFrame] = {}
    roll_masks: dict[str, pd.Series] = {}
    # 第二步口径 fail-loud：逐品种实际口径（built / X0-fallback / X0）。
    # build_contract_series 无合约可拼时静默回退 X0（fetcher 现已告警）——引擎侧
    # 再按归一化后帧有无 adj_close 判定并汇总，fallback 品种=信号/收益退回含换月
    # 跳空的原始口径（且无换月掩码剔除），必须可见。
    contract_source: dict[str, str] = {}
    for var in varieties:
        if contract == "built":
            df = build_contract_series(var, read_only=read_only)
            # 自建帧（adj_close/adj_return/main_contract/is_roll）归一化为引擎兼容 8 列：
            #   close = adj_close（前复权价），settle = 按 adj_return 累计的价水平（TSMOM
            #   用 settle 定方向，须隔跳空连续），volume＝hold 代理（engine 只用 close/settle）
            # 归一化抽为 fetcher_futures.normalize_built_frame（第零步：信号侧 daily_runner
            # 与回测同口径、同一函数）。注：部分新品种（AG/MA/M/Y/P/CF/SC）自建换月无
            # 数据、build 兜底返新浪 X0 原始 8 列帧（无 adj_close）——normalize 原样返回，
            # 两 schema 都兼容。
            if df is not None and not df.empty:
                df = normalize_built_frame(df)
                # 归一化后帧无 adj_close = build 兜底返回的 X0 原始帧（真 built 必有）
                contract_source[var] = (
                    "built" if "adj_close" in df.columns else "X0-fallback")
                # 换月日伪 PnL 剔除：保留 is_roll 并构造 roi 同日期的布尔掩码，
                # 供 _run_date_loop 在换月日把该品种归因收益置 0（不动信号方向）
                if "is_roll" in df.columns:
                    if "date" not in df.columns:
                        df["date"] = pd.to_datetime(df.index)
                    roll_masks[var] = pd.Series(
                        df["is_roll"].fillna(False).astype(bool).to_numpy(),
                        index=pd.to_datetime(df["date"]))
        else:
            df = get_futures_main_continuous(var, contract="X0",
                                             read_only=read_only)
            contract_source[var] = "X0"
            # M3：新浪 X0 无换月标记（未 back-adjust），合约切换 ~2% 跳空逐字计入生产 P&L，
            # built 反而剔除、X0 裸奔。新浪源不提供换月标记，用 built 自建连续（read_only=True
            # 读缓存为主，命中即用）的 is_roll 换月日对齐到 X0 交易日近似换月掩码。
            # 缓存未命中/异常 → 该品种换月剔除缺失（保险，不掉链子，不中断主流程）。
            if df is not None and not df.empty:
                try:
                    b = build_contract_series(var, read_only=True)
                    if b is not None and not b.empty and "is_roll" in b.columns:
                        bd = pd.to_datetime(b["date"])
                        roll_dates = set(bd[b["is_roll"].fillna(False)])
                        xd = pd.to_datetime(df["date"])
                        roll_masks[var] = pd.Series(
                            xd.isin(roll_dates).to_numpy(), index=xd)
                except Exception:
                    roll_masks[var] = None
        if df is not None and not df.empty:
            # S3：按 futures.start 截断起点（老品种早于 start 的历史进早窗会放大权重）
            if start is not None:
                st = pd.Timestamp(start)
                df = df[pd.to_datetime(df["date"]) >= st]
            if not df.empty:
                data[var] = df
    if not data:
        raise RuntimeError("无可用期货数据（品种主连拉取失败）")

    # 第二步口径 fail-loud 汇总：built 模式下兜底品种逐一点名（fetcher 侧已单品种
    # 告警，此处保证回测日志/结果里也能看到全量清单）。
    if contract == "built":
        fallback = [v for v, s in contract_source.items() if s == "X0-fallback"]
        if fallback:
            print(f"[WARN] futures.contract=built 但 {len(fallback)} 个品种自建换月"
                  f"无数据、已回退 X0 原始主连（含换月跳空、无换月掩码剔除）: "
                  f"{','.join(fallback)}", flush=True)

    strat_cfg = cfg["strategies"][strat_name]
    strat, class_name = build_strategy(strat_cfg)
    need_carry = _need_carry(class_name)

    carry_map: dict[str, pd.DataFrame] = {}
    if need_carry:
        for var in varieties:
            if var not in data:
                continue
            try:
                carry_map[var] = get_roll_yield_series(var, read_only=read_only)
            except Exception:
                carry_map[var] = pd.DataFrame()

    margin_ratios: dict[str, float] = {}
    try:
        rules = get_futures_rules(read_only=read_only)
        for var in varieties:
            r = rules.get(var, {})
            mr = r.get("margin_ratio")
            if mr:
                margin_ratios[var] = mr / 100.0
    except Exception:
        pass
    default_margin = float(cfg.get("futures", {}).get("margin", 0.1))
    for var in varieties:
        margin_ratios.setdefault(var, default_margin)

    risk_cfg = cfg.get("futures", {}).get("risk", {}) or {}

    cost_cfg = cfg.get("futures", {}).get("cost", {}) or {}
    commission = float(cost_cfg.get("commission", 0.0))
    slippage = float(cost_cfg.get("slippage", 0.0))

    # 对齐公共交易日轴（各品种日期并集，升序）
    all_dates = None
    for var, df in data.items():
        idx = pd.Series(pd.to_datetime(df["date"]).unique())
        all_dates = idx if all_dates is None else pd.concat([all_dates, idx]).drop_duplicates()
    dates = sorted(all_dates.tolist())

    roi = {var: _roi_series(data[var]) for var in data}
    vol = {var: _realized_vol(roi[var], window) for var in data}
    n_vars = len(data)

    # 逐日滚动回测主体：默认一段（全轴）/ 显式分窗（各 test 段独立 + 拼接）
    seg_results = []
    windowing = train_days is not None and test_days is not None
    if windowing and train_days and test_days:
        windows = _segment_dates(dates, train_days, test_days,
                                 step_days or test_days)
        if not windows:
            raise RuntimeError(
                f"分窗切片为空：train={train_days} test={test_days} 超过数据长度 {len(dates)}")
        for _train_dates, test_dates in windows:
            # 0c 段首状态重置：逐段重建策略实例。有状态策略（TimeStopWrapper 等：
            # 入场天数/方向记忆）不跨段泄漏；无状态策略重建后输出逐字一致——零回归
            # （此前的实例共享与 docstring"每段状态全新"的承诺不符，属文档级 bug）。
            strat, _ = build_strategy(strat_cfg)
            seg = _run_date_loop(data, roi, vol, test_dates, strat, class_name,
                                 need_carry, carry_map, margin_ratios, risk_cfg,
                                 commission, slippage, weight_method,
                                 rebalance_days, attribution, n_vars, roll_masks)
            seg_results.append(seg)
        merged = _splice_nav(seg_results)
    else:
        seg = _run_date_loop(data, roi, vol, dates, strat, class_name,
                             need_carry, carry_map, margin_ratios, risk_cfg,
                             commission, slippage, weight_method,
                             rebalance_days, attribution, n_vars, roll_masks)
        merged = seg
        seg_results = [seg]

    nav_series = merged["nav_series"]
    bench_series = merged["bench_series"]
    attribution = merged["attribution"]
    weights_log = merged["weights_log"]
    turnover_series = merged["turnover_series"]
    cost_series = merged["cost_series"]
    margin_series = merged["margin_series"]
    factor_attr_log = merged["factor_attr_log"]

    # 分窗模式：暴露每测试段逐因子归因（供治理管线算窗口稳定性）
    # + 段级摘要（起止/天数/策略与基准段内终值，第一步绝对绩效基准用）
    per_window = _per_window_summary(seg_results) if len(seg_results) > 1 else []

    nav_df = pd.DataFrame(nav_series)
    if nav_df.empty:
        return {"nav_df": nav_df, "stats": {}, "attribution": {}, "weights_log": {}}

    # 绩效（复用 walk_forward._stats；基准 = 无方向全多等权）
    stats = _stats(nav_df, bench_series, verbose)

    # 中短线 Phase 0a：极端日标注与敏感性。**追加键**，不改共享 _stats
    # （A股零回归）；口径见 _extreme_day_stats docstring。
    stats.update(_extreme_day_stats(merged, roi))
    # 中短线 Phase 0d：同号串持仓统计（追加键，同上零回归路线）
    stats.update(hold_days_stats(weights_log))
    if verbose:
        print(f"  极端日: {stats['extreme_days']} 天 / {stats['extreme_day_ratio']:.1%}"
              f"  剔除后年化 {stats['ann_return_ex_extreme']:+.2%}")
        print(f"  持仓: 平均 {stats['avg_hold_days']:.1f} 天 / 最长 {stats['max_hold_days']}"
              f" / {stats['n_runs']} 段（同号串口径，段末截断照计）")

    # 归因
    attr_ret: dict[str, float] = {}
    attr_vol: dict[str, float] = {}
    for var in data:
        arr = np.asarray(attribution[var], dtype=float)
        attr_ret[var] = float(np.nansum(arr))
        attr_vol[var] = float(np.nanstd(arr)) * np.sqrt(252) if len(arr) > 1 else 0.0

    if verbose:
        print("\n" + "=" * 50)
        print("  逐品种贡献归因")
        print("=" * 50)
        print(f"  {'品种':<6}{'阶段贡献':>12}{'年化波动':>12}")
        for var in data:
            print(f"  {var:<6}{attr_ret[var]:>12.2%}{attr_vol[var]:>12.2%}")

    # 逐因子归因（--attribution）：TSMOM-only / Carry-only 各自身价（相对无方向全多基准）
    factor_stats = None
    if attribution and factor_attr_log:
        f_df = pd.DataFrame(factor_attr_log)
        try:
            tm_stats = _stats(f_df[["date", "tsmom"]].rename(columns={"tsmom": "nav"}),
                              bench_series, False)
            cy_stats = _stats(f_df[["date", "carry"]].rename(columns={"carry": "nav"}),
                              bench_series, False)
            if "reversal" in f_df.columns:
                rev_stats = _stats(f_df[["date", "reversal"]].rename(columns={"reversal": "nav"}),
                                   bench_series, False)
            else:
                rev_stats = {}
        except Exception:
            tm_stats = cy_stats = rev_stats = {}
        factor_stats = {"tsmom": tm_stats, "carry": cy_stats, "reversal": rev_stats}
        if verbose:
            print("\n" + "=" * 50)
            print("  逐因子归因（TSMOM-only / Carry-only / Reversal-only，相对无方向全多基准）")
            print("=" * 50)
            for name, s in (("TSMOM", tm_stats), ("Carry", cy_stats), ("Reversal", rev_stats)):
                print(f"  {name:<6} 年化={s.get('ann_return','-'):>10}  "
                      f"夏普={s.get('sharpe','-'):>8}  累计={s.get('total_return','-'):>10}"
                      f"  超额/年={s.get('excess_annual','-'):>10}")

    return {"nav_df": nav_df, "stats": stats, "attribution": attr_ret,
            "attr_vol": attr_vol, "weights_log": weights_log,
            "turnover_series": pd.DataFrame(turnover_series),
            "cost_series": pd.DataFrame(cost_series),
            "margin_series": pd.DataFrame(margin_series),
            "factor_stats": factor_stats, "per_window": per_window,
            "contract_source": contract_source}


def output_paths(out_dir: Path, strategy: str) -> dict[str, Path]:
    """main() 产物路径（审计 #18 修复）：文件名带策略名。

    此前固定名 futures_walk_forward_nav.csv 等——futures_pipeline 三策略串行
    执行时前两个策略的产物被最后一个静默覆盖。带策略名后多策略互不覆盖、
    可横向对比（pipeline 透传 --out-dir 到 run-tag 子目录）。
    """
    return {
        "nav": out_dir / f"futures_walk_forward_{strategy}_nav.csv",
        "turnover": out_dir / f"futures_walk_forward_{strategy}_turnover.csv",
        "cost": out_dir / f"futures_walk_forward_{strategy}_cost.csv",
        "margin": out_dir / f"futures_walk_forward_{strategy}_margin.csv",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--strategy", default="tsmom_cta")  # 生产口径（2026-10-15 裁定 C：carry 下线）
    parser.add_argument("--contract", default=None,
                        help="主连口径 X0/built；默认读 config futures.contract（缺省 X0）")
    parser.add_argument("--rebalance-days", type=int, default=None,
                        help="组合再平衡频率（1=每日，21≈月频）；默认用 config futures.rebalance_days")
    parser.add_argument("--attribution", action="store_true",
                        help="逐因子归因（TSMOM-only/Carry-only 各自虚拟净值，逐日簿记翻倍）")
    parser.add_argument("--weight-method", default="vol_parity",
                        choices=["vol_parity", "equal", "decorr_vol_parity", "volparity_capped"])
    # 等权或波动率平价
    parser.add_argument("--train-days", type=int, default=None,
                        help="样本外分窗：训练段长度（天）。与 --test-days 同给才启用分窗")
    parser.add_argument("--test-days", type=int, default=None,
                        help="样本外分窗：测试段长度（天）")
    parser.add_argument("--step-days", type=int, default=None,
                        help="样本外分窗：步进（默认=test-days，无重叠滑窗）")
    parser.add_argument("--out-dir", default="reports",
                        help="产物根目录（默认 reports/；pipeline 透传 run-tag 子目录）")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # 若 futures.risk 未配 max_leverage，用 1.0 预算兜底
    result = run_futures_portfolio(cfg, args.strategy, contract=args.contract,
                                   weight_method=args.weight_method,
                                   rebalance_days=args.rebalance_days,
                                   attribution=args.attribution,
                                   train_days=args.train_days,
                                   test_days=args.test_days,
                                   step_days=args.step_days)
    if result["nav_df"].empty:
        print("回测无结果（可能无数据）")
        return

    # 保存 NAV 归因（#18：文件名带策略名，多策略串行互不覆盖）
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir, args.strategy)
    result["nav_df"].to_csv(paths["nav"], index=False)

    # 换手/成本/保证金时序落盘（若非空）
    for name, key in (("turnover", "turnover_series"), ("cost", "cost_series"),
                      ("margin", "margin_series")):
        dfx = result.get(key)
        if dfx is not None and not dfx.empty:
            dfx.to_csv(paths[name], index=False)
    print(f"\nNAV 已保存: {paths['nav']}")


if __name__ == "__main__":
    main()
