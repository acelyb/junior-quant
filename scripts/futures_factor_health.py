"""futures_factor_health.py — 期货时序因子个体体检表 + 治理决策（evaluate 型，非 predict 型）。

定位（docs §8.9 / plan 第 3 层）：
  治理对象 = **因子个体**（tsmom / carry / reversal），与因子集无关。
  新增因子 = 跑本脚本出一张健康卡，走同一条既有流水线，**不用全量重训**——
  正面消解"补因子白做"（ML 预测 w(t) 系在因子集上，加因子要重训）。

方法（逐因子、样本外）：
  1. 复用组合引擎 `run_futures_portfolio --attribution --train/--test/--step` 分窗，
     取每测试段逐因子虚拟 NAV（TSMOM-only/Carry-only/Reversal-only）。
  2. 对每个因子，统计每窗口：年化、夏普、超额；并算**窗口稳定性**——
     各测试段夏普符号一致率 + 全窗 IC 方向一致（正贡献窗口占比）。
  3. 治理规则（阈值驱动，非黑箱）：
       keep    样本外夏普>0 且 正贡献窗口占比≥2/3 且 单跑非灾难
       drop    样本外夏普≤0 或 符号翻转（正贡献窗口占比<1/3）或 单跑灾难(穿仓)
       reweight 其余 → 按样本外 IR/夏普幅度软调权重（低频、可解释、不系于因子集）
  4. "看穿因子马甲"诊断列（设计.md §10.3，2026-10-26；不改 decision）：
       残差检验   候选日收益对锚（tsmom）逐窗回归取残差，残差单独复利成策略——
                  残差夏普>0 = 锚之外有真东西；≤0 = 只是锚的影子
       尾部相关   候选最差 5% 日里锚同跌占比 − 无条件基线（lift）——
                  正 = 极端日共崩、负 = 极端日对冲（不用 Copula，简化版够用）

  产物：因子健康卡 CSV（research/futures_factor_health.csv）+ stdout 治理决策。

用法:
  python scripts/futures_factor_health.py --config config.yaml
  python scripts/futures_factor_health.py --train-days 504 --test-days 126 --step-days 126
  python scripts/futures_factor_health.py --suite tsmom_ens   # 第四步参数曲面：20/60/250 单点+集成 四卡
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from jq_framework.config.loader import load_config
from jq_framework.backtest.futures_walk_forward import run_futures_portfolio
from jq_framework.data.fetchers.fetcher_futures import resolve_contract

# 因子池从配置读取，不在此硬编码。
# 配置键：futures.factor_health.factors / extra_carriers
# 未配置时为空元组（外部用户需在 configs/local.yaml 里填自己的因子池）。
def _factor_pool(cfg: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    fh = (cfg.get("futures") or {}).get("factor_health") or {}
    return (tuple(fh.get("factors") or ()),
            tuple(fh.get("extra_carriers") or ()))

# ===== 套件（--suite，强化计划第四步引入）=====
# governance（默认）：因子全集治理卡——triple_factor 容器三腿 + 单因子载体，
#   行为与本改动前逐字一致（治理卡零回归）。
# tsmom_ens：lookback 参数曲面套件——tsmom_20/60/250 三单点 + 集成点，同一
#   tsmom_ens_cta 载体的 factor_legs 挂四腿 → 一次归因跑出四张卡。四腿进同一
#   L1O 组合 → 集成 vs 单点的 ΔIR 在同组合基准上可比（单点 250 的邻域证据）。
_SUITE_GOVERNANCE = "governance"
_SUITE_TSMOM_ENS = "tsmom_ens"
_SUITES = (_SUITE_GOVERNANCE, _SUITE_TSMOM_ENS)
_TSMOM_ENS_CARRIER = "tsmom_ens_cta"
_TSMOM_ENS_FACTORS = ("tsmom_20", "tsmom_60", "tsmom_250", "tsmom_ens")


def _window_lift(f_attr: list[dict], key: str) -> tuple[float, float] | None:
    """测试段内因子虚拟 NAV 的 (累计涨幅, 夏普口径 surrogate)。返回 None 若空。"""
    if not f_attr or all(key not in r for r in f_attr):
        return None
    series = pd.Series([r.get(key, 1.0) for r in f_attr])
    total = float(series.iloc[-1] / series.iloc[0] - 1.0) if len(series) > 1 else 0.0
    ret = series.pct_change().dropna()
    sr = float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 1 and ret.std() > 0 else 0.0
    return total, sr


# ===== "看穿因子马甲"检查（设计.md §10.3，2026-10-26 落地）=====
# 锚因子：残差回归与尾部相关的参照系。governance 套件 = tsmom（唯一生产 alpha）；
# tsmom_ens 套件 = tsmom_250（短回看点是否只是 250 的影子——第四步邻域证据的补强面）。
_TAIL_Q = 0.05  # 尾部相关：最差 5% 交易日


def _ref_factor(series_map: dict) -> str | None:
    """锚因子名：tsmom 优先，其次 tsmom_250（ens 套件），都没有 → None（跳过检查）。"""
    for cand in ("tsmom", "tsmom_250"):
        if cand in series_map:
            return cand
    return None


def _aligned_window_returns(fac_series: list, ref_series: list) -> list[pd.DataFrame]:
    """逐窗把因子与锚的 (date, nav) 对齐（内连接日期）→ (fac_ret, ref_ret) 日收益 DataFrame 列表。

    跨载体取序列（候选来自其载体跑、锚来自 triple_factor 容器跑）时日期轴可能有
    微小漂移，内连接只留共同日；窗内 <3 个有效收益对则弃（回归/占比都不可算）。
    """
    out = []
    for fs, rs in zip(fac_series, ref_series):
        if fs is None or rs is None:
            continue
        m = fs.merge(rs, on="date", suffixes=("_f", "_r"))
        if len(m) < 3:
            continue
        r = pd.DataFrame({"fac": m["nav_f"].pct_change(),
                          "ref": m["nav_r"].pct_change()}).dropna()
        if len(r) >= 3:
            out.append(r)
    return out


def residual_check(series_map: dict, fac: str, ref: str) -> dict | None:
    """残差检验——回答"这个因子是不是 tsmom 换了个马甲"。

    候选逐日收益对锚因子逐日收益做**逐窗** OLS 回归（r_fac = a + b·r_ref + eps，
    β/截距每窗独立估计——不跨窗借数据，无窗口间泄漏；窗内回归属诊断口径，
    β 含同窗信息，如实记录）。**"锚能解释的部分" = b·r_ref**——截距 a（α）与
    噪声都属"锚之外"，一并留在残差里：eps' = r_fac − b·r_ref。若把截距从残差
    中扣除，残差均值被构造性归零、夏普恒 0，检验失效（单测守护此语义）。
    残差 eps' 视作一个独立策略的日收益，复利成残差 NAV 后算年化/夏普：
      残差还能赚钱（夏普>0）= 因子里有锚之外的真东西；
      残差不赚钱 = 它只是锚的影子，不算独立信号。
    返回 {resid_ann, resid_sharpe, resid_beta}；样本不足/锚缺失 → None。
    """
    if fac == ref or fac not in series_map or ref not in series_map:
        return None
    wins = _aligned_window_returns(series_map[fac], series_map[ref])
    if not wins:
        return None
    resid: list[float] = []
    betas: list[float] = []
    for r in wins:
        x, y = r["ref"].to_numpy(dtype=float), r["fac"].to_numpy(dtype=float)
        if len(x) < 10 or x.std() == 0:
            continue
        X = np.column_stack([x, np.ones(len(x))])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid.extend((y - coef[0] * x).tolist())  # 只扣 β·r_ref，截距留在残差
        betas.append(float(coef[0]))
    if len(resid) < 60:  # 不足约一个测试窗的残差样本 → 不可判
        return None
    arr = np.asarray(resid)
    std = float(arr.std())
    sharpe = float(arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    nav = float(np.prod(1.0 + arr))
    ann = nav ** (252.0 / len(arr)) - 1.0
    return {"resid_ann": ann, "resid_sharpe": sharpe,
            "resid_beta": float(np.mean(betas))}


def tail_check(series_map: dict, fac: str, ref: str, q: float = _TAIL_Q) -> dict | None:
    """尾部相关——回答"暴跌那天它跟谁一起崩"。

    普通相关只描述平时；极端日的共崩/对冲行为它看不见（2007 量化地震模式）。
    做法（设计.md §10.3 简化版，不用 Copula——约 2000 天数据复杂方法会把噪声
    当规律）：挑候选因子最差 q 分位交易日，数这些天里锚因子收益也为负的占比
    （tail_cond_neg），与锚无条件负收益占比（tail_uncond_neg）对比：
      lift = cond − uncond 明显为正 = 极端日共崩；明显为负 = 极端日帮忙对冲。
    返回 {tail_cond_neg, tail_uncond_neg, tail_lift}；样本不足 → None。
    """
    if fac == ref or fac not in series_map or ref not in series_map:
        return None
    wins = _aligned_window_returns(series_map[fac], series_map[ref])
    if not wins:
        return None
    r = pd.concat(wins, ignore_index=True)
    if len(r) < 60:
        return None
    thresh = float(r["fac"].quantile(q))
    worst = r[r["fac"] <= thresh]
    if len(worst) < 10:
        return None
    cond = float((worst["ref"] < 0).mean())
    uncond = float((r["ref"] < 0).mean())
    return {"tail_cond_neg": cond, "tail_uncond_neg": uncond,
            "tail_lift": cond - uncond}


def _factor_series_by_window(wins: list[dict], fac: str) -> list:
    """每个测试段该因子的 (date, nav) 序列；该段无此因子 → None。供 L1O 组内贡献。"""
    series = []
    for w in wins:
        f_attr = w.get("factor_attr_log") or []
        rows = [r for r in f_attr if fac in r]
        if rows:
            series.append(pd.DataFrame({"date": [r["date"] for r in rows],
                                        "nav": [float(r[fac]) for r in rows]}))
        else:
            series.append(None)
    return series


def _combo_ir(window_series: list, factors: tuple, drop: str | None = None,
              rvol_window: int = 60) -> float:
    """逆波动缩放全集因子组合的样本外 IR（M8 因果滚动 std + M2 L1O 固定权重）。

    每窗口把 factors 中各因子 nav 对齐到共同日期 → 日收益；M8：每腿权重
    w_f(t) = 1/σ_f(t) 用**因果滚动已实现波动**（rolling std×√252，窗口 rvol_window）
    逐日估计、行内归一为等风险贡献 → 组成全组合日收益，跨窗拼接后算年化 IR。
    滚动 std 只用到 t 之前的数据，point-in-time 无缝（不用样本内 std 污染末日权重）。

    M2 / L1O：传 drop=<因子名> 时，保持全集（factors 全集中 available 的腿）权重
    不变、只把该腿收益贡献清零（Σ_{g≠drop} w_g r_g，**不重新归一化、不重算其余腿
    权重**）——ΔIR 反映干净的边际贡献而非重配权噪声腿。

    逆波动缩放是关键：各因子单跑波动差异很大（如 reversal 波动数倍于 tsmom），
    名义等权会被高波动腿主导 → 全组合 IR 失真。1/σ 缩放让每腿等风险贡献，
    全组合 IR 才是可信的分散化水平。
    """
    rets = []
    if not window_series or not factors:
        return 0.0
    n = len(window_series[0])
    for wi in range(n):
        # 本窗口内 avail_idx = 有数据的全集因子下标（即便 drop 某腿，权重仍按全集算）
        avail_idx = [i for i in range(len(factors))
                     if window_series[i][wi] is not None]
        if not avail_idx:
            continue
        included = [window_series[i][wi] for i in avail_idx]
        common = included[0]["date"]
        for df in included[1:]:
            common = common[common.isin(df["date"])]
        if len(common) < 3:
            continue
        cols = {}
        for i in avail_idx:
            sub = window_series[i][wi]
            sub = sub[sub["date"].isin(common)].set_index("date")["nav"]
            cols[factors[i]] = sub.pct_change()
        r = pd.DataFrame(cols).dropna()
        if len(r) < 3:
            continue
        # M8：因果滚动已实现波动 → 逐日 1/σ 权重（rolling 只用过去，point-in-time）
        vol = r.rolling(rvol_window, min_periods=max(3, rvol_window // 4)).std() * np.sqrt(252)
        w = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
        srow = w.sum(axis=1)
        r = r[srow > 0]
        w = w.div(srow.replace(0, np.nan), axis=0).fillna(0.0)
        w = w[srow > 0]
        if r.empty:
            continue
        # M2：drop 腿清零（权重按全集、不重归一；删的是收益贡献）
        if drop is not None and drop in w.columns:
            w = w.drop(columns=drop)
            r = r.drop(columns=drop)
        if r.shape[1] == 0:
            continue
        combo = r.mul(w, axis=1).sum(axis=1)
        rets.extend(combo.values)
    if len(rets) < 2:
        return 0.0
    arr = np.asarray(rets)
    std = arr.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(arr.mean() / std * np.sqrt(252))


def _cards_from_carrier(cfg, strat_name, factors, train_days, test_days, step_days,
                        contract, weight_method,
                        read_only: bool = False) -> tuple[list[dict], dict]:
    """对单个载体策略跑分窗归因，抽 factors 中每个因子的卡片 + 跨窗因子序列。

    返回 (rows, series_map)：
      rows       — 每因子一行单跑健康卡（OOS 累计/夏普/正贡献窗/原 keep-drop-reweight 决策）。
      series_map — {fac: window_series}，window_series[w] = 第 w 测试段该因子 (date, nav)
                   序列或 None。供 health_card 统一做 L1O 增量 IR（组内贡献）。
    全程同一分窗引擎、同口径。read_only=cache-only（不拉网不写缓存，复核跑法）。
    """
    res = run_futures_portfolio(
        cfg, strat_name, contract=contract, weight_method=weight_method,
        attribution=True, train_days=train_days, test_days=test_days,
        step_days=step_days, verbose=False, read_only=read_only)
    wins = res.get("per_window", [])
    if not wins:
        return [], {}
    rows = []
    series_map: dict = {}
    for fac in factors:
        wl = [w for w in wins if _window_lift(w["factor_attr_log"], fac) is not None]
        tot, sr = _window_lift(
            [r for w in wins for r in w["factor_attr_log"]], fac) or (0.0, 0.0)
        if not wl:
            continue
        win_signs = [1.0 if ((s or 0.0) > 0) else 0.0
                     for w in wl for t, s in [_window_lift(w["factor_attr_log"], fac)]]
        pos_ratio = float(np.mean(win_signs)) if win_signs else 0.0
        # 原单跑治理决策（保留作参考；L1O 增量 IR 列作决策依据，见 health_card）
        # M1 校准到文档口径（keep 需正贡献窗≥2/3）：
        #   sr<=0 或 pos_ratio<1/3            → drop（样本外不成立或符号翻转）
        #   pos_ratio>=2/3 且 sr>0            → keep
        #   [1/3,2/3)（边界段，弱正贡献窗口）→ reweight
        if sr <= 0 or pos_ratio < 1 / 3:
            decision, reason = "drop", f"单跑夏普{sr:+.2f}≤0 或 正贡献窗口{pos_ratio:.0%}<1/3"
        elif pos_ratio < 2 / 3:
            decision, reason = "reweight", f"正贡献窗口{pos_ratio:.0%}∈[1/3,2/3) 弱门控 → 参考降权"
        elif tot < 0:
            decision, reason = "reweight", f"单跑夏普{sr:+.2f}>0 但累计{tot:+.2%}负 → 降权"
        else:
            decision, reason = "keep", f"单跑夏普{sr:+.2f}>0 且 正贡献窗口{pos_ratio:.0%}≥2/3"
        rows.append({
            "factor": fac, "contract": contract, "weight_method": weight_method,
            "windows": len(wl),
            "oos_total": tot, "oos_sharpe": sr,
            "win_pos_ratio": pos_ratio,
            "single_decision": decision, "single_reason": reason,
        })
        series_map[fac] = _factor_series_by_window(wins, fac)
    return rows, series_map


def check_window_consistency(series_map: dict) -> None:
    """治理硬断言（守护 S2/S3 口径）：归因窗口数/NAV 质量不达标 → fail-fast 不产出伪卡。

    s1) 各因子跨窗序列数须一致（同窗数），否则 L1O 组合里某因子窗口缺失被静默当 None，
        ΔIR 分母不同、不可比。
    s2) 因子 NAV 无空洞：同窗内日期严格连续递增、无 NaN/Inf、基准恒正，
        否则权重/收益拼接引入空值噪声，ΔIR 失真。
    违反即 raise AssertionError（宁可跑挂也不静默产出不可比结果）。
    """
    nw = {fac: len(series_map[fac]) for fac in series_map}
    assert len(set(nw.values())) == 1, (
        f"因子窗口数不一致（Δ不可比）：{nw}。请检查分窗归因完整性，勿静默跳过。")
    for fac, wins in series_map.items():
        for wi, win in enumerate(wins):
            if win is None:
                continue
            dt = pd.to_datetime(win["date"])
            assert dt.is_monotonic_increasing and dt.nunique() == len(dt), (
                f"因子 {fac} 窗 {wi} NAV 日期非严格递增（有重复/乱序）：请修复归因输出")
            nav = win["nav"].to_numpy(dtype=float)
            assert np.isfinite(nav).all(), (
                f"因子 {fac} 窗 {wi} NAV 含 NaN/Inf（空洞）：请修复归因输出")
            assert (win["nav"].iloc[-1] > 0) and (win["nav"].iloc[0] > 0), (
                f"因子 {fac} 窗 {wi} NAV 含非正基准（起始/末值≤0），收益率无法定义")


def health_card(cfg: dict, train_days: int, test_days: int, step_days: int,
                contract: str, weight_method: str,
                suite: str = _SUITE_GOVERNANCE,
                read_only: bool = False) -> pd.DataFrame:
    """逐因子跑分窗归因，产出健康卡（每因子一行：单跑参考 + L1O 增量 IR 组内贡献）。

    方案 B：单跑卡保留作参考；**决策依据 = L1O 增量 IR**——对每个因子 f，
    比较「全因子等权组合 IR」vs「去掉 f 后的组合 IR」，Δ=组内边际贡献。
    只有删掉 f 组合才变差（Δ>0）的因子才保留；Δ≈0（冗余）或 Δ<0（拖累）→ drop/降权。

    suite：governance（默认，因子全集）| tsmom_ens（lookback 参数曲面四点，
    见 _SUITE_TSMOM_ENS 注释——同一 L1O 组合基准上对比集成 vs 单点）。
    read_only：cache-only 不拉网（复核跑法；默认 False 与历史行为一致）。
    """
    # 1) 收集全因子跨窗序列 + 单跑卡
    #    因子池来自配置（futures.factor_health），未配置时为空
    _factors, _extra_carriers = _factor_pool(cfg)
    if suite != _SUITE_TSMOM_ENS and not _factors:
        print("  ⚠️ 未配置 futures.factor_health.factors，跳过治理卡。\n"
              "     在配置里加上:\n"
              "       futures:\n"
              "         factor_health:\n"
              '           factors: ["<因子1>", "<因子2>"]\n'
              '           extra_carriers: ["<策略名>"]')
        return rows, series_map
    rows: list[dict] = []
    series_map: dict = {}
    if suite == _SUITE_TSMOM_ENS:
        r, sm = _cards_from_carrier(
            cfg, _TSMOM_ENS_CARRIER, _TSMOM_ENS_FACTORS, train_days, test_days,
            step_days, contract, weight_method, read_only=read_only)
        rows.extend(r); series_map.update(sm)
    else:
        r, sm = _cards_from_carrier(
            cfg, "triple_factor_cta", _factors, train_days, test_days, step_days,
            contract, weight_method, read_only=read_only)
        rows.extend(r); series_map.update(sm)
        # 额外单因子载体：其因子名即卡片名（如 low_vol_cta → low_vol）
        for carrier in _extra_carriers:
            scfg = cfg.get("strategies", {}).get(carrier)
            if not scfg:
                continue
            cls_name = scfg.get("class", "")
            clss = getattr(__import__(scfg["module"], fromlist=["x"]), cls_name)
            factor_name = getattr(clss, "factor_name", carrier)
            r, sm = _cards_from_carrier(
                cfg, carrier, (factor_name,), train_days, test_days, step_days,
                contract, weight_method, read_only=read_only)
            rows.extend(r); series_map.update(sm)
    if not rows:
        print("无分窗归因结果（分窗未启用或数据过短）")
        return pd.DataFrame()

    # 治理硬断言：窗口数一致性 + NAV 无空洞（见 check_window_consistency），坏输入 fail-fast。
    check_window_consistency(series_map)

    # 2) 单跑识别穿仓级灾难腿：直接 drop，**不进 L1O 健康组合**。
    #    业界 L1O 的前提是基础组合健康；把单跑穿仓的腿混进基础组合会污染所有 Δ。
    #    灾难判据：样本外累计 < -30%（超过账户规模近 1/3 的亏损 = 穿仓级）。
    for row in rows:
        if row["oos_total"] < -0.30:
            row.setdefault("l1o_full_ir", 0.0)
            row["l1o_delta"] = float("nan")
            row["decision"] = "drop"
            row["reason"] = f"单跑样本外累计 {row['oos_total']:+.1%} 穿仓级灾难 → 直接 drop（不进 L1O）"

    # 3) L1O 增量 IR：在健康组合上算。全因子等权→逆波动缩放组合 vs 去掉该因子。
    healthy = [f for f in list(series_map.keys())
               if next((r["oos_total"] for r in rows if r["factor"] == f), 0.0) >= -0.30]
    if not healthy:
        return pd.DataFrame(rows)
    combo_series = [series_map[f] for f in healthy]
    # M2：全集反波权重固定，L1O 只清零被删腿的收益贡献（不重归一化、不重算其余腿权重），
    # 故 ΔIR = 干净的组内边际贡献（旧实现重算子集权重 → Δ 混入重配权+噪声腿效应）。
    full_ir = _combo_ir(combo_series, healthy)
    l1o = {}
    for fac in healthy:
        ir_rest = _combo_ir(combo_series, healthy, drop=fac)
        l1o[fac] = {"full_ir": full_ir, "rest_ir": ir_rest,
                    "delta": full_ir - ir_rest}  # 组内贡献：删掉 f 后 IR 损失

    # 4) 综合决策：L1O 增量为主（健康因子），单跑卡作参考
    for row in rows:
        fac = row["factor"]
        if "decision" in row and row["decision"] == "drop" and "reason" in row and row["reason"].startswith("单跑样本外"):
            row["l1o_full_ir"] = round(full_ir, 3) if healthy else 0.0
            continue
        li = l1o.get(fac)
        if not li:
            row["l1o_full_ir"], row["l1o_rest_ir"], row["l1o_delta"] = 0.0, 0.0, 0.0
            row["decision"], row["reason"] = row["single_decision"], row["single_reason"]
            continue
        row["l1o_full_ir"] = round(li["full_ir"], 3)
        row["l1o_rest_ir"] = round(li["rest_ir"], 3)
        row["l1o_delta"] = round(li["delta"], 3)
        d = li["delta"]
        if d >= 0.02:
            decision, reason = "keep", f"组内增量 IR {d:+.3f}（删掉组合变差）"
        elif d > -0.02:
            decision, reason = "reweight", f"组内增量 IR {d:+.3f}（近零=冗余或中性）→ 参考单跑"
        else:
            decision, reason = "drop", f"组内增量 IR {d:+.3f}（删掉组合 IR 反升=拖累）"
        row["decision"] = decision
        row["reason"] = f"L1O:" + reason + f"｜单跑参考:{row['single_decision']}"

    # 5) "看穿因子马甲"检查（设计.md §10.3，锚=tsmom/tsmom_250）：
    #    残差年化/夏普/β + 最差 5% 日条件负占比。锚自身与样本不足者置 NaN。
    #    纯诊断列，不改 decision——马甲识别供用户综合裁决（如 xsmom L1O keep
    #    而残差不赚钱 = "对冲价值非独立 alpha"的又一佐证）。
    ref = _ref_factor(series_map)
    for row in rows:
        if ref is None or row["factor"] == ref:
            row["resid_ann"] = row["resid_sharpe"] = row["resid_beta"] = float("nan")
            row["tail_cond_neg"] = row["tail_lift"] = float("nan")
            continue
        rc = residual_check(series_map, row["factor"], ref)
        if rc:
            row["resid_ann"] = round(rc["resid_ann"], 4)
            row["resid_sharpe"] = round(rc["resid_sharpe"], 2)
            row["resid_beta"] = round(rc["resid_beta"], 3)
        else:
            row["resid_ann"] = row["resid_sharpe"] = row["resid_beta"] = float("nan")
        tc = tail_check(series_map, row["factor"], ref)
        if tc:
            row["tail_cond_neg"] = round(tc["tail_cond_neg"], 3)
            row["tail_lift"] = round(tc["tail_lift"], 3)
        else:
            row["tail_cond_neg"] = row["tail_lift"] = float("nan")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--contract", default=None, choices=["X0", "built"],
                        help="主连口径；默认读 config futures.contract（缺省 X0）")
    parser.add_argument("--weight-method", default="vol_parity",
                        choices=["vol_parity", "equal"])
    parser.add_argument("--suite", default=_SUITE_GOVERNANCE, choices=list(_SUITES),
                        help="因子套件：governance=因子全集治理卡（默认）| "
                             "tsmom_ens=lookback 参数曲面（20/60/250 单点+集成四卡）")
    parser.add_argument("--read-only", action="store_true",
                        help="cache-only 不拉网不写缓存（复核跑法）")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 第零步契约激活：--contract 缺省读 config futures.contract（缺省 X0）
    contract = resolve_contract(cfg, args.contract)
    df = health_card(cfg, args.train_days, args.test_days, args.step_days,
                     contract, args.weight_method, suite=args.suite,
                     read_only=args.read_only)
    if df.empty:
        print("无健康卡产出")
        return
    out = args.out or (
        "research/futures_tsmom_ensemble_health.csv"
        if args.suite == _SUITE_TSMOM_ENS else "research/futures_factor_health.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{'='*70}\n  期货因子健康卡（样本外分窗，{datetime.now():%Y-%m-%d %H:%M}）\n{'='*70}")
    print(f"  {'因子':<8}{'窗口':>5}{'OOS累计':>10}{'OOS夏普':>9}  {'L1O组内IR':>10}{'ΔIR':>7}  "
          f"{'残差年化':>9}{'残差夏普':>9}{'β':>6}{'尾部lift':>9}  决策")
    for _, r in df.iterrows():
        def _fmt(v, pct=False, digits=2):
            if pd.isna(v):
                return "    —"
            return f"{v:+.1%}" if pct else f"{v:+.{digits}f}"
        print(f"  {r['factor']:<8}{r['windows']:>5}{r['oos_total']:>9.1%}"
              f"{r['oos_sharpe']:>9.2f}  {r['l1o_full_ir']:>10.3f}{r['l1o_delta']:>7.3f}  "
              f"{_fmt(r['resid_ann'], pct=True):>9}{_fmt(r['resid_sharpe']):>9}"
              f"{_fmt(r['resid_beta'], digits=3):>6}{_fmt(r['tail_lift'], digits=3):>9}  "
              f"→ {r['decision']}: {r['reason']}")
    ref_note = "tsmom" if "tsmom" in set(df["factor"]) else "tsmom_250"
    print(f"\n（决策依据 = L1O 增量 IR：全因子等权组合 vs 去掉该因子，Δ>0 保留、Δ≈0 冗余、Δ<0 拖累）")
    print(f"（残差/尾部为纯诊断列，不改 decision：残差 = 候选日收益对 {ref_note} 逐窗回归的残差策略"
          f"（夏普>0 = 锚之外有真东西）；尾部 lift = 候选最差 {_TAIL_Q:.0%} 日里 {ref_note} 同跌占比 − 无条件基线"
          f"（正 = 极端日共崩、负 = 极端日对冲））")
    print(f"\n健康卡已保存: {out}")


if __name__ == "__main__":
    main()
