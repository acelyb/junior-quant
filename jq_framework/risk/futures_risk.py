"""
期货风控层：杠杆上限 + 单品种保证金上限 + 波动率目标 + 回撤降杠杆 + 保证金核算。

定位：与 A 股 risk/manager.py / vol_target.py / cppi.py 完全隔离（期货专属，不触碰
股票风险体系）。期货与股票风控的三处本质差异（见调研"杠杆下风控重设计"）：
  1. 总仓位杠杆上限（股票 w_max≤1 纯多头 → 期货多空杠杆上限 max_leverage）
  2. 保证金占用约束（|w_i|×margin_ratio_i 是硬约束，单边行情"波动×杠杆>预算就减仓"）
  3. 波动率目标重标定到"账户权益波动（×杠杆）"

第 4 步保证金预算：组合保证金总占用 Σ|w_i|·margin_i ≤ capital_utilization（对标
"保证金口径=约束+不计息"：组合保证金占用 ≤ 可用资金，超限整体等比缩仓）。闲置现金
不计息（不影响 nav 增益），保证金占用只通过缩仓→少赚间接影响 nav。

第 5 步回撤降杠杆（DD 协议，强化计划第五步）：vol-target 管波动幅度、杠杆/保证金管
瞬时暴露，唯独没有管"连续亏损序列"的防线。DD 协议补这条：净值自高水位回撤超
dd_entry → 全权重 × dd_scale，回补至 dd_exit 才恢复（滞回）。默认 enabled=false
（no-op，零回归），A/B 证据见 scripts/futures_dd_evidence.py。
"""
from __future__ import annotations

from collections import deque

import numpy as np


class FuturesRiskManager:
    """组合层期货风控：逐日 apply(weights, margin_ratios) → 缩放后权重 + 保证金核算。

    用法（futures_walk_forward 主循环每日一次）:
        rm = FuturesRiskManager(cfg)   # futures.risk 段
        w_scaled, margin_total = rm.apply(weights, margin_ratios, nav_t)

    weights:  {var: 带符号仓位}，如 {"RB": 1.0, "AU": -0.5}（long 正/short 负）
    margin_ratios: {var: 保证金比例}（如 0.11 = 11%）
    nav_t: t-1 收盘组合净值（vol-target 用；None → 跳过 vol-target）
    返回: (w_scaled, margin_total)。w_scaled 为带符号缩放后权重；margin_total=Σ|w|·margin。
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        # 杠杆上限：组合总 |w| ≤ max_leverage（None/<=0 → 不设限）
        ml = cfg.get("max_leverage")
        self.max_leverage = float(ml) if ml and float(ml) > 0 else None
        # 单品种保证金占用上限：|w_i|×margin_ratio_i ≤ max_single_margin
        ms = cfg.get("max_single_margin")
        self.max_single_margin = float(ms) if ms and float(ms) > 0 else None
        # 保证金预算：组合总保证金占用 ≤ capital_utilization（None/<=0 → 不设限）
        cu = cfg.get("capital_utilization")
        self.margin_budget = float(cu) if cu and float(cu) > 0 else None
        # 波动率目标（vol-target，与 VolTargetTimer 同范式，enabled=false → no-op）
        vt_cfg = cfg.get("vol_target") or {}
        self.enabled = bool(vt_cfg.get("enabled", False))
        self.sigma_target = float(vt_cfg.get("sigma_target", 0.10))
        self.window = int(vt_cfg.get("window", 60))
        self.w_max = float(vt_cfg.get("w_max", 1.0))
        self.w_min = float(vt_cfg.get("w_min", 0.20))
        # 回撤降杠杆（DD 协议，强化计划第五步；enabled=false → no-op 零回归）：
        # 组合净值自高水位回撤超过阈值 → 全权重按档缩放；回补至回撤 < dd_exit
        # 才恢复。两种形态（mode）：
        #   single（默认）单档滞回：回撤 > dd_entry → × dd_scale；[dd_exit, dd_entry]
        #     带内保持原状态（防临界区来回抖动）。
        #   linear 线性阶梯（主流机构"比例降险"）：scale = max(dd_floor, 1 − dd/dd_tol)
        #     ——回撤 5%/10%/15% ≈ ×0.75/×0.5/×0.25（tol=20% 时）。连续降险/渐进
        #     复险，无跳变故不需滞回（dd_exit 仅 single 模式使用）。
        dd_cfg = cfg.get("drawdown") or {}
        self.dd_enabled = bool(dd_cfg.get("enabled", False))
        self.dd_mode = str(dd_cfg.get("mode", "single"))
        if self.dd_mode not in ("single", "linear"):
            raise ValueError(f"drawdown.mode 须为 single/linear，got {self.dd_mode!r}")
        self.dd_entry = float(dd_cfg.get("dd_entry", 0.10))
        self.dd_exit = float(dd_cfg.get("dd_exit", 0.05))
        self.dd_scale = float(dd_cfg.get("dd_scale", 0.5))
        self.dd_tol = float(dd_cfg.get("dd_tol", 0.20))
        self.dd_floor = float(dd_cfg.get("dd_floor", 0.25))
        if self.dd_mode == "linear" and self.dd_tol <= 0:
            raise ValueError(f"linear 模式 dd_tol 须为正，got {self.dd_tol}")
        # 运行态：逐日组合净值收益序列（vol-target 用）
        self._prev_nav: float | None = None
        self._rets: deque[float] = deque(maxlen=self.window)
        # 运行态：高水位（DD 用；与 vol-target _prev_nav 同机制——引擎逐段新建
        # 管理器 → 段内连续、跨段重置，DD 以各段 nav=1.0 起步计量）+ 滞回开关
        # （single 模式用）/ 最近一档缩放（linear 模式 nav 无效时保持）
        self._hwm: float | None = None
        self._dd_active: bool = False
        self._dd_last_scale: float = 1.0
        # 诊断
        self._prev_margin_total: float | None = None

    # ---------- 风控各步 ----------

    def _leverage_clip(self, weights: dict[str, float]) -> dict[str, float]:
        """杠杆上限：组合总 |w| ≤ max_leverage，超出按比例全部缩。"""
        if self.max_leverage is None or not weights:
            return weights
        gross = sum(abs(w) for w in weights.values())
        if gross <= self.max_leverage:
            return weights
        scale = self.max_leverage / gross
        return {k: v * scale for k, v in weights.items()}

    def _per_margin_clip(self, weights: dict[str, float],
                         margin_ratios: dict[str, float]) -> dict[str, float]:
        """单品种保证金占用上限：|w_i|×margin_ratio_i ≤ max_single_margin，超出单腿缩。"""
        if self.max_single_margin is None or not weights:
            return weights
        out = dict(weights)
        for k in list(out.keys()):
            mr = margin_ratios.get(k)
            if not mr:
                continue
            used = abs(out[k]) * mr
            if used > self.max_single_margin:
                max_abs = self.max_single_margin / mr
                out[k] = np.sign(out[k]) * max_abs
        return out

    def _margin_budget_clip(self, weights: dict[str, float],
                            margin_ratios: dict[str, float]) -> dict[str, float]:
        """保证金预算（第 4 步）：组合总保证金占用 ≤ margin_budget，超出整体等比缩。

        与 _leverage_clip 同思路：整体等比缩放保住相对仓位，使
        margin_used = Σ|w_i|·margin_i ≤ margin_budget 恒成立。
        无 margin_budget (None/<=0) → no-op。
        """
        if self.margin_budget is None or not weights:
            return weights
        used = sum(abs(v) * margin_ratios.get(k, 0.0) for k, v in weights.items())
        if used <= self.margin_budget:
            return weights
        scale = self.margin_budget / used
        return {k: v * scale for k, v in weights.items()}

    def _vol_target_scale(self, nav_t: float | None) -> float:
        """波动率目标：返回总仓位缩放系数 w。enabled=false → 恒 1.0（no-op）。"""
        if not self.enabled:
            return 1.0
        if nav_t is None or not np.isfinite(nav_t) or nav_t <= 0:
            return self.w_min
        if self._prev_nav is not None and self._prev_nav > 0:
            self._rets.append(nav_t / self._prev_nav - 1.0)
        self._prev_nav = nav_t
        if len(self._rets) < max(5, self.window // 4):
            return 1.0  # 冷启动样本不足，满仓
        sigma_real = float(np.std(self._rets, ddof=1)) * np.sqrt(252)
        if sigma_real <= 0:
            return 1.0
        return float(np.clip(self.sigma_target / sigma_real, self.w_min, self.w_max))

    def _drawdown_scale(self, nav_t: float | None) -> float:
        """回撤降杠杆系数（enabled=false → 恒 1.0，no-op）。

        single（单档滞回）：回撤 > dd_entry → dd_scale；回补至 < dd_exit 恢复 1.0。
        滞回区间 [dd_exit, dd_entry] 内保持原状态。nav_t 无效（None/非有限/非正）
        时不更新状态、返回当前档位。
        linear（线性阶梯/比例降险）：scale = max(dd_floor, 1 − dd/dd_tol)，随当前
        回撤连续缩放（降险连续、复险渐进，无跳变故不需滞回）；nav_t 无效时保持
        最近一档。状态与 vol-target 同机制：apply 逐日喂 nav_t、引擎逐段新建
        管理器 → 段内连续、跨段重置（OOS 各段 nav 从 1.0 起步）。
        """
        if not self.dd_enabled:
            return 1.0
        if nav_t is None or not np.isfinite(nav_t) or nav_t <= 0:
            if self.dd_mode == "linear":
                return self._dd_last_scale
            return self.dd_scale if self._dd_active else 1.0
        if self._hwm is None or nav_t > self._hwm:
            self._hwm = nav_t
        dd = 1.0 - nav_t / self._hwm if self._hwm > 0 else 0.0
        if self.dd_mode == "linear":
            self._dd_last_scale = float(max(self.dd_floor, 1.0 - dd / self.dd_tol))
            return self._dd_last_scale
        if not self._dd_active and dd > self.dd_entry:
            self._dd_active = True
        elif self._dd_active and dd < self.dd_exit:
            self._dd_active = False
        return self.dd_scale if self._dd_active else 1.0

    # ---------- 主入口 ----------

    def apply(self, weights: dict[str, float], margin_ratios: dict[str, float],
              nav_t: float | None = None) -> tuple[dict[str, float], float]:
        """逐日风控主流程：vol-target → 杠杆上限 → 单品种保证金 → 保证金预算 → 核算。

        返回 (w_scaled, margin_total)。margin_total = Σ|w|·margin_ratio_i，
        第 4 步后必然 ≤ margin_budget（若有预算）。
        """
        # 1) 波动率目标：先缩总仓位（组合级，最高优先）
        w_scale = self._vol_target_scale(nav_t)
        # 1b) 回撤降杠杆（第五步 DD 协议）：组合级乘子，与 vol-target 独立叠加
        #     （两个乘子可交换，均作用于全权重；disabled → 恒 1.0 零回归）
        dd_scale = self._drawdown_scale(nav_t)
        w = {k: v * w_scale * dd_scale for k, v in weights.items()}
        # 2) 杠杆上限：组合总 |w| 超限按比例缩
        w = self._leverage_clip(w)
        # 3) 单品种保证金占用上限：逐腿缩
        w = self._per_margin_clip(w, margin_ratios)
        # 4) 保证金预算：组合总保证金占用超预算整体等比缩（约束+不计息口径）
        w = self._margin_budget_clip(w, margin_ratios)
        # 5) 保证金核算
        margin_total = sum(abs(v) * margin_ratios.get(k, 0.0) for k, v in w.items())
        self._prev_margin_total = margin_total
        return w, margin_total

    def diagnostics(self) -> dict:
        """诊断：杠杆、保证金占用、vol-target 系数等（None=未启用）。"""
        return {
            "max_leverage": self.max_leverage,
            "max_single_margin": self.max_single_margin,
            "margin_budget": self.margin_budget,
            "vol_enabled": self.enabled,
            "sigma_target": self.sigma_target if self.enabled else None,
            "dd_enabled": self.dd_enabled,
            "dd_mode": self.dd_mode,
            "dd_active": self._dd_active if self.dd_enabled else None,
            "prev_margin_total": self._prev_margin_total,
        }


def _smoke():
    """单元烟测：杠杆 clip、保证金越界断言、vol-target 高低波动触发。"""
    print("=== FuturesRiskManager 烟测 ===")

    # 1) 杠杆上限：总 |w|=2.0 > max_leverage=1.5 → 缩到 1.5，比例保持
    rm = FuturesRiskManager({"max_leverage": 1.5})
    w, mt = rm.apply({"RB": 1.2, "AU": 0.8},
                     {"RB": 0.11, "AU": 0.10})
    gross = sum(abs(x) for x in w.values())
    print(f"杠杆: RB={w['RB']:.3f} AU={w['AU']:.3f} gross={gross:.3f} margin_total={mt:.3f}")
    assert abs(gross - 1.5) < 1e-9, "杠杆超限应按比例降到 1.5"
    assert abs(w["RB"] / w["AU"] - 1.2 / 0.8) < 1e-6, "杠杆缩应保持原比例"
    # 杠杆不设限（None）→ 不缩
    rm0 = FuturesRiskManager({})
    w0, _ = rm0.apply({"RB": 1.5, "AU": 1.0}, {"RB": 0.11, "AU": 0.10})
    assert abs(sum(abs(x) for x in w0.values()) - 2.5) < 1e-9, "未设限不应缩"
    print("杠杆上限: PASS")

    # 2) 单品种保证金越界：RB margin 0.5×0.11=0.055 未超 0.30；构造高占比品种超限
    rm = FuturesRiskManager({"max_single_margin": 0.30})
    w, _ = rm.apply({"RB": 0.5}, {"RB": 0.11})
    assert abs(w["RB"] - 0.5) < 1e-9, "0.5×0.11=0.055<0.30 不应触发"
    w2, _ = rm.apply({"RB": 3.0}, {"RB": 0.11})  # 3.0×0.11=0.33 > 0.30 → 缩到 0.30/0.11
    assert abs(w2["RB"] - 0.30 / 0.11) < 1e-6, "保证金越界应缩到上限"
    print("单品种保证金上限: PASS")

    # 3) vol-target：高波动应触发减仓(w<1.0)，低波动应满仓
    navs_hi = [1.0]
    for _ in range(120):
        navs_hi.append(navs_hi[-1] * (1 + np.random.randn() * 0.03))
    rm = FuturesRiskManager({"max_leverage": 1.5,
                             "vol_target": {"enabled": True, "sigma_target": 0.10,
                                            "window": 60, "w_max": 1.0, "w_min": 0.20}})
    ws = [rm.apply({"RB": 1.0}, {"RB": 0.11}, nav_t=v)[0]["RB"] for v in navs_hi]
    active = [x for x in ws if x < 1.0]
    print(f"高波动: w范围={min(active):.3f}~{max(active):.3f} n_active={len(active)}/{len(ws)}")
    assert len(active) > 50, "高波动应触发 vol-target 减仓"
    assert min(active) >= 0.20 - 1e-9, "w 不应低于 w_min"

    # 低波动 → 满仓
    navs_lo = [1.0]
    for _ in range(120):
        navs_lo.append(navs_lo[-1] * (1 + np.random.randn() * 0.0003))
    rm2 = FuturesRiskManager({"vol_target": {"enabled": True, "sigma_target": 0.10,
                                             "window": 60, "w_max": 1.0, "w_min": 0.20}})
    ws_lo = [rm2.apply({"RB": 1.0}, {"RB": 0.11}, nav_t=v)[0]["RB"] for v in navs_lo]
    hot = [x for x in ws_lo if x >= 1.0 - 1e-9]
    print(f"低波动: 满仓占比={len(hot)}/{len(ws_lo)}")
    assert len(hot) > 50, "低波动应顶满仓"
    print("vol-target: PASS")

    # 4) no-op 门控：vol_enabled=false → 恒满仓，杠杆未设限
    rm3 = FuturesRiskManager({})
    w3, _ = rm3.apply({"RB": 1.2, "AU": 0.8}, {"RB": 0.11, "AU": 0.10}, nav_t=1.0)
    assert abs(w3["RB"] - 1.2) < 1e-9 and abs(w3["AU"] - 0.8) < 1e-9, "no-op 应不缩放"
    print("no-op 门控: PASS")

    # 5) 保证金预算（第 4 步）：组合总保证金占用超 capital_utilization → 整体等比缩
    rm = FuturesRiskManager({"capital_utilization": 0.5})
    w, mt = rm.apply({"RB": 3.0, "AU": 3.0}, {"RB": 0.11, "AU": 0.10})
    # used=3*0.11+3*0.10=0.63 > 0.5 → scale=0.5/0.63，比例 RB/AU=1.0/1.0 保持
    assert abs(mt - 0.5) < 1e-9, f"超预算应缩到 cap=0.5, got {mt}"
    assert abs(w["RB"] / w["AU"] - 1.0) < 1e-6, "预算缩应保持原比例"
    # 未设预算（None）→ 不缩
    rm0 = FuturesRiskManager({})
    w0, mt0 = rm0.apply({"RB": 3.0, "AU": 3.0}, {"RB": 0.11, "AU": 0.10})
    assert abs(mt0 - 0.63) < 1e-9, "未设预算不应缩"
    print("保证金预算约束: PASS")

    print("ALL PASS")


if __name__ == "__main__":
    _smoke()
