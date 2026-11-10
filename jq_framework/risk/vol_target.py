"""
仓位层波动率目标（Vol-Target）实时缩放器 —— 方案B（Moreira & Muir 2017）。

定位（与 risk/cppi.py 同层）：仓位层非选股层，与 factors/ 截面选股正交。
- 选股层定"买什么"（截面排序）
- 本模块定"配多少"（时间序列，逐日对组合总仓位施加系数 w∈[w_min, w_max]）

机制：
  σ_realized(t-1) = 过去 W 日组合日收益 std × √252（年化已实现波动率）
  w(t) = clip(σ_target / σ_realized(t-1), w_min, w_max)
  port_ret(t) *= w(t)

与 CPPI 的根本差异：
  CPPI 看净值水平（cushion=V-floor），跌得多→减仓（反应性保底，管"损失预算"轴）。
  Vol-Target 看波动率（σ_target/σ_realized），波动大→减仓（预防性控风险水平，管"风险水平"轴）。
  两者正交可串联取 min：CPPI 在深跌后兜底，Vol-Target 在波动飙升时提前减仓。

为什么与 CPPI 同样必须逐日 update 而非预计算查表：
  σ_realized 依赖组合已实现日收益序列，而组合日收益 = 持仓权重 × 个股收益，持仓在调仓日
  才变、个股收益逐日变 → 组合日收益逐日变 → σ_realized 逐日变。但与 CPPI 不同的是，
  σ_realized 用的是"满仓组合"的已实现波动（不受本缩放器自身 w 影响），故理论上可预计算
  满仓 port_ret 序列后查表。本实现仍走逐日 update 路径，原因：①与 CPPI 同接口同接入点
  统一、②串联 CPPI 时若取 min 需实时、③保持"缩放器不碰选股只缩总仓"的架构对称。
  逐日 update 用满仓 port_ret 的滚动 σ（不受 w 反馈），无未来函数：σ_realized(t-1) 只用
  到 t-1 及之前收益，update 在 t 日 port_ret 计算前调用。

防未来函数：update 在 walk_forward 算完 t-1 日 port_ret、更新 nav 后、算 t 日 port_ret 前
调用。σ 用 nav 累计收益序列的最近 W 个日收益（nav 是含历史 w 缩放的真实净值，反映真实
持仓波动；若用满仓 port_ret 则需额外维护满仓 nav 序列，此处用真实 nav 的日收益更稳且无歧义）。

三个旋钮：
  σ_target：目标年化波动（如 0.10=10%），σ_realized 高于它则减仓。
  window：已实现波动率回看窗口（日，默认 60，半衰期加权可选后续加）。
  w_min/w_max：仓位上下限。w_min>0 防波动率飙升时归零踏空（与 CPPI cash-lock 同理）。

四硬约束（与 risk/cppi.py / risk/timing.py 同）：
- Windows：纯标准库 + numpy，无文件 I/O
- 正确性：单实例有可变状态（收益序列 deque），每 run_walk_forward 一个实例跨窗口延续，
  线程内顺序调用无并发写
- 资源：update O(window)，回测主循环每日一次，零额外开销
"""
from __future__ import annotations

from collections import deque

import numpy as np


class VolTargetTimer:
    """组合层波动率目标缩放器：回测中逐日 update(nav_t) 实时算总仓位系数 w。

    用法：
        timer = VolTargetTimer(cfg)   # cfg=None 或 enabled=false → no-op（恒返 1.0）
        w = timer.update(nav_t)       # 回测主循环每日调用，nav_t=t-1 收盘净值
        port_ret *= w                 # 缩总仓位，现金 (1-w) 收益为 0

    门控：enabled=false 时 update 恒返 1.0（零开销，不破回归门，与 CPPITimer/MarketTimer 同范式）。
    状态（收益序列 deque）跨窗口延续：run_walk_forward 内创建一次，nav 跨窗口累积不重置，
    故收益序列也跨窗口累积——σ_realized 用跨窗口的滚动收益，窗口切换处不重置（与 CPPI hwm 跨窗口同理）。
    """

    def __init__(self, cfg: dict | None = None):
        """cfg: multifactor.vol_target 字典（config.yaml）。None 或 enabled=false → no-op。"""
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.sigma_target = float(cfg.get("sigma_target", 0.10))
        self.window = int(cfg.get("window", 60))
        self.w_max = float(cfg.get("w_max", 1.0))
        self.w_min = float(cfg.get("w_min", 0.20))
        # 运行态：逐日 nav 收益序列（跨窗口延续）
        self._prev_nav: float | None = None
        self._rets: deque[float] = deque(maxlen=self.window)
        # 诊断
        self._n_calls = 0
        self._w_sum = 0.0
        self._w_min_seen = self.w_max
        self._w_max_seen = self.w_min

    def update(self, nav_t: float) -> float:
        """用 t-1 收盘净值 nav_t 更新已实现波动率并算 t 日总仓位系数 w（无未来函数）。

        enabled=false → 恒返 1.0（no-op）。nav_t<=0 → 返回 w_min（净值归零保守最低仓）。
        收益序列不足 window 时（冷启动）返 1.0 满仓——样本不足估 σ 不可靠，不缩仓优于乱缩。
        """
        if not self.enabled:
            return 1.0
        if nav_t is None or not np.isfinite(nav_t) or nav_t <= 0:
            return self.w_min
        # 累计日收益（用 t-1 → t 的真实净值收益，跨窗口延续）
        if self._prev_nav is not None and self._prev_nav > 0:
            self._rets.append(nav_t / self._prev_nav - 1.0)
        self._prev_nav = nav_t
        # 冷启动：样本不足返满仓
        if len(self._rets) < max(5, self.window // 4):
            return 1.0
        sigma_real = float(np.std(self._rets, ddof=1)) * np.sqrt(252)
        if sigma_real <= 0:
            return 1.0
        w = self.sigma_target / sigma_real
        w = float(np.clip(w, self.w_min, self.w_max))
        # 诊断累计
        self._n_calls += 1
        self._w_sum += w
        self._w_min_seen = min(self._w_min_seen, w)
        self._w_max_seen = max(self._w_max_seen, w)
        return w

    def coverage(self) -> tuple[int, float, float, float]:
        """诊断：返回 (调用次数, 平均仓位, 最低仓位, 最高仓位)。未启用返 (0,1,1,1)。"""
        if not self.enabled or self._n_calls == 0:
            return (0, 1.0, 1.0, 1.0)
        n = self._n_calls
        return (n, self._w_sum / n, self._w_min_seen, self._w_max_seen)


def _smoke():
    """单元烟测：构造已知 nav 序列验证 update 行为。"""
    # 高波动序列：日收益 ±5% 交替 → 年化 ~79%，σ_target=0.10 应大幅减仓
    np.random.seed(0)
    navs = [1.0]
    for _ in range(120):
        navs.append(navs[-1] * (1 + np.random.randn() * 0.03))  # 3% 日波动
    t = VolTargetTimer({"enabled": True, "sigma_target": 0.10, "window": 60,
                        "w_max": 1.0, "w_min": 0.20})
    ws = [t.update(v) for v in navs]
    active = [w for w in ws if w < 1.0]
    print(f"nav range: {min(navs):.3f}~{max(navs):.3f}")
    print(f"w range (active): {min(active):.3f}~{max(active):.3f}, n_active={len(active)}/{len(ws)}")
    cov = t.coverage()
    print(f"coverage: n={cov[0]} avg={cov[1]:.3f} min={cov[2]:.3f} max={cov[3]:.3f}")
    # 高波动下应触发减仓（w<1.0）
    assert len(active) > 50, "高波动应触发减仓"
    assert cov[2] >= 0.20 - 1e-9, "w 不应低于 w_min"
    assert cov[3] <= 1.0 + 1e-9, "w 不应高于 w_max"
    print("减仓触发: PASS")

    # no-op 门控
    t0 = VolTargetTimer({"enabled": False})
    assert all(t0.update(v) == 1.0 for v in navs), "disabled must return 1.0"
    print("no-op: PASS")

    # 冷启动：前若干点应返 1.0
    t1 = VolTargetTimer({"enabled": True, "sigma_target": 0.10, "window": 60})
    cold = [t1.update(v) for v in navs[:5]]
    assert all(w == 1.0 for w in cold), "冷启动应满仓"
    print("冷启动满仓: PASS")

    # 低波动序列：日收益 ±0.3% → 年化 ~4.7%，σ_target=0.10 应顶满仓
    navs_low = [1.0]
    for _ in range(120):
        navs_low.append(navs_low[-1] * (1 + np.random.randn() * 0.0003))
    t2 = VolTargetTimer({"enabled": True, "sigma_target": 0.10, "window": 60})
    ws_low = [t2.update(v) for v in navs_low]
    hot = [w for w in ws_low if w >= 1.0 - 1e-9]
    print(f"低波动满仓占比: {len(hot)}/{len(ws_low)}")
    assert len(hot) > 50, "低波动应顶满仓"
    print("低波动满仓: PASS")

    print("ALL PASS")


if __name__ == "__main__":
    _smoke()
