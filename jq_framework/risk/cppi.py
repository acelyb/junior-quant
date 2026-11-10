"""
仓位层 CPPI（恒定比例组合保险）实时缩放器。

定位（重要）：这是**仓位层**非选股层，与 factors/ 截面选股正交：
- 选股层（composite_score）定"买什么"——横截面排序，21 天周期重排 + turnover_buffer
- 仓位层（本模块）定"配多少"——时间序列，逐日对组合总仓位施加系数 w∈[w_min, w_max]
两者串联：个股目标权重 = 选股等权(或波动率加权) × CPPI 系数 w。CPPI 只缩总仓位，
不碰个股相对权重，不触发选股——这与净值层仿真（overlay_cppi.py）假设逐字一致。

为什么 CPPI 不能像 RSRS（risk/timing.py）那样预计算查表：
  RSRS 乘数依赖外生指数（high/low），回测前一次性算好 {date: mult} 字典，回测中 O(1) 查表，
  且只在调仓日缩放 new_weights（持仓权重调仓日重建、非调仓日原样保留）。
  CPPI 乘数依赖**内生 nav**（cushion = V - floor，V 是组合自身净值，逐日变），无法预计算——
  必须回测中逐日 update(nav)。用户确认日频 CPPI：每个交易日都按 t-1 nav 重算 w 驱动 t 日收益，
  非调仓日也必须缩仓，否则 21 天里 20 天仓位不受控（RSRS 只在调仓日缩权重的路径走不通）。
  故 CPPI 在 walk_forward 的 **port_ret 层**施加（对当日组合收益乘 w），而非权重层。

机制（与 overlay_cppi.py:simulate_cppi 同口径）：
  cushion(t-1) = max(V(t-1) - floor(t-1), 0)
  w(t) = clip(cushion/V(t-1) × M, w_min, w_max)
  V(t) = V(t-1) × (1 + w(t) × r(t))
  floor 锁高水位（lock_high_water=True）：floor = hwm × floor_pct，hwm = 历史最高 V。
  涨→cushion 厚→w 升（虚拟加仓）；跌→逼近 floor→w 降（虚拟减仓）；触底→w=w_min（不全空）。
  左侧数值驱动，不依赖趋势判断，熊市反弹不会重满仓（补 RSRS 短板）。

三个必须（净值层验证 [[cppi-overlay-validated]]）：
  1. w_min>0 底仓：纯 CPPI（w_min=0）有 cash-lock——V 跌破 floor 仓位归零永久清仓踏空牛市。
  2. lock_high_water=True：固定 floor 在上涨 nav 上失效（净值涨高 cushion 巨大恒满仓）。
  3. w_max≤1 纯多头不杠杆。

防未来函数：update(nav_t) 用 t-1 收盘 nav 算 w_t，喂 t 日 port_ret，无未来函数泄漏。

四硬约束（与 risk/timing.py 同）：
- Windows：纯标准库 + numpy，无文件 I/O
- 正确性：单实例有可变状态（hwm），但每 run_walk_forward 只创建一个实例跨窗口延续，
  线程内顺序调用，不存在并发写；get_multiplier 接口保留兼容但走 update 实时路径
- 资源：update O(1)，回测主循环每日一次，零额外开销
"""
from __future__ import annotations

import numpy as np


class CPPITimer:
    """CPPI 仓位缩放器：回测中逐日 update(nav) 实时算总仓位系数 w。

    用法：
        timer = CPPITimer(cfg)       # cfg=None 或 cppi.enabled=false → no-op（恒返 1.0）
        w = timer.update(nav_t)      # 回测主循环每日调用，nav_t=t-1 收盘净值
        port_ret *= w                # 缩总仓位，现金 (1-w) 收益为 0

    门控：enabled=false 时 update 恒返 1.0（零开销，不破回归门，与 MarketTimer/intraday_filter 同范式）。
    状态（hwm/v0）跨窗口延续：run_walk_forward 内创建一次，nav 跨窗口累积不重置（walk_forward.py:485），
    CPPI 的 hwm 也必须跨窗口延续，否则每窗口 hwm 重置破坏 HWM 锁语义。
    """

    def __init__(self, cfg: dict | None = None):
        """cfg: multifactor.cppi 字典（config.yaml）。None 或 enabled=false → no-op。"""
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.floor_pct = float(cfg.get("floor_pct", 0.75))
        self.m_multiplier = float(cfg.get("m_multiplier", 4.0))
        self.w_max = float(cfg.get("w_max", 1.0))
        self.w_min = float(cfg.get("w_min", 0.20))
        self.lock_high_water = bool(cfg.get("lock_high_water", True))
        # 运行态
        self.v0: float | None = None      # 初始净值基准（lock_high_water=False 时 floor=v0×floor_pct）
        self.hwm: float | None = None     # 历史最高净值（lock_high_water=True 时 floor=hwm×floor_pct）
        # 诊断
        self._n_calls = 0
        self._w_sum = 0.0
        self._w_min_seen = self.w_max
        self._w_max_seen = self.w_min

    def update(self, nav_t: float) -> float:
        """用 t-1 收盘净值 nav_t 算 t 日总仓位系数 w（无未来函数）。

        enabled=false → 恒返 1.0（no-op）。nav_t<=0 → 返回 w_min（净值归零保守最低仓）。
        首次调用初始化 v0 与 hwm 为 nav_t。
        """
        if not self.enabled:
            return 1.0
        if nav_t is None or not np.isfinite(nav_t) or nav_t <= 0:
            return self.w_min
        if self.v0 is None:
            self.v0 = nav_t
        if self.hwm is None:
            self.hwm = nav_t
        else:
            self.hwm = max(self.hwm, nav_t)
        base = self.hwm if self.lock_high_water else self.v0
        floor = base * self.floor_pct
        cushion = max(nav_t - floor, 0.0)
        w = cushion / nav_t * self.m_multiplier
        w = float(np.clip(w, self.w_min, self.w_max))
        # 诊断累计
        self._n_calls += 1
        self._w_sum += w
        self._w_min_seen = min(self._w_min_seen, w)
        self._w_max_seen = max(self._w_max_seen, w)
        return w

    def get_multiplier(self, nav_t: float) -> float:
        """与 MarketTimer 同名接口（集成点统一调用），内部走 update 实时路径。

        MarketTimer.get_multiplier(date) 是查表（依赖外生指数预计算）；
        CPPI 无法查表（依赖内生 nav），故此接口直接转发 update(nav_t)。
        """
        return self.update(nav_t)

    def coverage(self) -> tuple[int, float, float, float]:
        """诊断：返回 (调用次数, 平均仓位, 最低仓位, 最高仓位)。未启用返 (0,1,1,1)。"""
        if not self.enabled or self._n_calls == 0:
            return (0, 1.0, 1.0, 1.0)
        n = self._n_calls
        return (n, self._w_sum / n, self._w_min_seen, self._w_max_seen)


def _smoke():
    """单元烟测：构造已知 nav 序列验证 update 行为，并对照 overlay_cppi.py 口径校准。"""
    # 场景：起始1.0 → 深跌到0.4 → 反弹到0.8 → 新高到1.5
    navs = [1.0, 0.7, 0.4, 0.8, 1.5]
    t = CPPITimer({"enabled": True, "floor_pct": 0.75, "m_multiplier": 4.0,
                   "w_max": 1.0, "w_min": 0.20, "lock_high_water": True})
    ws = [t.update(v) for v in navs]
    print("navs :", navs)
    print("w    :", [f"{w:.3f}" for w in ws])
    # 手算校验
    # t0: v0=hwm=1.0, floor=0.75, cushion=0.25, w=0.25/1.0*4=1.0 → clip 1.0
    # t1: nav=0.7, hwm=1.0, floor=0.75, cushion=0, w=0 → clip w_min=0.20 (cash-lock 修复)
    # t2: nav=0.4, hwm=1.0, floor=0.75, cushion=0, w=0 → 0.20
    # t3: nav=0.8, hwm=1.0, floor=0.75, cushion=0.05, w=0.05/0.8*4=0.25 → 0.25
    # t4: nav=1.5, hwm=1.5, floor=1.125, cushion=0.375, w=0.375/1.5*4=1.0 → 1.0 (HWM 锁：floor 随新高升)
    expected = [1.0, 0.20, 0.20, 0.25, 1.0]
    ok = all(abs(w - e) < 1e-9 for w, e in zip(ws, expected))
    print("expected:", expected, "PASS" if ok else "FAIL")
    assert ok, f"w mismatch: {ws} vs {expected}"
    cov = t.coverage()
    print(f"coverage: n={cov[0]} avg={cov[1]:.3f} min={cov[2]:.3f} max={cov[3]:.3f}")

    # no-op 门控
    t0 = CPPITimer({"enabled": False})
    assert all(t0.update(v) == 1.0 for v in navs), "disabled must return 1.0"
    print("no-op: PASS")

    # 对照 overlay_cppi.py:simulate_cppi 口径（逐日 w 序列应一致）
    try:
        import pandas as pd
        from overlay_cppi import simulate_cppi
        nav_series = pd.Series(navs)
        v_sim, w_sim = simulate_cppi(nav_series, floor_pct=0.75, m_multiplier=4.0,
                                     w_max=1.0, w_min=0.20, lock_high_water=True)
        # simulate_cppi: w[0] 在 lock_high_water=True 时设为 w_max=1.0，w[1:] 逐日算
        # CPPITimer.update: 每次调用算一次 w（含首点用 nav 自身初始化 hwm）
        # 首点口径：simulate w[0]=1.0(w_max)；timer w[0] 用 nav[0]=1.0 算得 1.0，一致
        # 后续点：simulate w[t] 用 V[t-1]；timer.update(navs[t-1]) 用 navs[t-1]，
        # 但 timer 是顺序调用 navs[0..4]，simulate 的 V[t-1] 是仿真净值非原始 nav。
        # 两者口径不同（timer 喂原始 nav 序列 vs simulate 喂满仓 nav 算仿真 V），
        # 故只在"满仓 nav 即原始 nav"的简化下可比——此处仅校验首点 w[0]=1.0 一致，
        # 完整口径校验由真实回测 A/B 承担（见计划 §3）。
        print(f"simulate_cppi w[0]={w_sim.iloc[0]:.3f} vs timer w[0]={ws[0]:.3f} "
              f"{'match' if abs(w_sim.iloc[0] - ws[0]) < 1e-9 else 'DIFFER(口径差异,见注释)'}")
    except ImportError:
        print("overlay_cppi 不可导入，跳过交叉校验（非阻断）")
    print("ALL PASS")


if __name__ == "__main__":
    _smoke()
