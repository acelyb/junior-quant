"""方向/权重决策策略（DirectionPolicy）——框架层接口。

背景
----
训练段拿到每个因子的 IC 后，要决定三件事：
  1. 用不用它（sign = +1 正向 / -1 反向 / 0 剔除）
  2. 用多大权重
  3. 为什么（reason，便于人读）

把"IC → 权重"的换算抽成策略，是因为**这里正是各家 alpha 的分野**：
阈值定在哪、用 IC 还是 ICIR、幅度要不要裁、裁多少——同一批因子，
换一套换算就是另一套组合。

接口
----
只暴露一个方法 `weight_from_ic`，签名固定。IC 的**计算**（昂贵、通用）
留在框架里共用，不在策略里重复。

    policy.weight_from_ic(sign, ic_mean, icir, base_weight) -> float

  sign:        +1 / -1 / 0（0 表示该因子本轮被剔除，策略应返回 0.0）
  ic_mean:     训练段 IC 均值
  icir:        IC / std(IC)，可能 NaN
  base_weight: 配置里的静态权重（static 策略用）

内置实现
--------
  StaticDirection    weight = base_weight（方向已由 sign 决定）
  ICRatioDirection   weight = sign · clip(|ICIR| 或 |IC均值|, floor, cap)

私有扩展
--------
继承 DirectionPolicy 实现自己的换算即可；配置里用 `direction_policy`
指定模块路径，由 resolve_policy 动态加载（与因子/策略同一套机制）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DirectionPolicy(ABC):
    """IC → 权重 的换算策略。无状态，可被多进程 pickle。"""

    @abstractmethod
    def weight_from_ic(self, sign: int, ic_mean: float, icir: float,
                       base_weight: float) -> float:
        """返回该因子本轮权重。sign==0 时必须返回 0.0。"""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


class StaticDirection(DirectionPolicy):
    """静态权重：方向由 IC 符号定，权重用配置值。

    这是最朴素的形态——不做幅度加权，稳定的老做法。
    """

    def weight_from_ic(self, sign, ic_mean, icir, base_weight):
        return 0.0 if sign == 0 else float(base_weight)


class ICRatioDirection(DirectionPolicy):
    """ICIR 幅度加权：weight = sign · clip(|ICIR|, floor, cap)。

    - 用 |ICIR| 而非 |IC均值|：ICIR 已经含了稳定性信息（IC/std），
      比裸 IC 均值更不容易奖励"偶然大、平时乱"的因子。
    - ICIR 为 NaN 时回退 |IC均值|（数据太少算不出 std 的场景）。
    - floor / cap 防两个极端：ICIR 爆炸时单因子独大、ICIR 太小时因子被丢光。
    """

    def __init__(self, floor: float = 0.25, cap: float = 1.5):
        self.floor = float(floor)
        self.cap = float(cap)

    def weight_from_ic(self, sign, ic_mean, icir, base_weight):
        if sign == 0:
            return 0.0
        mag = abs(icir) if not np.isnan(icir) else abs(ic_mean)
        return sign * float(np.clip(mag, self.floor, self.cap))


# weighting 配置值 → 内置策略
_BUILTIN = {
    "static": lambda clamp: StaticDirection(),
    "ic": lambda clamp: ICRatioDirection(*clamp),
}


def resolve_policy(weighting: str | None = None,
                   clamp: tuple = (0.25, 1.5),
                   spec: dict | None = None) -> DirectionPolicy:
    """按配置构造策略。

    spec（可选）：{"module": "...", "class": "...", "params": {...}}
                  给了就走动态加载，供私有实现接入。
    weighting：   "static" / "ic"（内置）；未知值抛错，不静默回退。
    """
    if spec:
        import importlib
        module = importlib.import_module(spec["module"])
        cls = getattr(module, spec["class"])
        return cls(**spec.get("params", {}))
    if weighting not in _BUILTIN:
        raise ValueError(
            f"未知 weighting={weighting!r}；内置支持 {sorted(_BUILTIN)}，"
            f"或提供 direction_policy 的 module/class 配置")
    return _BUILTIN[weighting](clamp)
