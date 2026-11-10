"""端到端示例：不需要任何真实数据，用合成数据跑通框架。

用途
----
外部用户 clone 公开仓库后，不用拉数据、不用配 token，直接:

    pip install -e .
    python -m jq_framework.examples.run_demo

就能看到：因子计算 → 截面处理 → 组合打分 → 选股 → 策略信号 的完整链路。

这同时是"框架没坏"的冒烟测试：任何一步出错都会在这里暴露。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from jq_framework.factors.examples.demo_factors import Momentum20, Reversal20
from jq_framework.strategies.examples.demo_strategies import DualMaExample, DonchianExample


def _synthetic_prices(n_sym: int = 20, n_days: int = 250, seed: int = 42):
    """造一批带趋势的合成价格：{symbol: DataFrame}。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    out = {}
    for i in range(n_sym):
        drift = rng.normal(0.0003, 0.0008)          # 各标的趋势强弱不同
        close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.015, n_days)))
        out[f"SYM{i:02d}"] = pd.DataFrame({
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.002, n_days)),
            "high": close * (1 + np.abs(rng.normal(0, 0.006, n_days))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, n_days))),
            "close": close,
            "volume": rng.integers(100_000, 2_000_000, n_days),
        })
    return out


def main() -> None:
    line = "=" * 68
    print(line)
    print("  jq_framework 端到端示例（合成数据，无需联网）")
    print(line)

    # ---------- 1. 因子 ----------
    print("\n[1] 因子计算与截面处理")
    price_dfs = _synthetic_prices()
    print(f"    合成标的 {len(price_dfs)} 个，"
          f"每个 {len(next(iter(price_dfs.values())))} 根日线")

    factors = [
        (Momentum20(params={"window": 20}), 1.0),
        (Reversal20(params={"window": 20}), 0.5),
    ]
    for fac, weight in factors:
        cs = fac.calc_cross_section(price_dfs)
        top = cs.sort_values(ascending=False).head(3)
        print(f"    {fac.name:<12} 权重={weight}  "
              f"截面 {len(cs)} 个有效值，前三: "
              f"{', '.join(f'{k}={v:.3f}' for k, v in top.items())}")

    # ---------- 2. 组合打分 ----------
    print("\n[2] 加权合成为组合打分")
    score = sum(fac.calc_cross_section(price_dfs) * w for fac, w in factors)
    ranked = score.sort_values(ascending=False)
    print(f"    合成打分 {len(ranked)} 个，取前 5 名:")
    for sym, val in ranked.head(5).items():
        print(f"      {sym}  {val:+.4f}")

    # ---------- 3. 策略信号 ----------
    print("\n[3] 单标的策略信号")
    for cls in (DualMaExample, DonchianExample):
        strat = cls()
        sym, df = next(iter(price_dfs.items()))
        sig = strat.on_bar(df)
        print(f"    {strat.name():<18} {sym}  →  {sig.direction:<5} "
              f"target_pos={sig.target_pos:.2f}  ({sig.note})")

    # ---------- 4. 契约自检 ----------
    print("\n[4] 契约自检")
    sig = DualMaExample().on_bar(next(iter(price_dfs.values())))
    assert sig.direction in ("long", "short", "flat", "hold"), "direction 非法"
    assert 0.0 <= sig.target_pos <= 1.0, "target_pos 越界"
    print("    ✓ direction 取值合法")
    print("    ✓ target_pos 在 [0, 1]")

    print("\n" + line)
    print("  框架正常。下一步：")
    print("    · 写自己的因子 → jq_framework/factors/examples/demo_factors.py")
    print("    · 写自己的策略 → jq_framework/strategies/examples/demo_strategies.py")
    print("    · 接真实数据     → jq_framework/data/fetchers/")
    print(line)


if __name__ == "__main__":
    main()
