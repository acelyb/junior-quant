"""dynamic_pool 冒烟：point-in-time 正确性 / 退化行为 / 确定性（离线，纯函数级）。

不读真实缓存，全部用构造数据。覆盖：
  1. 逐日选池按市值升序取 top_n；
  2. point-in-time：某日选池不得使用该日之后的市值（造一个"未来暴跌"用例）；
  3. 流动性门槛按当日往前 amt_window 个交易日算，且不看未来；
  4. 上市时长门槛用可用交易日数作代理；
  5. 某日全市场无市值 → 返回 None（不施加过滤，预登记第八节(1) 的退化）；
  6. 同市值并列时按 symbol 定序（§10 确定性）；
  7. enabled=false / top_n<=0 → parse_pool_cfg 返回 None（现行路径零改动）。

用法：python scripts/dynamic_pool_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from jq_framework.factors.dynamic_pool import build_pool_selector, parse_pool_cfg

DATES = pd.bdate_range("2018-01-02", periods=200)


def _mc(sym_values: dict[str, list[float]]) -> dict[str, pd.DataFrame]:
    return {s: pd.DataFrame({"date": DATES[:len(v)], "market_cap": v})
            for s, v in sym_values.items()}


def _cfg(**kw) -> dict:
    base = {"enabled": True, "top_n": 2, "min_amount": 0.0,
            "min_listed_days": 0, "amt_window": 5, "min_market_cap": None}
    base.update(kw)
    return base


def _prices(syms, n=200, close=10.0, volume=5e6):
    return {s: pd.DataFrame({"date": DATES[:n], "close": close, "volume": volume})
            for s in syms}


def main() -> int:
    print("== 动态训练池冒烟 ==")

    # ---- 1. 按市值升序取 top_n ----
    mc = _mc({"A": [100.0] * 200, "B": [50.0] * 200, "C": [200.0] * 200})
    px = _prices(mc.keys())
    sel = build_pool_selector(mc, px, _cfg(top_n=2))
    got = sel(DATES[100], set(mc))
    assert got == {"B", "A"}, f"应取市值最小的 B,A（实际 {got}）"
    print("  [1] 市值升序取 top_n OK")

    # ---- 2. point-in-time：不得使用未来市值 ----
    # A 在 DATES[150] 之后市值暴跌到 1；在 DATES[100] 选池时 A 仍应是 100（不应被选中）
    vals = [100.0] * 200
    for i in range(150, 200):
        vals[i] = 1.0
    mc2 = _mc({"A": vals, "B": [50.0] * 200, "C": [200.0] * 200})
    sel2 = build_pool_selector(mc2, _prices(mc2.keys()), _cfg(top_n=2))
    at100 = sel2(DATES[100], set(mc2))
    at190 = sel2(DATES[190], set(mc2))
    assert at100 == {"B", "A"}, f"DATES[100] 不应看到 A 未来的暴跌（实际 {at100}）"
    assert at190 == {"A", "B"}, f"DATES[190] A 已是最小（实际 {at190}）"
    print("  [2] point-in-time（不看未来市值）OK")

    # ---- 3. 流动性门槛按当日往前窗口算 ----
    # B 成交额 1000（远低于门槛 1e8）→ 被剔除；A/C 保留
    px3 = _prices(mc.keys())
    px3["B"] = pd.DataFrame({"date": DATES, "close": 1.0, "volume": 1000.0})
    sel3 = build_pool_selector(mc, px3, _cfg(top_n=5, min_amount=1e7))
    got3 = sel3(DATES[150], set(mc))
    assert got3 == {"A", "C"}, f"低流动性 B 应被剔除（实际 {got3}）"
    print("  [3] 流动性门槛 OK")

    # ---- 4. 上市时长门槛（可用交易日数代理）----
    # A 从 DATES[30] 起才有行情；B/C 有完整 200 天。min_listed_days=120
    #   DATES[50]  → A 仅 21 个交易日 < 120 → 剔除
    #   DATES[150] → A 有 121 个交易日 ≥ 120 → 入池
    off = 30
    mc4 = _mc({"A": [100.0] * 200, "B": [50.0] * 200, "C": [200.0] * 200})
    px4 = {"A": pd.DataFrame({"date": DATES[off:], "close": 10.0, "volume": 5e6})}
    for s in ("B", "C"):
        px4[s] = pd.DataFrame({"date": DATES, "close": 10.0, "volume": 5e6})
    sel4 = build_pool_selector(mc4, px4, _cfg(top_n=5, min_listed_days=120))
    got4 = sel4(DATES[50], set(mc4))
    assert "A" not in got4, f"A 上市不足应被剔除（实际 {got4}）"
    got4b = sel4(DATES[150], set(mc4))
    assert got4b == {"A", "B", "C"}, f"满足门槛后应全入池（实际 {got4b}）"
    print("  [4] 上市时长门槛 OK")

    # ---- 5. 无市值数据的日子 → 退回 fallback（= 基线训练域）----
    mc5 = _mc({"A": [100.0] * 100, "B": [50.0] * 100})   # 只有前 100 天
    sel5 = build_pool_selector(mc5, _prices(mc5.keys()), _cfg(top_n=1))
    assert sel5(DATES[50], set(mc5)) == {"B"}, "有市值时应正常过滤"
    assert sel5(DATES[150], set(mc5)) is None, \
        f"不传 fallback 时应返回 None（实际 {sel5(DATES[150], set(mc5))}）"
    # 传入 fallback 时必须返回 fallback ∩ candidates，绝不能返回"当日全部候选"
    fb = {"A"}   # 假设选股池只有 A（B 不在 universe）
    sel5b = build_pool_selector(mc5, _prices(mc5.keys()), _cfg(top_n=1),
                               fallback_symbols=fb | {"ZZZ"})
    got5 = sel5b(DATES[150], set(mc5))
    assert got5 == {"A"}, \
        f"缺市值日应退回基线训练域∩候选（应得 A，实际 {got5}）；退回全部候选=训练域膨胀"
    print("  [5] 市值缺失日退回基线训练域 OK")

    # ---- 6. 同市值并列按 symbol 定序 ----
    mc6 = _mc({"C": [50.0] * 200, "A": [50.0] * 200, "B": [50.0] * 200})
    sel6 = build_pool_selector(mc6, _prices(mc6.keys()), _cfg(top_n=2))
    assert sel6(DATES[100], set(mc6)) == {"A", "B"}, \
        f"同市值应按 symbol 升序取 A,B（实际 {sel6(DATES[100], set(mc6))}）"
    print("  [6] 同市值并列的确定性排序 OK")

    # ---- 7. 关闭 / 非法配置不生效 ----
    for bad in ({"enabled": False}, {"enabled": True, "top_n": 0}, {}, None):
        assert parse_pool_cfg(bad) is None, f"{bad} 应解析为 None"
        s = build_pool_selector(mc, px, bad)
        assert s(DATES[100], set(mc)) is None, f"{bad} 构造的选择器应恒返回 None"
    print("  [7] 关闭/非法配置零生效 OK")

    # ---- 8. 最小市值门槛 ----
    sel8 = build_pool_selector(mc, px, _cfg(top_n=5, min_market_cap=60.0))
    assert sel8(DATES[100], set(mc)) == {"A", "C"}, \
        f"市值<60 的 B 应被剔除（实际 {sel8(DATES[100], set(mc))}）"
    print("  [8] 最小市值门槛 OK")

    print("\n== 动态训练池冒烟 全部通过 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
