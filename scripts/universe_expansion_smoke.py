"""universe_expansion 冒烟：并集语义 / point-in-time / 剂量单调 / 退化（离线，纯函数级）。

不读真实缓存，全部用构造数据。逐条对应 M7 预登记（计划§八）的约束：
  1. 并集语义：成分股不被剔除——即使它市值很大、流动性很差、上市时间短；
  2. 门槛只作用于非成员：非成员不达标则不入池，成员不达标仍入池；
  3. point-in-time：不得使用 t 之后的市值/行情（造"未来暴跌"与"未来放量"用例）；
  4. 市值陈旧度：超过 max_stale_days 的市值视为不可观测；
  5. 剂量单调：K 增大时 U 只增不减；
  6. 确定性：同市值并列按 symbol 定序；
  7. 基准逐日换成员：等权收益用当日成员算，不得混入未入池标的；
  8. 关闭/非法配置返回空（现行路径零改动）。

用法：python scripts/universe_expansion_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from jq_framework.factors.universe_expansion import (build_universe_panel, panel_to_returns,
                                        parse_expansion_cfg)

DATES = pd.bdate_range("2018-01-02", periods=300)
START = DATES[0]


def _mc(sym_vals: dict[str, list[float]], start_idx: int = 0) -> dict[str, pd.DataFrame]:
    return {s: pd.DataFrame({"date": DATES[start_idx:start_idx + len(v)],
                             "market_cap": v})
            for s, v in sym_vals.items()}


def _px(syms, n=300, close=10.0, volume=5e6, start_idx=0,
        over: dict | None = None) -> dict[str, pd.DataFrame]:
    # 注意：不能把 over 写成 **over——`_px(..., over={...})` 会把整个字典塞进名为 "over"
    # 的键里，自定义收盘价静默失效（本文件首版即栽在此，断言因此读到的全是常数序列）。
    over = over or {}
    out = {}
    for s in syms:
        n_ = over.get(s, {}).get("n", n)
        c = over.get(s, {}).get("close", close)
        v = over.get(s, {}).get("volume", volume)
        c = np.asarray(c if isinstance(c, (list, np.ndarray)) else [c] * n_, dtype=float)
        v = np.asarray(v if isinstance(v, (list, np.ndarray)) else [v] * n_, dtype=float)
        out[s] = pd.DataFrame({"date": DATES[start_idx:start_idx + len(c)],
                               "close": c, "volume": v})
    return out


def _cfg(**kw) -> dict:
    base = {"enabled": True, "top_n": 2, "min_amount": 0.0,
            "min_listed_days": 0, "amt_window": 5, "max_stale_days": 15}
    base.update(kw)
    return base


def main() -> int:
    print("== 扩充可投资域冒烟 ==")

    # ---- 1+2. 并集语义：成员永不剔除，门槛只作用于非成员 ----
    # M1/M2 为成员（M2 流动性极差、市值极大）；N1/N2 为非成员（正常）
    mc = _mc({"M1": [100.0] * 300, "M2": [900.0] * 300,
              "N1": [10.0] * 300, "N2": [20.0] * 300, "N3": [30.0] * 300})
    px = _px(["M1", "M2", "N1", "N2", "N3"],
             over={"M2": {"volume": 1.0}})      # M2 流动性远低于门槛
    members = {"M1", "M2"}
    U = build_universe_panel(mc, px, members, DATES, _cfg(top_n=1, min_amount=1e7))
    t = DATES[200]
    # 门槛 1e7：N1/N2/N3 成交额 5e6*10=5e7 均达标；M2 成交额 10 不达标
    got = U[t]
    assert "M1" in got, f"成员 M1 必须在池（实际 {got}）"
    assert "M2" in got, f"成员 M2 即使流动性差也必须在池（并集语义，实际 {got}）"
    assert "N1" in got, f"非成员 N1 市值最小应入池（实际 {got}）"
    assert "N2" not in got and "N3" not in got, f"K=1 时只取最小非成员 N1（实际 {got}）"
    print("  [1][2] 并集语义 + 门槛只作用于非成员 OK")

    # ---- 3. point-in-time：不得使用未来市值 ----
    # N1 在 DATES[250] 之后市值暴跌到 1；在 DATES[100] 选池时不应被选中
    vals = [100.0] * 300
    for i in range(250, 300):
        vals[i] = 1.0
    mc2 = _mc({"N1": vals, "N2": [50.0] * 300, "N3": [80.0] * 300})
    U2 = build_universe_panel(mc2, _px(mc2.keys()), set(), DATES, _cfg(top_n=1))
    at100 = U2[DATES[100]]
    at290 = U2[DATES[290]]
    assert at100 == {"N2"}, f"DATES[100] 不应看到 N1 未来的暴跌（实际 {at100}）"
    assert at290 == {"N1"}, f"DATES[290] N1 已是最小（实际 {at290}）"
    print("  [3] point-in-time（不看未来市值）OK")

    # ---- 4. 市值陈旧度 ----
    # N1 的市值序列在 DATES[100] 后中断；在 DATES[200] 应视为不可观测
    mc3 = _mc({"N1": [1.0] * 100})                # 只有前 100 天
    mc3.update(_mc({"N2": [50.0] * 300}, start_idx=0))
    U3 = build_universe_panel(mc3, _px(["N1", "N2"]), set(), DATES, _cfg(top_n=1))
    assert U3[DATES[50]] == {"N1"}, f"有新鲜市值时应选中 N1（实际 {U3[DATES[50]]}）"
    assert U3[DATES[200]] == {"N2"}, \
        f"N1 市值已陈旧，应跳过（实际 {U3[DATES[200]]}）——否则等于用过期市值定位"
    print("  [4] 市值陈旧度截断 OK")

    # ---- 5. 剂量单调：K 增大，池只增不减 ----
    mc5 = _mc({f"N{i}": [float(10 * i)] * 300 for i in range(1, 11)})
    px5 = _px(mc5.keys())
    prev = None
    for K in (1, 3, 5, 10):
        Uk = build_universe_panel(mc5, px5, set(), DATES, _cfg(top_n=K))
        s = Uk[DATES[200]]
        if prev is not None:
            assert prev <= s, f"K={K} 的池不是前一档的超集（{prev} ⊄ {s}）"
        prev = s
    assert len(prev) == 10, f"K=10 应取满 10 只（实际 {len(prev)}）"
    print("  [5] 剂量单调（K 增大池只增不减）OK")

    # ---- 6. 同市值并列按 symbol 定序 ----
    mc6 = _mc({"C": [50.0] * 300, "A": [50.0] * 300, "B": [50.0] * 300})
    U6 = build_universe_panel(mc6, _px(mc6.keys()), set(), DATES, _cfg(top_n=2))
    assert U6[DATES[100]] == {"A", "B"}, \
        f"同市值应按 symbol 升序取 A,B（实际 {U6[DATES[100]]}）"
    print("  [6] 同市值并列的确定性排序 OK")

    # ---- 7. 基准逐日换成员 ----
    # 第 0..149 日只有 A 在池；第 150 日起 B 也入池。基准收益必须按当日成员算。
    close_a = [10.0] * 300
    for i in range(1, 300):
        close_a[i] = close_a[i - 1] * 1.01        # A 每日 +1%
    close_b = [10.0] * 300
    for i in range(1, 300):
        close_b[i] = close_b[i - 1] * 0.99        # B 每日 -1%
    px7 = _px(["A", "B"], over={"A": {"close": close_a}, "B": {"close": close_b}})
    panel7 = {DATES[i]: ({"A"} if i < 150 else {"A", "B"}) for i in range(300)}
    r = panel_to_returns(panel7, px7, START)
    early = r.iloc[10]     # 只有 A → +1%
    late = r.iloc[200]     # A+B 等权 → 0%
    assert abs(early - 0.01) < 1e-9, f"早期基准应 = A 的 +1%（实际 {early}）"
    assert abs(late) < 1e-9, f"后期基准应 = (1% + (-1%))/2 = 0（实际 {late}）"
    print("  [7] 基准逐日换成员（等权用当日成员）OK")

    # ---- 8. 关闭/非法配置 ----
    for bad in ({"enabled": False}, {"enabled": True, "top_n": 0}, {}, None):
        assert parse_expansion_cfg(bad) is None, f"{bad} 应解析为 None"
        assert build_universe_panel(mc, px, members, DATES, bad) == {}, \
            f"{bad} 应返回空池"
    print("  [8] 关闭/非法配置零生效 OK")

    # ---- 9. 停牌中的成分股不得被剔除（U ⊇ csi1000 的硬保证）----
    # M1 是成分股，但在 DATES[200] 当天停牌（日线里没有这一天）。基线判据是
    # "截至当日有任意历史数据"（walk_forward 的 `(df["date"] <= date).any()`），
    # 不要求当日有行情。若 U 用"当日有行情"判定成分股，M1 会被剔出 → U 不再是
    # csi1000 的超集 → M7 与基线的差混入"剔除停牌股"这一未登记因素。
    d_all = list(DATES)
    d_gap = [DATES[i] for i in range(len(DATES)) if i != 200]      # 缺 DATES[200]
    px9 = {"M1": pd.DataFrame({"date": d_gap, "close": 10.0, "volume": 5e6}),
           "N1": pd.DataFrame({"date": d_all, "close": 10.0, "volume": 5e6}),
           "N2": pd.DataFrame({"date": d_gap, "close": 1.0, "volume": 5e6})}
    # N2 市值最小但当日停牌；N1 市值稍大且当日可交易 → 应选 N1
    mc9 = _mc({"M1": [100.0] * 300, "N1": [20.0] * 300, "N2": [10.0] * 300})
    U9 = build_universe_panel(mc9, px9, {"M1"}, d_all, _cfg(top_n=1))
    assert "M1" in U9[DATES[200]], \
        f"停牌当日的成员必须仍在池中（实际 {U9[DATES[200]]}）——否则 U 不是 csi1000 的超集"
    assert "N2" not in U9[DATES[200]], \
        f"当日停牌的非成员不应入池（买入需要当日可交易，实际 {U9[DATES[200]]}）"
    assert "N1" in U9[DATES[200]], \
        f"当日可交易的非成员应入池（实际 {U9[DATES[200]]}）"
    print("  [9] 停牌成员不被剔除（U ⊇ csi1000）OK")

    print("\n== 扩充可投资域冒烟 全部通过 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
