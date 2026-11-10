"""
期货头寸换算层：权重 → 手数/名义/保证金/资金，回测与每日信号共用。

主流期货流程「信号 → 实际头寸」的关键落点：策略/风控产出的是**仓位权重**
（{var: 带符号 w}，long 正/short 负，|w|≤1 或杠杆），而实盘要的是**开几手**。
本模块做两个方向的事：
  - `pos_size`：把组合权重 × 账户净值换算成每品种 {方向, 手数, 名义, 保证金, 资金}
  - `futures_trade_cost`：期货单边交易成本（佣金+滑点，无印花税）
  - `apply_margin_budget`：保证金预算约束（组合保证金占用 ≤ cap_util，等比缩仓）

与 A 股（risk/manager.py 等）完全隔离：本模块只 import futures 相关，服务期货线。

核心口径约定（与 futures_walk_forward/futures_risk 一致）：
  - margin_ratio 是**小数**（规则表返回 % 数时调用方须 ÷100，如 11 → 0.11）
  - multiplier 为合约乘数（每手对应名义 = price × multiplier）
  - 手数**必向下取整**（floor），防超资金预算；< 1 手归零当平仓
"""
from __future__ import annotations

import math


def apply_margin_budget(weights: dict[str, float], margin_ratios: dict[str, float],
                        cap_util: float = 0.9) -> dict[str, float]:
    """保证金预算约束：组合保证金占用 Σ|w_i|·margin_i ≤ cap_util。

    超预算则整体等比缩放（保住相对仓位，与 futures_risk._leverage_clip 同思路），
    使 margin_used ≤ cap_util 恒成立。cap_util=None/<=0 → 不设限（no-op）。
    回测与每日信号共用此函数，保证两处口径一致。
    """
    if not weights or not cap_util or cap_util <= 0:
        return weights
    used = sum(abs(v) * margin_ratios.get(k, 0.0) for k, v in weights.items())
    if used <= cap_util:
        return weights
    scale = cap_util / used
    return {k: v * scale for k, v in weights.items()}


def pos_size(weights: dict[str, float], nav: float, prices: dict[str, float],
             multiplier: dict[str, float], margin: dict[str, float],
             cap_util: float = 0.9,
             default_multiplier: float | None = None) -> dict[str, dict]:
    """组合权重 × 账户净值 → 每品种实际头寸 {var: {方向, 手数, 名义, 保证金}}。

    手数 = floor(|w|·nav / (price·multiplier))，必向下取整防超预算；<1 手归零（平仓）。
    名义 = 手数 × price × multiplier；保证金 = 名义 × margin_ratio。
    multiplier 缺失时用 default_multiplier 兜底；都不给则跳过该品种。
    margin 为小数（% 数须先 ÷100）。cap_util 引入前先按保证金预算约束权重
    （先缩权重再算手数，绝不可反，否则手数会突破保证金预算）。
    """
    # 先按保证金预算约束权重（顺序关键：先缩 w 再算手数）
    w_scaled = apply_margin_budget(weights, margin, cap_util=cap_util)
    out: dict[str, dict] = {}
    for var, w in w_scaled.items():
        if w == 0:
            continue
        price = prices.get(var)
        mult = multiplier.get(var) or default_multiplier
        if not price or price <= 0 or not mult or mult <= 0:
            continue
        mr = margin.get(var, 0.0)
        notional_per_lot = price * mult
        lots = math.floor(abs(w) * nav / notional_per_lot)
        if lots < 1:
            out[var] = {"direction": 0, "lots": 0, "notional": 0.0, "margin": 0.0}
            continue
        notional = lots * notional_per_lot
        out[var] = {
            "direction": 1 if w > 0 else -1,
            "lots": lots,
            "notional": notional,
            "margin": notional * mr,
        }
    return out


def futures_trade_cost(prev_w: dict[str, float], target_w: dict[str, float],
                       commission: float, slippage: float) -> float:
    """期货单边交易成本（相对组合权重，返回费率）。

    对齐主流「买入+卖出+换手」三口径（期货无印花税）：
      - buy  = Σ 新进场（prev 为 0 → target 非 0）
      - sell = Σ 清仓（prev 非 0 → target 为 0）
      - rebal = Σ|Δw| 用于调仓的权重变动部分（|+0.5→−0.5| 计 1.0，天然处理多空换向）
    cost = (buy + sell + rebal) × rate / 2，rate = commission + slippage。

    调用方：回测把 prev_w 设为『上期已执行权重』、target_w 设为『本期风控后目标权重』，
    只在调仓日计成本。返回费率，作用于 nav: nav *= (1+port_ret) * (1 - cost)。
    """
    rate = float(commission) + float(slippage)
    if rate <= 0:
        return 0.0
    keys = set(prev_w) | set(target_w)
    buy = sum(max(float(target_w.get(k, 0.0)) - float(prev_w.get(k, 0.0)), 0.0)
              for k in keys if prev_w.get(k, 0.0) == 0 and target_w.get(k, 0.0) != 0)
    sell = sum(abs(float(prev_w.get(k, 0.0)))
               for k in keys if prev_w.get(k, 0.0) != 0 and target_w.get(k, 0.0) == 0)
    rebal = sum(abs(float(target_w.get(k, 0.0)) - float(prev_w.get(k, 0.0))) for k in keys)
    return (buy + sell + rebal) * rate / 2.0


def margin_used(weights: dict[str, float], margin_ratios: dict[str, float]) -> float:
    """组合保证金占用 = Σ|w_i|·margin_i（诊断/记录用）。"""
    return sum(abs(v) * margin_ratios.get(k, 0.0) for k, v in weights.items())


def _smoke():
    """单元烟测：手数/名义/保证金手算核实 + 预算约束 + 成本核算。"""
    print("=== portfolio/futures_sizing 烟测 ===")

    # 1) pos_size 手算核实：w=0.5, nav=1e6, price=4000, mult=10, margin=0.11
    #    名义/手 = 4000*10=40000；手数=floor(0.5*1e6/40000)=floor(12.5)=12
    #    名义=12*40000=480000；保证金=480000*0.11=52800
    w = {"RB": 0.5}
    prices = {"RB": 4000.0}
    mult = {"RB": 10.0}
    margin = {"RB": 0.11}
    out = pos_size(w, 1_000_000, prices, mult, margin)
    rb = out["RB"]
    assert rb["lots"] == 12, f"手数应=12, got {rb['lots']}"
    assert abs(rb["notional"] - 480000) < 1e-6, f"名义应=480000, got {rb['notional']}"
    assert abs(rb["margin"] - 52800) < 1e-6, f"保证金应=52800, got {rb['margin']}"
    assert rb["direction"] == 1
    print("pos_size 手算核实: PASS")

    # 2) floor 向下取整 + <1手归零：w=0.05 → 0.05*1e6/40000=1.25→1手；w=0.001→0归零
    out2 = pos_size({"RB": 0.05}, 1_000_000, {"RB": 4000.0}, {"RB": 10.0}, {"RB": 0.11})
    assert out2["RB"]["lots"] == 1
    out3 = pos_size({"RB": 0.001}, 1_000_000, {"RB": 4000.0}, {"RB": 10.0}, {"RB": 0.11})
    assert out3["RB"]["lots"] == 0 and out3["RB"]["direction"] == 0
    print("floor取整 + <1手归零: PASS")

    # 3) multiplier=None 兜底：default_multiplier=10 生效
    out4 = pos_size({"CU": 0.5}, 1_000_000, {"CU": 70000.0}, {}, {"CU": 0.08},
                    default_multiplier=5)
    # 名义/手=70000*5=350000；0.5*1e6/350000=1.428→1手
    assert out4["CU"]["lots"] == 1
    # 都不给 → 跳过
    out5 = pos_size({"ZN": 0.5}, 1e6, {"ZN": 100.0}, {}, {"ZN": 0.1})
    assert "ZN" not in out5
    print("multiplier 兜底: PASS")

    # 4) apply_margin_budget：高保证金占用缩仓，margin_used≤cap_util
    w_hi = {"RB": 1.0, "AU": 1.0}
    m = {"RB": 0.3, "AU": 0.3}  # used=0.6 > 0.9? no → 0.6. 用更激进的
    w_hi2 = {"RB": 2.0, "AU": 2.0}  # used=1.2 > 0.9 → scale=0.9/1.2=0.75
    ws = apply_margin_budget(w_hi2, {"RB": 0.3, "AU": 0.3}, cap_util=0.9)
    used = sum(abs(v) * {"RB": 0.3, "AU": 0.3}[k] for k, v in ws.items())
    assert abs(used - 0.9) < 1e-9 and abs(ws["RB"] / ws["AU"] - 1.0) < 1e-9, \
        "超预算应等比缩到 cap_util，比例保持"
    assert apply_margin_budget({"RB": 0.5}, {"RB": 0.11}, None) == {"RB": 0.5}, "无cap不缩"
    print("保证金预算约束: PASS")

    # 5) futures_trade_cost：多空换向 |+0.5→−0.5| 计 1.0 → rebal=1.0
    prev = {"RB": 0.5}
    tgt = {"RB": -0.5}
    cost = futures_trade_cost(prev, tgt, commission=0.0001, slippage=0.0002)
    # rate=0.0003; buy=0(prev非0), sell=0(target非0), rebal=1.0 → 1.0*0.0003/2=0.00015
    assert abs(cost - 0.00015) < 1e-12, f"换向成本应=0.00015, got {cost}"
    # 建仓：prev空→target 0.5：buy=0.5,sell=0,rebal=0.5 → (0.5+0.5)*0.0003/2=0.00015
    cost2 = futures_trade_cost({}, {"RB": 0.5}, 0.0001, 0.0002)
    assert abs(cost2 - 0.00015) < 1e-12, f"建仓成本应=0.00015, got {cost2}"
    # 零成本配置 → 0
    assert futures_trade_cost(prev, tgt, 0, 0) == 0.0
    print("futures_trade_cost: PASS")

    print("ALL PASS")


if __name__ == "__main__":
    _smoke()
