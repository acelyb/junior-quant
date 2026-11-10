"""
每日信号 runner（A 路径）。

作用：对所有标的跑一遍策略，产出"今日信号清单"，作为交易指导。
关键：与回测共用同一套策略代码（strategies/base.py 的 on_bar），
      只是这里喂完整历史数据，看最新一根 bar 该干嘛。

用法：
  python -m signals.daily_runner --config config.yaml

输出：reports/daily_signals.txt（含每个标的的方向、仓位、原因）
"""
from __future__ import annotations

import argparse
import importlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from jq_framework.data.fetchers.fetcher import load_symbol
from jq_framework.data.fetchers.fetcher_futures import (
    build_contract_series,
    get_futures_rules,
    get_roll_yield_series,
    normalize_built_frame,
    resolve_contract,
)
from jq_framework.portfolio.futures_sizing import pos_size
from jq_framework.strategies.base import Signal


def _futures_variety(sym: str) -> str:
    """期货符号 → 品种代码：新浪主连 'RB0' → 'RB'（去尾部 0）。"""
    return sym[:-1] if sym.endswith("0") else sym


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def build_strategy(strat_cfg: dict):
    module = importlib.import_module(strat_cfg["module"])
    cls = getattr(module, strat_cfg["class"])
    return cls(params=strat_cfg.get("params", {})), strat_cfg["type"]


# carry 类策略需注入展期收益率序列（按策略类名识别，与 futures_walk_forward 同口径）
_CARRY_CLASS_NAMES = {"CarryStrategy", "TsmomCarryStrategy", "TripleFactorStrategy"}


def _need_carry(strat) -> bool:
    return type(strat).__name__ in _CARRY_CLASS_NAMES


def _is_strategy_allowed(data_market: str, strat_name: str, ds: dict) -> bool:
    """信号出口白名单判断（纯函数，可单测）。

    期货收敛到 TSMOM 核心（治理结论）：futures_strategies 白名单命中才出货，
    被否策略(reversal/triple_factor)仍注册但不进每日信号。A 股 stock_strategies
    留空=默认全部 stock 策略出货（零回归）。与历史行为逐字一致。
    """
    if data_market == "futures":
        allow = ds.get("futures_strategies")
        if allow is not None and strat_name not in allow:
            return False
        return True
    # stock
    allow_stock = ds.get("stock_strategies") or []
    if allow_stock and strat_name not in allow_stock:
        return False
    return True


def run_daily(cfg: dict) -> list[dict]:
    """
    遍历所有标的，对每个标的应用对应策略，收集信号。
    返回信号记录列表。
    """
    records = []

    # 第零步契约激活：口径一个来源（config futures.contract，缺省 X0 与历史一致）。
    # built 时信号侧走自建换月帧 + 与回测同一归一化函数（normalize_built_frame）。
    contract = resolve_contract(cfg)
    # 第二步口径 fail-loud：built 模式逐品种标注实际口径（真 built / X0 兜底），
    # 判据 = 归一化后帧有无 adj_close（与回测引擎同判据）。
    contract_sources: dict[str, str] = {}

    # 跑配置里所有策略
    for strat_name, strat_cfg in cfg["strategies"].items():
        strat, market = build_strategy(strat_cfg)
        # 策略 type(stock/cta) 映射到数据层市场名(stock/futures)
        data_market = "stock" if market == "stock" else "futures"
        # 信号出口白名单：期货收敛到 TSMOM 核心（治理结论），被否策略仍注册供研究层
        # 但不进每日信号。A 股 stock_strategies 留空=默认全部 stock 策略出货（零回归）。
        ds = cfg.get("daily_signal", {})
        if not _is_strategy_allowed(data_market, strat_name, ds):
            continue
        symbols = cfg["stock"]["symbols"] if data_market == "stock" \
            else cfg["futures"]["symbols"]
        adjust = cfg["stock"].get("adjust", "qfq") if data_market == "stock" else None
        need_carry = _need_carry(strat)

        for sym in symbols:
            try:
                if data_market == "futures" and contract == "built":
                    # built 口径：自建换月帧归一化（与 futures_walk_forward 同函数）。
                    # read_only 走缓存（built 逐合约缓存由 pipeline fetch stage 预热），
                    # 未命中合约才补拉——避免每日信号触发全品种逐月枚举批量拉取。
                    df = normalize_built_frame(
                        build_contract_series(_futures_variety(sym), read_only=True))
                    # 无 adj_close = build 兜底返回的 X0 原始帧（真 built 必有）
                    contract_sources[sym] = (
                        "built" if "adj_close" in df.columns else "X0-fallback")
                else:
                    df = load_symbol(sym, market=data_market, adjust=adjust) if adjust \
                        else load_symbol(sym, market=data_market)
                if df is None or df.empty:
                    records.append(_err_record(strat_name, market, sym, "无数据"))
                    continue
                # 期货 carry 类策略：on_bar 前注入该品种展期收益率序列（时序信号用自身历史）
                extra = {}
                if need_carry:
                    try:
                        carry = get_roll_yield_series(_futures_variety(sym))
                        strat.inject_carry(carry)
                        extra["carry"] = carry
                        # 第三步 carry fail-loud：腿状态分类（无源=静默退化为半信号，必须可见）
                        extra["carry_status"] = _carry_status(
                            carry, pd.Timestamp(df["date"].iloc[-1]))
                    except Exception as e:
                        records.append(_err_record(strat_name, market, sym, f"carry注入失败: {e}"))
                        continue
                sig: Signal = strat.on_bar(df)
                rec = {
                    "strategy": strat_name,
                    "market": market,
                    "symbol": sym,
                    "direction": sig.direction,
                    "target_pos": sig.target_pos,
                    "price": sig.price,
                    "note": sig.note,
                    "latest_date": str(df["date"].iloc[-1].date()),
                    # 期货因子值：CM 类策略 = 合成方向分，其余 = 最新收盘价（占位栅栏）
                    "factor": _factor_value(strat, df, sig, extra),
                }
                # 第二步口径标注：仅 built 模式的期货记录注入（X0 模式输出零回归）
                if data_market == "futures" and contract == "built":
                    rec["contract"] = contract_sources.get(sym, "built")
                # 第三步 carry 标注：仅 carry 类策略的期货记录注入（A 股零回归）
                if need_carry and "carry_status" in extra:
                    rec["carry"] = extra["carry_status"]
                # 期货记录追加头寸换算（开几手/名义/保证金）。A 股记录不追加（零回归）。
                # 注：record.market 是策略 type("cta"/"stock")，用 data_market 判期货
                if data_market == "futures":
                    rec.update(_pos_fields(cfg, sym, sig.direction, sig.target_pos,
                                           float(df["close"].iloc[-1])))
                records.append(rec)
            except Exception as e:
                records.append(_err_record(strat_name, market, sym, str(e)))

    # 第二步口径 fail-loud 汇总：built 模式兜底品种逐一点名（fetcher 侧已单品种告警）
    if contract == "built":
        fallback = [s for s, src in contract_sources.items() if src == "X0-fallback"]
        if fallback:
            print(f"[WARN] futures.contract=built 但 {len(fallback)} 个品种自建换月"
                  f"无数据、信号已退回 X0 原始主连口径（含换月跳空）: "
                  f"{','.join(fallback)}", flush=True)

    # 第三步 carry 覆盖 fail-loud：静默降级（无源/过期品种腿=0 → 组合实为半信号）
    # 比没有信号更危险——覆盖率低于阈值必须醒目告警（强化计划第三步）
    cov = _carry_coverage_line(records)
    if cov:
        print(cov, flush=True)
        n_active = sum(1 for r in records if "carry" in r and r["carry"]["active"])
        n_total = sum(1 for r in records if "carry" in r)
        if n_active < _CARRY_WARN_THRESHOLD:
            print(f"[WARN] carry 活跃品种仅 {n_active}/{n_total}（<{_CARRY_WARN_THRESHOLD}）："
                  f"无源/过期品种的组合信号已退化为 0.5×TSMOM 半信号"
                  f"（覆盖分组与裁定证据见 期货CTA_结论.md §2.2）", flush=True)

    return records


# 第三步 carry fail-loud：覆盖率告警阈值（强化计划：活跃品种 < 8 即醒目告警——
# carry 缺数据的品种组合静默退化为 0.5×TSMOM 半信号，比没有信号更危险）
_CARRY_WARN_THRESHOLD = 8
# carry 新鲜度容差（日历天）：展期序列末值滞后价格数据超此值 → 过期（腿信号吃旧值）
_CARRY_STALE_DAYS = 14
# 完整/短史分界（carry 缓存实测两组：2018-01-02 起 vs 2021-09-22 起，中间无中间态）
_CARRY_FULL_SINCE = pd.Timestamp("2021-01-01")


def _carry_status(carry_df, price_last) -> dict:
    """carry 腿状态分类（纯函数，网络无关）。

    返回 {status, active, start, end, lag_days}：
      - 无源：序列空（腿=0，组合该品种实为 0.5×TSMOM 半信号）
      - 完整(2018起) / 短史(2021-09起)：按序列起点分组
      - 过期(N天)：序列末值滞后价格末日 > _CARRY_STALE_DAYS（腿信号吃旧值）
    active = 非空且未过期——覆盖率的分子。
    """
    if carry_df is None or len(carry_df) == 0:
        return {"status": "无源", "active": False, "start": None,
                "end": None, "lag_days": None}
    d = pd.to_datetime(carry_df["date"])
    start, end = d.iloc[0], d.iloc[-1]
    lag = (pd.Timestamp(price_last) - end).days
    if lag > _CARRY_STALE_DAYS:
        return {"status": f"过期(滞后{lag}天)", "active": False,
                "start": start, "end": end, "lag_days": lag}
    grp = "完整(2018起)" if start <= _CARRY_FULL_SINCE else "短史(2021-09起)"
    return {"status": grp, "active": True, "start": start, "end": end,
            "lag_days": lag}


def _carry_coverage_line(records: list[dict]) -> str | None:
    """从信号记录聚合 carry 覆盖率一行（无 carry 记录时返回 None——A 股零回归）。"""
    stats = [r["carry"] for r in records if "carry" in r]
    if not stats:
        return None
    total = len(stats)
    active = sum(1 for s in stats if s["active"])
    by_grp: dict[str, int] = {}
    for s in stats:
        key = "无源" if s["status"] == "无源" else (
            "过期" if s["status"].startswith("过期") else (
                "完整" if s["status"].startswith("完整") else "短史"))
        by_grp[key] = by_grp.get(key, 0) + 1
    parts = " / ".join(f"{k} {v}" for k, v in sorted(by_grp.items()))
    return f"carry 覆盖: {active}/{total} 品种活跃（{parts}）"


def _factor_value(strat, df, sig: Signal, extra: dict):
    """期货记录追加因子值栏（不破坏 A 股记录格式，A股继续无此键）。

    取值：TSMOM+Carry / 多周期集成等容器策略取内部加权方向分；TripleFactor 取软市况权重合成分；
    Carry 取最新平滑展期收益；其余取最新行情（收盘/结算），供人工核对。
    """
    try:
        if type(strat).__name__ in ("TsmomCarryStrategy", "TripleFactorStrategy",
                                    "TsmomEnsembleStrategy"):
            return round(strat._score(strat, df), 3)
        if type(strat).__name__ == "CarryStrategy":
            cs = strat._carry_series
            if cs is not None and not cs.empty:
                roll = cs.sort_values("date")["roll_yield"].astype(float)
                return round(float(roll.iloc[-1]), 6)
            return None
        for col in ("settle", "close"):
            if col in df.columns:
                return float(df[col].iloc[-1])
        return None
    except Exception:
        return None


def _pos_fields(cfg: dict, sym: str, direction: str, target_pos: float,
                price: float) -> dict:
    """期货记录 append 头寸换算字段（开几手/名义/保证金）。

    主流期货流程「信号 → 实际头寸」：把单品种方向 × target_pos（带符号权重）
    经 pos_size 换成手数/名义/保证金。A 股记录不调用本函数（零回归门）。
    """
    try:
        var = _futures_variety(sym)
        fut = cfg.get("futures", {})
        nav = float(fut.get("nav", 1_000_000.0))
        default_mult = fut.get("default_multiplier")
        # 规则表 + 配置兜底（multiplier 缺失用 default_multiplier；margin %→小数）
        rules = get_futures_rules(read_only=True)
        mult = {}
        margin = {}
        if var in rules:
            mult[var] = rules[var].get("multiplier")
            mr = rules[var].get("margin_ratio")
            if mr:
                margin[var] = mr / 100.0
        if not margin.get(var):
            margin[var] = fut.get("margin", 0.1)
        cap_util = float(fut.get("risk", {}).get("capital_utilization", 0.9) or 0.9)
        w = {"long": target_pos, "short": -target_pos}.get(direction, 0.0)
        pos = pos_size({var: w}, nav, {var: price}, mult, margin,
                       cap_util=cap_util, default_multiplier=default_mult)
        p = pos.get(var, {})
        return {
            "nav": nav,
            "lots": p.get("lots", 0),
            "notional": round(float(p.get("notional", 0.0)), 2),
            "margin": round(float(p.get("margin", 0.0)), 2),
        }
    except Exception as e:
        # 审计 #19 修复：静默归零会把"算不出来"伪装成"0 手可执行"——打印失败品种，
        # 落盘值仍为 0（不中断报告），但失败可见、可人工介入
        print(f"[WARN] _pos_fields: {sym} 头寸换算失败（按 0 手输出）: {e}", flush=True)
        return {"lots": 0, "notional": 0.0, "margin": 0.0}


def _err_record(strat_name, market, sym, msg):
    return {"strategy": strat_name, "market": market, "symbol": sym,
            "direction": "error", "target_pos": 0, "price": None,
            "note": f"出错: {msg}", "latest_date": "-"}


def format_report(records: list[dict]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"  每日交易信号  生成时间: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("=" * 60)

    # 方向标记
    mark = {"long": "▲ 做多", "short": "▼ 做空", "flat": "■ 平仓",
            "hold": "— 观望", "error": "✗ 错误"}

    for r in records:
        d = mark.get(r["direction"], r["direction"])
        price = f"{r['price']:.2f}" if r["price"] else "-"
        lines.append(f"\n[{r['market']}] {r['symbol']}  <{r['strategy']}>")
        lines.append(f"  方向   : {d}")
        lines.append(f"  目标仓 : {r['target_pos']:.0%}")
        lines.append(f"  参考价 : {price}   (数据截至 {r['latest_date']})")
        # 审计 #19 修复：lots 只在 data_market=="futures" 时注入（_pos_fields 同口径）；
        # r["market"] 存的是策略 type（"cta"），判 =="futures" 恒假 → 手数算好不渲染。
        # 以 "lots" in r 作判据与注入条件逐字对齐（错误记录/A 股记录均无 lots）。
        if "lots" in r:
            lines.append(f"  手数   : {r['lots']} 手  ({r['direction']})")
            lines.append(f"  名义   : {r['notional']:.0f}  保证金: {r['margin']:.0f}")
        # 第二步口径标注：仅 built 模式注入（有该键才渲染），兜底品种带 ⚠️
        if "contract" in r:
            src = r["contract"]
            tag = "⚠️ X0 兜底（含换月跳空）" if src == "X0-fallback" else "built 复权"
            lines.append(f"  口径   : {tag}")
        # 第三步 carry 标注：仅 carry 类策略记录注入（有该键才渲染）；无源=半信号必须可见
        if "carry" in r:
            st = r["carry"]["status"]
            tag = f"{st} → carry腿=0（实为0.5×TSMOM 半信号）" if st == "无源" else st
            lines.append(f"  carry  : {tag}")
        lines.append(f"  原因   : {r['note']}")

    # 汇总可操作信号
    actionable = [r for r in records if r["direction"] in ("long", "short", "flat")]
    lines.append("\n" + "-" * 60)
    lines.append(f"可操作信号: {len(actionable)} 条")
    for r in actionable:
        d = mark[r["direction"]]
        lines.append(f"  {d:8} {r['market']:6} {r['symbol']}  ({r['note']})")
    # 第三步 carry 覆盖率汇总（有 carry 记录才渲染——A 股零回归）
    cov = _carry_coverage_line(records)
    if cov:
        lines.append(cov)
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = run_daily(cfg)
    report = format_report(records)

    out_path = Path(cfg.get("daily_signal", {}).get("output", "reports/daily_signals.txt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n已保存到: {out_path}")


if __name__ == "__main__":
    main()
