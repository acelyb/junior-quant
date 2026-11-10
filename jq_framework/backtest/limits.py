"""
A 股涨跌停成交可行性判断（成交执行语义）。

回测引擎在"次日开盘成交"时调用 `is_tradeable` 判断该股当日开盘能否实际成交：
- 涨停开盘锁死（open==high、开盘即涨停）→ 买不到，买入应跳过
- 跌停开盘锁死（open==low、开盘即跌停）→ 卖不掉，卖出应保留持仓次日重试

旧回测引擎按锁死开盘价"成交"=无法成交的虚假成交，系统性高估收益。本模块只拒绝
"已确认的涨跌停锁死"，缺失行/停牌/数据缺失一律返回 True（既有 op-is-None 路径已处理）。

仅被 backtest/walk_forward.py 与 backtest/portfolio_runner.py 使用（成交执行路径），
不做 IC 分析、不做信号——故不归 data/（纯数据）或 risk/（止盈止损/波动率）。
"""
from __future__ import annotations

import pandas as pd


def _limit_threshold(sym: str) -> float:
    """板块涨跌停幅度：科创板/创业板 0.20，主板 0.10。"""
    # 688=科创板, 300/301=创业板(20%); 600/601/603/605/000/001/002/003=主板(10%)
    if sym.startswith(("688", "300", "301")):
        return 0.20
    return 0.10


def is_tradeable(sym: str, date, valid: dict[str, pd.DataFrame], direction: str,
                 tol: float = 0.001) -> bool:
    """`sym` 在 `date` 开盘能否成交（按 A 股涨跌停规则）？

    direction="buy":  开盘相对昨收达/超涨停阈值 → False（开盘锁死，买不到）。
    direction="sell": 开盘相对昨收达/超跌停阈值 → False（开盘锁死，卖不掉）。

    检测：open/prev_close - 1 与 (threshold - tol) 比。tol=0.001 容 qfq 复权四舍五入，
    使 +9.95% 的真涨停仍被识别（10% 板阈值线 9.9%、20% 板 19.9%）。+9.9% 的真实高开
    会被误拒——罕见且代价是持现金，按规范("0.099 not 0.0995")可接受。

    数据缺失（无行/无昨收）→ 返回 True。既有 op-is-None 路径已处理缺失行，
    本函数只拒绝"已确认的涨跌停锁死"，缺失/停牌路径行为字节不变。

    性质：无状态、纯 pandas → 线程安全（回测循环单线程）；无 path/os/文件 I/O →
    Windows 安全；每次 2 次 df 查找，只在调仓/成交日触发，无新瓶颈。
    """
    df = valid.get(sym)
    if df is None:
        return True
    row = df[df["date"] == date]
    if row.empty:
        return True
    prev = df[df["date"] < date]
    if prev.empty:
        return True
    open_price = row["open"].iloc[0]
    prev_close = prev["close"].iloc[-1]
    if prev_close <= 0 or pd.isna(open_price) or pd.isna(prev_close):
        return True
    thresh = _limit_threshold(sym) - tol
    change = open_price / prev_close - 1
    if direction == "buy":
        return change < thresh
    return change > -thresh
