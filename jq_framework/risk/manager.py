"""
风控模块：止盈止损 + 波动率加权仓位。

设计原则：
- 与选股逻辑解耦：选股决定"买什么"，风控决定"何时卖/买多少"
- 信号 T 日收盘产生，T+1 日开盘成交（见 portfolio_runner）
- 止盈止损在每日开盘前检查，触发则当日开盘卖出

持仓状态（portfolio_runner 维护）：
  holdings: {symbol: {"weight": 权重, "entry_price": 买入价, "entry_date": 买入日}}
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RiskManager:
    """
    风控管理器。集成止盈止损与波动率加权仓位。

    参数:
      stop_loss: 个股止损比例，如 0.10 表示从买入价下跌10%卖出（None=不止损）
      take_profit: 个股止盈比例，如 0.30 表示上涨30%卖出（None=不止盈）
      target_vol: 目标年化波动率，如 0.15。用过去lookback日波动率反推仓位倍数
      lookback: 波动率回看天数
      max_leverage: 仓位倍数上限（防极端）
    """
    def __init__(self, stop_loss: float | None = 0.10,
                 take_profit: float | None = 0.30,
                 target_vol: float | None = 0.15,
                 lookback: int = 20,
                 max_leverage: float = 1.5,
                 stop_loss_mode: str = "fixed",
                 stop_vol_k: float = 1.0,
                 stop_vol_min_days: int = 10,
                 trail_pct: float | None = None,
                 trail_activate: float | None = None,
                 sl_trail_pct: float | None = None,
                 sl_trail_activate: float | None = None):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.target_vol = target_vol
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.stop_loss_mode = stop_loss_mode  # "fixed" | "vol"（波动自适应）
        self.stop_vol_k = stop_vol_k
        self.stop_vol_min_days = stop_vol_min_days  # 持有天数下限，防持有初期阈值过窄误杀
        # 路径类(trailing)止盈止损。无状态：峰值浮盈从 df 的[entry_date,date]区间推导。
        # --- 止盈侧 trail（锁大利润回吐）---
        # trail_pct：从峰值浮盈回落该比例即落袋。trail_activate：峰值浮盈需先达该激活线才启动
        #（默认=trail_pct，但生产应设高激活线如+15%，避免对微利头寸过早设回撤线误杀）。
        self.trail_pct = trail_pct
        self.trail_activate = (trail_activate if trail_activate is not None
                               else (trail_pct if trail_pct is not None else None))
        # --- 止损侧 trail（破位清死钱）：曾盈利的头寸回撤转亏即清，不等 rank 被动换出 ---
        # sl_trail_pct：从峰值盈利转亏的清理幅度（0=跌回成本即清）。sl_trail_activate：需峰值浮盈先达此线
        #（默认 None=关闭整条，仅用静态/波动止损兜底）。
        self.sl_trail_pct = sl_trail_pct
        self.sl_trail_activate = (sl_trail_activate if sl_trail_activate is not None
                                  else self.trail_activate)

    def _rolling_vol(self, df: pd.DataFrame, date: str) -> float | None:
        """过去 lookback 日已实现波动率(日 std)，口径与 vol_weight_multiplier 一致。无则 None。

        严格 < date（不含执行日当行）：止损决策发生在 date 开盘、当日收盘不可知，
        含当日行会把 close(T)/close(T-1) 前视进 vol 档位（审计#1 同类，2026-09-24 修）。
        """
        if df is None or len(df) < self.lookback + 1:
            return None
        sub = df[df["date"] < date]
        if len(sub) < self.lookback + 1:
            return None
        rets = sub["close"].pct_change().iloc[-self.lookback:]
        vol = rets.std()
        if vol <= 0 or np.isnan(vol):
            return None
        return float(vol)

    def _stop_threshold(self, vol: float | None, hold_days: int = 1) -> float | None:
        """
        止损阈值(相对 entry 的累积浮亏幅度，与 -10% 同量纲)。
        fixed: 固定 self.stop_loss。
        vol: k × 日波动率 × √持有天数，下限天数保护防止持有初期过窄误杀。
          高波动票阈值宽、低波动窄、老仓更宽——直接回应静态 -10% 误杀震荡票。
        """
        if self.stop_loss_mode == "vol":
            if vol is None:
                return None  # 无波动率数据时不止损(不误杀)
            days = max(hold_days, self.stop_vol_min_days)
            return self.stop_vol_k * vol * np.sqrt(days)
        return self.stop_loss

    def check_stop(self, current_price: float, entry_price: float,
                   vol: float | None = None, hold_days: int = 1) -> str | None:
        """
        检查个股是否触发止盈止损。
        返回: "stop_loss" / "take_profit" / None
        vol: 个股当日已实现波动率(波动自适应止损用)。fixed 模式忽略。
        hold_days: 已持有交易天数(波动模式累积幅度用)。fixed 模式忽略。
        """
        if entry_price <= 0 or current_price <= 0:
            return None
        pnl = current_price / entry_price - 1
        thr = self._stop_threshold(vol, hold_days)
        if self.stop_loss is not None and thr is not None and pnl <= -thr:
            return "stop_loss"
        if self.take_profit is not None and pnl >= self.take_profit:
            return "take_profit"
        return None

    def _hold_days(self, df: pd.DataFrame, date, entry_date) -> int:
        """自 entry_date 到 date 的交易日数(含当日)。无 entry_date 或缺失时回退 1。"""
        if entry_date is None:
            return 1
        try:
            dates = df["date"].to_numpy()
            t_d = pd.Timestamp(date)
            e_d = pd.Timestamp(entry_date)
            cnt = int((dates <= t_d).sum() - (dates <= e_d).sum()) + 1
            return max(cnt, 1)
        except Exception:
            return 1

    def _peak_pnl(self, df: pd.DataFrame, date, entry_price, entry_date=None) -> float | None:
        """自入场到 date 前一收盘的峰值浮盈(收盘口径)。无足够数据或无效入场价返回 None。

        严格 < date：trail 决策发生在 date 开盘，当日收盘不可知不可作峰值候选
        （审计#1 同类前视，2026-09-24 修；原 <= 会用当日收盘抬高 peak 提前触发 trail）。
        """
        if entry_price is None or entry_price <= 0:
            return None
        try:
            dates = df["date"].to_numpy()
            t_d = pd.Timestamp(date)
            e_d = pd.Timestamp(entry_date) if entry_date is not None else dates[0]
            mask = (dates < t_d) & (dates >= e_d)
            if mask.sum() == 0:
                return None
            peak_close = df.loc[mask, "close"].max()
            return float(peak_close / entry_price - 1)
        except Exception:
            return None

    def _decide_exit(self, df, date, open_price, entry_price, entry_date=None):
        """统一退出决策：trail(路径)优先，其次静态止盈/止损(或波动止损)。
        返回 ("take_profit"|"stop_loss"|None)，与引擎 reason 口径一致。
        trail 分两条，均只在"大盈利/明确破位"才触发，不对微利头寸设线：
          - 止盈锁利润：peak_pnl>=trail_activate(高激活线) 且 cur<=peak-trail_pct → take_profit
          - 破位清死钱：peak_pnl>=sl_trail_activate 且 cur<=peak-sl_trail_pct(默认0即回撤转亏) → stop_loss
        """
        cur_pnl = open_price / entry_price - 1
        # 1a) 止盈侧 trail：锁大利润回吐（需先赚够 trail_activate）
        if self.trail_pct is not None:
            peak_pnl = self._peak_pnl(df, date, entry_price, entry_date)
            if peak_pnl is not None and peak_pnl >= self.trail_activate:
                if cur_pnl <= peak_pnl - self.trail_pct:
                    return "take_profit"
        # 1b) 止损侧 trail：曾盈利的头寸当前跌回成本线以下(转亏) → 破位清死钱
        if self.sl_trail_pct is not None:
            peak_pnl = self._peak_pnl(df, date, entry_price, entry_date)
            if peak_pnl is not None and peak_pnl >= self.sl_trail_activate:
                # sl_trail_pct=0 → 跌回成本即清；>0 → 成本以下更深才清；<0 → 成本上方提前清
                if cur_pnl <= -self.sl_trail_pct:
                    return "stop_loss"
        # 2) 静态/波动止盈止损（止盈同现，止损随 stop_loss_mode）
        vol = self._rolling_vol(df, date) if self.stop_loss_mode == "vol" else None
        hold_days = self._hold_days(df, date, entry_date) if self.stop_loss_mode == "vol" else 1
        return self.check_stop(open_price, entry_price, vol=vol, hold_days=hold_days)

    def filter_stops(self, holdings: dict, date: str,
                     price_lookup: dict[str, pd.DataFrame]) -> list[str]:
        """
        返回当日应因止盈止损卖出的股票列表。
        price_lookup: {symbol: df}，用 df 当日 open 作为检查价（开盘卖出）
        """
        to_sell = []
        for sym, info in holdings.items():
            df = price_lookup.get(sym)
            if df is None:
                continue
            row = df[df["date"] == date]
            if row.empty:
                continue
            open_price = row["open"].iloc[0]
            reason = self._decide_exit(df, date, open_price, info["entry_price"], info.get("entry_date"))
            if reason:
                to_sell.append(sym)
        return to_sell

    def check_stop_for_date(self, sym: str, date: str,
                            price_lookup: dict[str, pd.DataFrame],
                            entry_price: float, entry_date=None) -> str | None:
        """按日期粒度算退出 reason（供引擎在 filter_stops 之后复算，口径一致）。
        与 filter_stops 用同一 _decide_exit，trail/vol 模式 reason 一致。"""
        df = price_lookup.get(sym)
        if df is None:
            return None
        row = df[df["date"] == date]
        if row.empty:
            return None
        open_price = row["open"].iloc[0]
        return self._decide_exit(df, date, open_price, entry_price, entry_date)

    def vol_weight_multiplier(self, sym: str, date: str,
                              price_lookup: dict[str, pd.DataFrame]) -> float:
        """
        计算波动率加权的仓位倍数。
        target_vol / 实际波动率，截断到 [0, max_leverage]。
        波动率高 → 降仓；波动率低 → 加仓。

        严格 < date（不含执行日当行）：调用点在 T+1 开盘执行块，date 开盘时刻当日收盘
        不可知，<= 会把 close(T+1)/close(T) 前视进当日仓位权重（审计#1，2026-09-24 修；
        修复后权重只由 T 及之前的收盘决定，开盘时点可复现）。
        """
        if self.target_vol is None:
            return 1.0
        df = price_lookup.get(sym)
        if df is None or len(df) < self.lookback + 1:
            return 1.0
        sub = df[df["date"] < date]
        if len(sub) < self.lookback + 1:
            return 1.0
        rets = sub["close"].pct_change().iloc[-self.lookback:]
        vol = rets.std()
        if vol <= 0 or np.isnan(vol):
            return 1.0
        # 年化波动率 = 日波动率 * sqrt(252)
        ann_vol = vol * np.sqrt(252)
        mult = self.target_vol / ann_vol
        return float(np.clip(mult, 0.0, self.max_leverage))
