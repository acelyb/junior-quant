"""
回测 runner（B 路径）。

输入：标的 + 策略 + 时间段
输出：绩效报告（收益、回撤、夏普、交易次数）+ 资金曲线图

用法：
  python -m backtest.runner --config config.yaml --strategy dual_ma_stock
  python -m backtest.runner --config config.yaml --strategy dual_ma_cta --symbol RB0
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import backtrader as bt
import pandas as pd
import yaml

from jq_framework.data.fetchers.fetcher import load_symbol
from jq_framework.strategies.adapter import BacktraderAdapter


def load_config(path: str | None = None, market: str | None = None) -> dict:
    """加载配置。path=None 时走发现链（private/configs → configs → config.yaml）。"""
    from jq_framework.config.loader import load_config as _framework_load
    return _framework_load(path, market)


def build_strategy(cfg: dict, strat_name: str):
    """根据 config 里的 strategies 段，实例化一个 StrategyBase。"""
    strat_cfg = cfg["strategies"][strat_name]
    module = importlib.import_module(strat_cfg["module"])
    cls = getattr(module, strat_cfg["class"])
    return cls(params=strat_cfg.get("params", {})), strat_cfg["type"]


class PandasData(bt.feeds.PandasData):
    """让 backtrader 直接吃我们的 DataFrame。"""
    params = (
        ("datetime", None),  # 用 index
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


def run_backtest(cfg: dict, strat_name: str, symbol: str | None = None):
    strat, market = build_strategy(cfg, strat_name)

    bt_cfg = cfg["backtest"]

    # 决定标的与数据
    if market == "stock":
        symbols = cfg["stock"]["symbols"]
        adjust = cfg["stock"].get("adjust", "qfq")
        data_fn = lambda s: load_symbol(s, market="stock", start=bt_cfg["start"],
                                        end=bt_cfg["end"], adjust=adjust)
    else:
        symbols = cfg["futures"]["symbols"]
        data_fn = lambda s: load_symbol(s, market="futures")

    if symbol:
        symbols = [symbol]

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(bt_cfg["cash"])
    cerebro.broker.setcommission(commission=bt_cfg["commission"])

    for sym in symbols:
        df = data_fn(sym)
        df = df.set_index("date")
        feed = PandasData(dataname=df)
        cerebro.adddata(feed, name=sym)

    # 单标的时直接加策略；多标的暂只对第一个标的跑（保持简单）
    cerebro.addstrategy(BacktraderAdapter, strat=strat, market=market)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    print(f"=== 回测: {strat_name} on {symbols} ({market}) ===")
    print(f"    起始资金: {bt_cfg['cash']:,.0f}")

    results = cerebro.run()
    strat_inst = results[0]
    final = cerebro.broker.getvalue()

    # 绩效
    sharpe = strat_inst.analyzers.sharpe.get_analysis()
    dd = strat_inst.analyzers.drawdown.get_analysis()
    trades = strat_inst.analyzers.trades.get_analysis()
    rets = strat_inst.analyzers.returns.get_analysis()

    pnl = final - bt_cfg["cash"]
    pnl_pct = pnl / bt_cfg["cash"] * 100

    print("\n=== 绩效报告 ===")
    print(f"    期末资金 : {final:,.0f}")
    print(f"    净收益   : {pnl:,.0f}  ({pnl_pct:+.2f}%)")
    print(f"    夏普比率 : {sharpe.get('sharperatio', 'N/A')}")
    print(f"    最大回撤 : {dd.get('drawdown', 'N/A')}%")
    print(f"    总交易数 : {trades.get('total', {}).get('total', 0)}")
    print(f"    年化收益 : {rets.get('rnorm100', 'N/A')}%")

    # 画资金曲线（保存到 reports/）
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    fig_path = reports_dir / f"backtest_{strat_name}.png"
    try:
        cerebro.plot(style="candlestick", volume=False)[0][0].savefig(str(fig_path))
        print(f"\n    资金曲线图已保存: {fig_path}")
    except Exception as e:
        print(f"\n    画图失败（无显示环境时常见，不影响回测结果）: {e}")

    return {"final": final, "pnl_pct": pnl_pct, "sharpe": sharpe, "dd": dd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--strategy", required=True, help="config 中的策略名")
    parser.add_argument("--symbol", default=None, help="可选，指定单个标的")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_backtest(cfg, args.strategy, args.symbol)


if __name__ == "__main__":
    main()
