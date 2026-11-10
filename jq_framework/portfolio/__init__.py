"""期货头寸换算层包（回测与每日信号共用单源）。

定位：主流期货流程「信号 → 仓位权重 → 实际头寸（手数/名义/保证金/资金）」的落地点，
把数据层已拉到但此前从不消费的 multiplier（合约乘数）用起来。放独立 portfolio/ 包，
不属 strategies 信号层，也不改 backtest/walk_forward——回测引擎只 import 函数，不动 A 股。
"""
