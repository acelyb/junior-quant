# junior-quant

面向 **A 股 + 期货** 的量化研究框架。

**它不是策略。** 仓库里只有框架和教学用的示例因子/策略，真实的策略实现与参数不在这里。clone 下来可以直接跑通全流程，然后换上你自己的东西。

```
数据 → 因子 → 组合 → 回测/信号
```

---

## 快速开始

```bash
git clone https://github.com/acelyb/junior-quant.git
cd junior-quant
pip install -e .

python -m jq_framework.examples.run_demo      # 端到端示例，合成数据，不联网
python -m unittest discover -s tests/futures -p "test_*.py"    # 124 例
```

`run_demo` 会走完：因子计算 → 截面处理 → 加权合成 → 策略信号 → 契约自检。

---

## 三个命令式工具

```bash
# 每日信号：输出可执行手数清单
python scripts/futures_combination_signal.py

# 一键编排：拉数 → 回测 → 健康卡 → 对账
python scripts/futures_pipeline.py

# 因子体检：留一法增量 IR + 残差检验 + 尾部相关
python scripts/futures_factor_health.py
```

都支持 `--help`、`--config`、`--read-only`（只读缓存不拉网）。

> 注：`futures_factor_health.py` 的因子池来自配置的 `futures.factor_health.factors`，
> 示例配置里是空的，需填入你自己的因子。

---

## 目录

```
jq_framework/
├── data/fetchers/      数据层（股票 / 期货 / 基本面 / 分钟 / 快照）
├── factors/            因子框架：base、composite、ic、特征工程、动态池
├── strategies/         StrategyBase + backtrader 适配
├── backtest/           单标的 / 组合 / 分窗回测引擎
├── risk/               风控：vol_target、CPPI、择时、期货四步风控
├── portfolio/          期货手数与保证金换算
├── signals/            每日信号（含 fail-loud）
├── config/loader.py    配置发现链
└── examples/run_demo.py
```

约 11,900 行，124 例单元测试。

---

## 写一个因子

```python
from jq_framework.factors.base import FactorBase
import pandas as pd

class MyFactor(FactorBase):
    name = "my_factor"

    def calc(self, df: pd.DataFrame) -> float | None:
        # df：到当前时点为止的日线
        if len(df) < 21:
            return None
        return df["close"].iloc[-1] / df["close"].iloc[-21] - 1
```

在配置里注册：

```yaml
multifactor:
  factors:
    my_factor:
      module: "my_package.my_factor"
      class: "MyFactor"
      family: "trend"
      weight: 1.0
```

**两条硬约束**

1. `calc` 只能用 `df` 最后一行及以前的数据 —— 否则是未来函数
2. 负向因子（波动率、换手率等）**必须取负号** —— 框架约定"值越大越好"

## 写一个策略

```python
from jq_framework.strategies.base import Signal, StrategyBase
import pandas as pd

class MyStrategy(StrategyBase):
    default_params = {"window": 20}

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        if len(bars) < self.params["window"] + 1:
            return Signal(direction="hold")
        # ... 你的逻辑
        return Signal(direction="long", target_pos=1.0, price=price)
```

**一条硬约束**：**不要保存跨 bar 状态**。分窗引擎会逐段重建策略实例，状态会丢。

---

## 配置

按顺序找第一个存在的：

```
1. --config 指定的路径
2. $JQ_CONFIG 环境变量
3. configs/local.yaml     ← 你自己的
4. configs/example.yaml   ← 仓库自带
```

```bash
cp configs/example.yaml configs/local.yaml
```

---

## 为什么值得看

**1. 期货换月拼接**（`data/fetchers/fetcher_futures.py`）

中国商品期货连续合约的坑：选错主力合约会让 `hold` 差 70 倍，而 `close` 相关系数仍有 0.997（错误被掩盖）；换月不复权会让长周期累计收益被假跳空翻转。这里做了逐日按持仓量选主力 + 换月日 rebase 复权 + 双源对账。

**2. 留一法因子治理**（`scripts/futures_factor_health.py`）

普通回测只说"整体赚不赚钱"。L1O 逐因子回答"删掉它组合会变好还是变差"，配两个诊断列：

- **残差检验** —— 这个因子的收益有多少只是跟着主因子一起涨
- **尾部相关** —— 暴跌那天它是不是也在崩

**3. 分窗 + 逐段重置**

`walk_forward.py` 每个训练窗口重算因子方向，测试段严格样本外；策略实例逐段重建，杜绝跨段状态泄漏。

---

## 关于 private/

`private/` 是一个 submodule，指向作者的私有实现。**没有访问权限不影响使用**——`git clone` 后它是空目录，框架自动回落到 `configs/example.yaml`。

---

## 诚实说明

这个框架**不展示盈利能力**。

作者用它检验过的想法里，绝大多数在样本外被否：支撑压力线、趋势线、量价背离、放量确认、季节性、截面动量、多周期集成。期货线最终只留了一个时序动量单腿。

价值在方法 —— 预注册口径、多重比较校正、分品种检验、前后半同号、留一法归因。

**如果你在找赚钱的策略，这里没有。如果你在建自己的研究流程，这里应该能省你一些时间。**

---

## 数据源与许可

数据通过 [akshare](https://github.com/akfamily/akshare) 等公开源获取，请遵守各数据源条款。本仓库**不包含行情数据**。

[AGPL-3.0](LICENSE)
