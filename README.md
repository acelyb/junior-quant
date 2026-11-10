# junior-quant

一个面向 **A 股 + 期货** 的量化研究框架：数据获取 → 因子计算 → 组合合成 → 回测 → 每日信号。

**它不是一个策略。** 仓库里只有框架和一个教学用的示例策略，真正的策略实现与参数在作者的私有模块中。你 clone 下来可以直接跑通全流程，然后换上自己的因子和策略。

```
数据层 → 因子 → 组合打分 → 回测/信号
```

---

## 为什么值得一看

三个部分在个人量化项目里少见：

**1. 期货换月拼接的完整实现**（`data/fetchers/fetcher_futures.py`）

中国商品期货连续合约的坑很多：选错主力合约会让 `hold` 差 70 倍而 `close` 相关系数仍有 0.997（错误被掩盖）；换月不复权会让 250 日累计收益被假跳空翻转。这里实现了逐日按持仓量选主力 + 换月日 rebase 复权 + 双源对账。

**2. Leave-One-Out 因子治理**（`scripts/futures_factor_health.py`）

普通回测只说"整体赚不赚钱"。L1O 逐因子回答"删掉它组合会变好还是变差"：

```
删掉因子 A → 组合 IR 变差  → 它是真的在贡献，keep
删掉因子 B → 组合 IR 变好  → 它在拖累组合，drop
```

配套两个"看穿马甲"诊断列：**残差检验**（这个因子的收益有多少只是跟着主因子一起涨）和**尾部相关**（暴跌那天它是不是也在崩）。很多因子就是被残差检验识破的。

**3. 分窗 + 逐段重置的回测引擎**

`walk_forward.py` 每个训练窗口重算因子方向，测试段严格样本外；策略实例逐段重建，杜绝跨段状态泄漏。

---

## 快速开始

```bash
git clone https://github.com/acelyb/junior-quant.git
cd junior-quant
pip install -e .

# 端到端示例（合成数据，不联网）
python -m jq_framework.examples.run_demo
```

输出：

```
[1] 因子计算与截面处理
    momentum20   权重=1.0  截面 20 个有效值，前三: SYM05=1.000, ...
[2] 加权合成为组合打分
[3] 单标的策略信号
[4] 契约自检
```

跑测试：

```bash
python -m unittest discover -s tests/futures -p "test_*.py"   # 127 例全绿
```

---

## 结构

```
junior-quant/
├── jq_framework/              框架本体（约 11,900 行）
│   ├── data/fetchers/         数据获取
│   │   ├── fetcher.py              股票日线
│   │   ├── fetcher_futures.py      期货（换月拼接 / 复权 / 展期收益率）
│   │   ├── fetcher_fundamental.py  基本面（按公告日对齐，防未来函数）
│   │   ├── fetcher_minute.py       分钟线
│   │   └── snapshot.py             数据快照冻结（可复现实验）
│   ├── risk/                  风控
│   │   ├── manager.py              A股风控
│   │   ├── futures_risk.py         期货四步风控
│   │   ├── vol_target.py           波动率目标
│   │   ├── cppi.py                 CPPI 保护
│   │   └── timing.py               择时
│   ├── factors/               因子框架
│   │   ├── base.py                 FactorBase + 截面处理
│   │   ├── composite.py            两层合成（族内正交+ICIR → 族间风险平价）
│   │   ├── composite_ml.py         ML 合成
│   │   ├── ic.py                   IC / ICIR 计算
│   │   ├── feature_engineering.py  非线性特征派生
│   │   ├── universe_expansion.py   可投资域扩充
│   │   ├── dynamic_pool.py         逐日动态训练池
│   │   └── examples/               示例因子
│   ├── strategies/            策略框架
│   │   ├── base.py                 StrategyBase（on_bar → Signal）
│   │   ├── adapter.py              backtrader 适配
│   │   └── examples/               示例策略
│   ├── backtest/              回测引擎
│   │   ├── runner.py               单标的回测
│   │   ├── portfolio_runner.py     组合回测
│   │   ├── walk_forward.py         A股分窗引擎
│   │   ├── futures_walk_forward.py 期货分窗引擎
│   │   ├── direction.py            方向策略（IC → 权重）
│   │   └── limits.py               涨跌停/停牌约束
│   ├── portfolio/             仓位换算
│   │   └── futures_sizing.py       期货手数/保证金
│   ├── signals/               每日信号
│   │   ├── daily_runner.py         期货信号（含 fail-loud）
│   │   ├── daily_portfolio.py      A股信号
│   │   ├── ic_redline.py           滚动 IC 红线监控
│   │   └── ml_model_store.py       ML 模型加载
│   ├── config/loader.py       配置发现链
│   └── examples/run_demo.py   端到端示例
│
├── scripts/                   命令行工具（20 个）
│   ├── futures_*.py                期货信号/回测/对账入口
│   ├── snapshot*.py                数据快照
│   ├── *_smoke.py                  离线冒烟测试
│   ├── guard_public.py             公开仓库守卫
│   └── install_guard.sh            守卫安装
│
├── tests/
│   ├── futures/               框架测试（127 例，可独立运行）
│   └── integration/           集成测试（224 例，需私有模块）
│
├── configs/example.yaml       示例配置（开箱可跑）
├── data/                      缓存目录骨架（.gitkeep）
├── research/attribution.py    收益归因工具
├── private/                   私有 alpha 模块（submodule）
├── pyproject.toml
├── LICENSE                    AGPL-3.0
└── README.md
```

约 **11,900 行**框架代码，**127 例**框架测试。

另有 224 例集成测试在 `tests/integration/`，需配合私有模块运行。

---

## 写自己的因子

```python
from jq_framework.factors.base import FactorBase
import pandas as pd

class MyFactor(FactorBase):
    name = "my_factor"

    def calc(self, df: pd.DataFrame) -> float | None:
        # df = 到当前时点为止的日线
        # 约定：值越大 → 预期未来收益越高（负向因子要取负号）
        if len(df) < 21:
            return None
        return df["close"].iloc[-1] / df["close"].iloc[-21] - 1
```

配置里注册：

```yaml
multifactor:
  factors:
    my_factor:
      module: "my_package.my_factor"
      class: "MyFactor"
      family: "trend"
      weight: 1.0
```

**两条硬约束**：
- `calc` 只能用到 `df` 最后一行及以前的数据（否则未来函数）
- 负向因子（波动率、换手率等）必须取负号，否则合成方向反了

## 写自己的策略

```python
from jq_framework.strategies.base import Signal, StrategyBase
import pandas as pd

class MyStrategy(StrategyBase):
    default_params = {"window": 20}

    def on_bar(self, bars: pd.DataFrame) -> Signal:
        # bars = 到当前 bar 为止的历史
        if len(bars) < self.params["window"] + 1:
            return Signal(direction="hold")
        ...
        return Signal(direction="long", target_pos=1.0, price=price)
```

**一条硬约束**：**不要保存跨 bar 状态**。分窗引擎会逐段重建策略实例，状态会丢。需要状态时参考包装器模式。

---

## 配置

配置有发现链，按顺序找第一个存在的：

```
1. CLI --config 指定的路径
2. $JQ_CONFIG 环境变量
3. configs/local.yaml        ← 你自己的
4. configs/example.yaml      ← 仓库自带的示例
```

复制一份改：

```bash
cp configs/example.yaml configs/local.yaml
```

---

## 关于私有模块

本仓库的 `private/` 目录是一个 **submodule**，指向作者的私有 alpha 实现（因子、策略、参数）。

- **没有权限也能用**：`git clone` 后 `private/` 是空的，框架自动回落到 `configs/example.yaml`
- **有权限时**：
  ```bash
  git clone --recurse-submodules <repo>
  pip install -e private/
  ```
  配置发现链会自动优先读 `private/configs/config.yaml`

---

## 诚实说明

这个框架**不展示盈利能力**。

作者用它做过的检验里，绝大多数想法被否决了：支撑压力线、趋势线、量价背离、放量确认、季节性、截面动量、多周期集成……都在样本外被否。期货线最终上线的只有一个 250 日时序动量单腿。

框架的价值在方法：**怎么严谨地否掉一个想法** —— 预注册口径、多重比较校正、分品种检验、前后半同号、留一法归因。

如果你在找一个赚钱的策略，这里没有。如果你在建自己的研究流程，这里的东西应该能省你一些时间。

---

## 数据源

数据通过 [akshare](https://github.com/akfamily/akshare) 等公开源获取，请遵守各数据源的使用条款。本仓库**不包含任何行情数据**。

---

## 许可

[AGPL-3.0](LICENSE)
