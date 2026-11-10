"""统一配置加载与发现链。

为什么需要
----------
开源后有两类使用者：
  · 外部用户：clone 公开仓库，没有私有 submodule → 用 configs/example.yaml
  · 仓库作者：有私有 submodule → 用 private/configs/*.yaml（含真实参数）

两者代码相同、路径不同。把所有入口的 `--config` 默认值收拢到本模块，
就能做到"代码零改动、行为随环境自动切换"。

发现链（按顺序，取第一个存在的）
--------------------------------
  1. 显式传入的 path（CLI --config）
  2. $JQ_CONFIG 环境变量
  3. private/configs/{market}.yaml   ← 私有 submodule
  4. configs/{market}.yaml           ← 公开示例

market 取 "futures" / "stock"，由调用方指定；不指定时按 3→4 逐级找
futures.yaml → stock.yaml → config.yaml。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# 仓库根 = 本文件的上三级（jq_framework/config/loader.py → 仓库根）
# 注意：拆分为 jq_framework/ 包后此处自动跟随，无需改硬编码层级。
REPO_ROOT = Path(__file__).resolve().parents[2]

_PRIVATE_DIR = "private/configs"
_PUBLIC_DIR = "configs"
_LEGACY_NAME = "config.yaml"
_EXAMPLE_NAME = "example.yaml"


def _candidates(market: str | None = None) -> list[Path]:
    names = [f"{market}.yaml"] if market else ["futures.yaml", "stock.yaml", _LEGACY_NAME]
    out: list[Path] = []
    for base in (_PRIVATE_DIR, _PUBLIC_DIR):
        out.extend(REPO_ROOT / base / n for n in names)
    if not market:
        out.append(REPO_ROOT / _LEGACY_NAME)               # 旧位置兜底
        out.append(REPO_ROOT / _PUBLIC_DIR / _EXAMPLE_NAME)  # 外部用户最后兜底
    else:
        # 指定 market 时也给示例兜底，保证外部用户开箱能跑
        out.append(REPO_ROOT / _PUBLIC_DIR / _EXAMPLE_NAME)
    return out


def find_config(path: str | Path | None = None, market: str | None = None) -> Path:
    """按发现链定位配置文件；找不到则 fail-loud 并列出所有尝试过的路径。"""
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        return p

    env = os.environ.get("JQ_CONFIG")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            return p

    tried = _candidates(market)
    for p in tried:
        if p.exists():
            return p

    listing = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(
        f"找不到配置文件（market={market!r}）。已尝试:\n{listing}\n"
        f"提示: 外部用户请从 configs/example.yaml 复制一份并改名；"
        f"作者请确认私有 submodule 已初始化（git submodule update --init）。")


def load_config(path: str | Path | None = None, market: str | None = None) -> dict:
    """加载配置字典。path 为 None 时走发现链。"""
    p = find_config(path, market)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
