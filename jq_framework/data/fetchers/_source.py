"""选库逻辑 —— 同一类数据可由不同数据源拉取（akshare/tushare/efinance），按 --source 选择。

不同数据源的覆盖深度 / 权限门槛 / 频率不同：
  - 资金流：efinance 免费浅史（最近~120日） vs tushare 深史（2018+，需 token/积分）
  - 一致预期：tushare consensus（历史发布会时点）门槛高；akshare 只给当前快照（探针实测半能）
未来新数据类（北向/两融/龙虎榜/财报）往这里加 registry 项即可，落盘契约复用 canonical schema。

用法（在各类 fetcher 内）：
  from jq_framework.data.fetchers._source import SOURCE_REGISTRY, make_sources
  # 如资金流：
  srcs = make_sources("fundflow", cfg)   # {"efinance": fetch_efinance, "tushare": fetch_tushare}
  df = srcs[args.source](code, start)    # 按 --source 选实拉函数
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional


def _tushare_pro():
    """构造 tushare pro 客户端。token/网关走环境变量（不入库，见 .gitignore TUSHARE_TOKEN*）。

    高阶账号走第三方 HTTP 网关时，tushare 需显式注入私有字段：
        pro._DataApi__token   = <token>
        pro._DataApi__http_url = <网关地址>
    否则默认直连 pro.tushare.pro 官方地址。
    """
    import tushare as ts
    tok = os.environ.get("TUSHARE_TOKEN") or ""
    if not tok:
        raise SystemExit(
            "--source tushare / 一致预期需 TUSHARE_TOKEN 环境变量(或 --token)。见 docs/stock/tushare接入流程.md")
    pro = ts.pro_api(tok)
    url = os.environ.get("TUSHARE_HTTP_URL")
    if url:
        # 高阶账号走中继网关：覆盖默认官方地址
        pro._DataApi__token = tok
        pro._DataApi__http_url = url
    return pro


# 数据类名 -> 可用数据源列表（顺序=推荐优先级）。（akshare 主行情是 fetchers.fetcher 直接管，不进此 registry）
_DATA_SOURCES: dict[str, list[str]] = {
    "fundflow":   ["tushare", "efinance"],   # tushare 深史优先，efinance 浅史兜底
    "consensus":  ["tushare"],               # 历史发布会时点一致预期仅 tushare
    "margin":     ["akshare"],               # 两融逐日（akshare 深史实测可用，见探针）
    "lhb":        ["akshare"],               # 龙虎榜（akshare 实测 2018 通过但截面稀疏）
}


@dataclass
class Source:
    """一个数据源：名字 + 实拉函数签名 + 落盘契约（schema/目录/文件名模式）。"""
    name: str
    fetch: Callable                       # fetch(identifier, **kw) -> canonical df
    schema: tuple                        # canonical 列序（对齐 fundflow_schema.CANONICAL_COLS /
                                         #  各数据自身 schema）
    dir: str                             # 相对项目根的落盘目录
    pattern: str                         # 文件名模式，{symbol} 占位，如 ff_{symbol}.parquet
    merge: Optional[Callable] = None     # 可选：增量合并函数(existing, new) -> merged
    _extra: dict = field(default_factory=dict)


def _registry(data_class: str) -> list[str]:
    srcs = _DATA_SOURCES.get(data_class)
    if srcs is None:
        raise KeyError(f"未知数据类 {data_class!r}，可选: {list(_DATA_SOURCES)}")
    return srcs


def sources_of(data_class: str) -> list[str]:
    """该数据类的可用源名列表（推荐顺序）。"""
    return _registry(data_class)


class SourceRegistry:
    """按数据类注册/取源。各类 fetcher 模块启动时注册自己的实拉函数。"""
    def __init__(self):
        self._impls: dict[str, dict[str, Source]] = {}

    def register(self, data_class: str, src: Source) -> None:
        self._impls.setdefault(data_class, {})[src.name] = src

    def get(self, data_class: str, source: str) -> Source:
        impls = self._impls.get(data_class, {})
        if source not in impls:
            raise KeyError(f"{data_class} 未注册源 {source!r}，可用: {list(impls)}")
        return impls[source]

    def select(self, data_class: str, source: str | None) -> Source:
        """取源；source 未给时用该数据类推荐优先级第一个（已注册的）。"""
        if source is None:
            for name in _registry(data_class):
                if name in self._impls.get(data_class, {}):
                    return self._impls[data_class][name]
            raise KeyError(f"{data_class} 无任何已注册源")
        return self.get(data_class, source)


REGISTRY = SourceRegistry()
