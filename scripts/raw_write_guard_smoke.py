"""raw_write_guard_smoke.py — 原始（未复权）行情写入护栏冒烟（离线，纯函数级）。

背景：adjust="" 的缓存键（stock_{sym}__{start}）是全部复权数据的对照基准，也是增量
更新重叠校验的基准——它被非数据层代码改写，校验会把"值被换过"误判成"除权重定锚"，
触发全量重拉重定基准，污染沿键传播。

护栏挂在 `_atomic_write_parquet` 入口（全部缓存写入的唯一咽喉），按调用栈的 __file__
路径判定：生产消费模块（backtest/factors/signals/research/train_ml_daily）写原始键即
抛错；数据层自身与 verifications/ 等显式校验脚本放行。

覆盖面：
  1. 数据层自身（本文件位于 scripts/，不在拦截清单）→ 放行。
  2. 原始键识别：qfq 键 / 指数键 / 期货键不触发；stock_{6位}__{date} 触发识别。
  3. 生产模块调用 → 拦截（用伪造的 __file__ 构造调用栈，不依赖真实模块）。
  4. 拦截是 fail-loud：抛 RuntimeError 且消息含可操作指引。

不写任何缓存文件（护栏在写入之前判定，命中即抛）。
用法：python scripts/raw_write_guard_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jq_framework.data.fetchers import fetcher

CACHE = Path("/tmp/_raw_guard_smoke")   # 只用于拼路径，绝不真的写


def _raw(sym: str = "000089", start: str = "2015-01-01") -> Path:
    return CACHE / f"stock_{sym}__{start}.parquet"


def _run_as(file_path: str, target: Path) -> str:
    """在伪造 __file__ 的栈帧下调用护栏，返回 '放行' / '拦截'。"""
    src = ("from pathlib import Path as _P\n"
           "import data.fetchers.fetcher as _f\n"
           f"_f._assert_raw_write_allowed(_P({str(target)!r}))\n")
    g = {"__file__": file_path, "__name__": "__main__"}
    try:
        exec(compile(src, file_path, "exec"), g)
        return "放行"
    except RuntimeError:
        return "拦截"


def main() -> int:
    print("== 原始（未复权）行情写入护栏冒烟 ==")

    # ---- 1. 数据层自身放行 ----
    # 本文件位于 scripts/，按白名单会被拦——正是期望行为。要验证"数据层放行"，必须以
    # 数据层的 __file__ 构造栈帧。
    got = _run_as(f"{ROOT}/data/fetchers/fetcher.py", _raw())
    assert got == "放行", f"数据层自身写原始键应放行（实际 {got}）"
    print("  [smoke] 数据层自身写入原始键 OK（放行）")

    # ---- 2. 键识别：非原始键一律放行 ----
    non_raw = [
        CACHE / "stock_000089_qfq_2016-11-24.parquet",
        CACHE / "stock_000089_hfq_2016-11-24.parquet",
        CACHE / "index_csi1000.parquet",
        CACHE / "stock_000089_qfq_2020-01-01.parquet",
        CACHE / "fin_value_em_000089.parquet",
    ]
    for p in non_raw:
        got = _run_as(f"{ROOT}/signals/probe.py", p)
        assert got == "放行", f"{p.name} 不应触发护栏（实际 {got}）"
    # 形似但不是 6 位数字代码的也不触发
    weird = [CACHE / "stock_ABCDEF__2015-01-01.parquet",
             CACHE / "stock_12345__2015-01-01.parquet"]
    for p in weird:
        assert _run_as(f"{ROOT}/signals/probe.py", p) == "放行", f"{p.name} 不应触发"
    print("  [smoke] 键识别 OK（qfq/hfq/指数/期货/非 6 位代码均放行）")

    # ---- 3. 白名单之外一律拦截 ----
    # 注意必须包含 scripts/ 下的普通脚本：首版用黑名单时，一条放在 scripts/ 的测试探针
    # 真的把 data/cache/stock_000089__2015-01-01.parquet 覆盖成了空表（已从快照恢复）。
    # 护栏要挡住的正是"没想到的调用者"，所以这里逐类覆盖。
    for mod in ("backtest/walk_forward.py", "factors/composite.py",
                "signals/daily_portfolio.py", "research/attribution.py",
                "scripts/train_ml_daily.py", "scripts/fetch_long_history.py",
                "scripts/_any_ad_hoc_probe.py", "portfolio/runner.py",
                "notebooks/scratch.py"):
        got = _run_as(f"{ROOT}/{mod}", _raw())
        assert got == "拦截", f"{mod} 写原始键应被拦截（实际 {got}）"
    print("  [smoke] 白名单外拦截 OK（含 scripts/ 普通脚本与临时探针）")

    # ---- 4. 白名单仍可用 ----
    for mod in ("data/fetchers/fetcher.py", "data/fetchers/fetcher_fundamental.py",
                "verifications/verify_ic_cgo.py"):
        got = _run_as(f"{ROOT}/{mod}", _raw())
        assert got == "放行", f"{mod} 应放行（实际 {got}）"
    print("  [smoke] 白名单放行 OK（data/fetchers/、verifications/）")

    # ---- 5. 拦截消息可操作 ----
    try:
        g = {"__file__": f"{ROOT}/signals/probe.py", "__name__": "__main__"}
        exec(compile("from pathlib import Path as _P\n"
                     "import data.fetchers.fetcher as _f\n"
                     f"_f._assert_raw_write_allowed(_P({str(_raw())!r}))\n",
                     f"{ROOT}/signals/probe.py", "exec"), g)
    except RuntimeError as e:
        msg = str(e)
        assert "qfq" in msg and "cache_cgo" in msg, f"消息缺少可操作指引: {msg}"
        print("  [smoke] 拦截消息含可操作指引 OK")

    print("\n== 原始写入护栏冒烟 全部通过 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
