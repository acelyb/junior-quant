"""
数据层：akshare 拉取 A 股 / 期货日线数据 + parquet 本地缓存。

设计原则：
- 对外只暴露 get_stock_daily / get_futures_daily 两个函数，返回统一的 DataFrame
- 统一列名：date(datetime) / open / high / low / close / volume
- 本地 parquet 缓存，超过有效期则重新拉取（避免每次跑都打 akshare）
- akshare 是外部网络依赖，可能失败/变慢，所有调用都做了容错
- 全局 socket 超时（2026-10-15）：akshare 内部 requests 不传 timeout，对端静默掉线时
  read() 永久阻塞且不抛异常——所有 except 兜底都会因"异常永不发生"而成为死代码。
  进程级默认超时保证失败必抛、兜底必达。
"""
from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 数据快照层（P1-4）：改写前屏障 + 严格 cache-only。包上下文缺失（本文件被直接执行）时
# 退化为同目录导入，仅 __main__ 自测路径受影响。
try:
    from jq_framework.data.fetchers import snapshot as _snapshot
except ImportError:  # pragma: no cover - 直接执行本文件时
    from snapshot import _snapshot  # type: ignore

# 死连接兜底（2026-10-15 挂死事故：成分股 TTL 刷新打 csindex，对端静默掉线，
# 无超时的读阻塞把串行回测卡死 1h+，新浪→东财→陈旧缓存三层兜底全部未触发）。
# socket.timeout ⊂ OSError ⊂ Exception：设默认超时后，任何取数调用（含 akshare 内部
# 发起的）在 60s 内必抛异常 → 既有 except 链恢复生效。仅影响失败路径，成功路径行为
# 与数值逐字不变。walk_forward 必 import 本模块，故所有回测进程自动获得该保护。
socket.setdefaulttimeout(60)

# akshare import 放在函数内部，避免没装依赖时 import 整个包就崩
# 这样也能让 fetcher 在缺包时给出清晰提示


class DataFetchError(Exception):
    """数据拉取失败"""


def _load_config() -> dict:
    """加载 config.yaml 中的 data 段。

    环境变量覆盖：若 QUANT_CACHE_DIR 非空，覆盖 data.cache_dir（不影响其他字段）。
    用于把长历史（2010+）数据集隔离到独立目录（如 data/cache_long），不污染 data/cache
    （回归门保护对象），以及 compare_ml --snapshot 钉住快照目录。无环境变量=行为逐字不变。
    QUANT_STRICT_CACHE=1 → strict_cache=True（严格 cache-only：命中即读、未命中 fail-loud
    报错，绝不拉网绝不写；钉住快照运行用，见 data/fetchers/snapshot.py）。
    """
    from jq_framework.config.loader import REPO_ROOT, load_config
    cfg = load_config().get("data", {})
    env_cache = os.environ.get("QUANT_CACHE_DIR")
    if env_cache:
        cfg["cache_dir"] = env_cache
    if os.environ.get("QUANT_STRICT_CACHE") == "1":
        cfg["strict_cache"] = True
    return cfg


def _cache_path(cache_dir: str, key: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / cache_dir / f"{key}.parquet"


def _need_refresh(path: Path, max_cache_days: int) -> bool:
    """缓存是否过期（age > max_cache_days）。用于快路径判断是否走增量。"""
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(days=max_cache_days)


def _need_full_refresh(path: Path, full_refresh_days: int) -> bool:
    """是否需要强制全量重拉（兜底复权历史修正）。基于文件内容最后日期到今天的天数。"""
    if not path.exists():
        return True
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return True
        last = pd.Timestamp(df["date"].iloc[-1])
        return (datetime.now() - last.to_pydatetime()) > timedelta(days=full_refresh_days)
    except Exception:
        return True


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    """原子写 parquet：写临时文件 → os.replace 原子替换（POSIX/Windows 同卷均原子）。

    防场景级进程并行 read_only 回退时，多 worker 同时缺同一文件并发写产生半截损坏 parquet
    （硬约束：线程安全）。写完后另一并发 reader 若拿到旧/新完整版本皆可读，绝不读到半截。

    P1-4 改写前屏障（2026-10-21）：本函数是全部缓存写入的唯一咽喉，入口处先执行
    snapshot.freeze_barrier——每天第一次内容改写前，把当前缓存状态冻结为不可变快照
    （硬链接写时复制；当日幂等；QUANT_SNAPSHOT_BARRIER=0 可紧急关闭）。

    原始键写入护栏（2026-10-29）：本函数同时是"原始数据只读"约定的强制点——生产消费
    模块（回测/因子/信号/训练）写 adjust="" 键会在此直接抛错，见 _assert_raw_write_allowed。
    """
    _assert_raw_write_allowed(path)
    _snapshot.freeze_barrier(path.parent)
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            df.to_parquet(f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_REANCHOR_FILE = ".reanchor_times.json"

# 原始（未复权）行情写入护栏（2026-10-29）：adjust="" 的缓存键（stock_{sym}__{start}）
# 是全部复权数据的对照基准，也是"原始数据只读"这一长期约定的落点。它一旦被非数据层
# 代码改写，增量更新的重叠校验会把"值被换过"误判成"除权重定锚"，进而触发全量重拉并
# 重定基准——污染会沿键传播。
#
# 判定采用**白名单**（默认拒绝）：只有数据层自身（data/fetchers/）与显式声明需要原始价
# 的校验脚本（verifications/）可以写原始键，其余一律拒绝。
#
# 为什么是白名单而不是"拦截已知生产模块"（2026-10-29 首版教训）：黑名单只列了 backtest/
# factors/signals/research/train_ml_daily，而 scripts/ 下几十个脚本不在其中——一条放在
# scripts/ 下的测试探针真的把 data/cache/stock_000089__2015-01-01.parquet 覆盖成了空表
# （2838 行 → 0 行，已从快照恢复）。护栏的价值在于"没想到的调用者"也能被挡住，黑名单
# 天然做不到这一点。
#
# 判定用 __file__ 路径而非模块名：脚本直接运行时 __name__ 是 "__main__"，模块名匹配
# 不到（train_ml_daily 就是这么跑的）。路径判定对 import 与直接执行两种方式都成立。
# 无 __file__ 的栈帧（交互式 exec/REPL）按允许处理——那是人工操作，不是自动化链路。
_RAW_WRITE_ALLOW_PATHS = ("/data/fetchers/", "/verifications/")


def _assert_raw_write_allowed(path) -> None:
    """原始键写入护栏：仅数据层与显式校验脚本可写 adjust="" 缓存（白名单）。"""
    path = Path(path)
    name = path.name
    # 原始键文件名形如 stock_000089__2015-01-01.parquet（adjust 为空 → 双下划线）
    if not (name.startswith("stock_") and "__" in name):
        return
    sym = name[len("stock_"):].split("__", 1)[0]
    if not (len(sym) == 6 and sym.isdigit()):
        return
    import inspect as _inspect
    frame = _inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            raw_file = frame.f_globals.get("__file__")
            if raw_file:
                f = str(raw_file).replace("\\", "/")
                if any(a in f for a in _RAW_WRITE_ALLOW_PATHS):
                    return          # 白名单命中，放行
                raise RuntimeError(
                    f"禁止从 {f} 写入原始（未复权）行情缓存 {name}。"
                    f"原始数据是复权数据的对照基准，且增量更新依赖其内容不变——"
                    f"被改写会让重叠校验误判为除权重定锚，污染沿键传播。"
                    f"训练/回测/信号一律用 adjust='qfq'；确需原始价请用独立的缓存目录"
                    f"（如 cache_cgo）或带 adjust='' 的独立直取函数。"
                    f"（白名单：data/fetchers/、verifications/——确需在此处写原始价请"
                    f"使用独立缓存目录，不要共用 data/cache 的原始键）")
            frame = frame.f_back
        return                      # 栈上无 __file__（交互式），放行
    finally:
        del frame


def _load_anchor_times(cache_dir: str) -> dict:
    """读全量重锚时间标记（{缓存键: epoch 秒}）。缺失/损坏按空处理。"""
    root = Path(__file__).resolve().parents[3]
    try:
        return json.loads((root / cache_dir / _REANCHOR_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _record_anchor(cache_dir: str, key: str) -> None:
    """记录一次全量重锚完成时刻（tmp+rename 原子写）。

    并发读改写丢更新最坏后果=个别键下周期多一次全量重拉，无正确性影响，不设锁。

    best-effort（2026-10-29 修）：本函数只写重锚时间标记，标记缺失/写失败的后果
    仅是下周期多一次全量重拉（幂等自愈，见 _anchor_fresh），绝不影响已经落盘的行情
    数据本身。原先临时文件名固定为 .reanchor_times.tmp，并发线程 A 写入后 B 抢先
    rename 走，A 再 os.replace 即 ENOENT → 异常冒泡到 get_batch_daily 被记成"该股
    拉取失败"从训练池剔除（2026-10-27 丢失 25 只 / 10-29 丢失 23 只，而数据其实
    已正确落盘）。改用 mkstemp 唯一临时名 + 吞掉 OSError：标记尽力而为，数据正确性
    不受影响。
    """
    root = Path(__file__).resolve().parents[3]
    p = root / cache_dir / _REANCHOR_FILE
    times = _load_anchor_times(cache_dir)
    times[key] = time.time()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(times))
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        # 并发竞争（临时文件被另一线程 rename 走 / 目录瞬时不可写）：
        # 仅丢失本次标记 → 下周期多一次全量重锚，幂等自愈
        pass


def _anchor_fresh(cache_dir: str, key: str, cfg: dict) -> bool:
    """上次全量重锚是否足够新（qfq 接缝加固，2026-10-23）。

    新浪源在除权后的数天窗口内会返回"历史段旧锚 + 新行新锚"的内部不一致序列，
    增量路径的 10 日重叠校验只比对重叠区（旧锚两侧一致），结构性看不见增量段
    内部的锚切换 → 拼接后接缝日留下 ≈ 一次分红幅度的人为收益缺口（2026-10-23
    实证：跨簇收益对比扫描定位 1,274 份带缝文件，2018 生产簇 46 份；工具
    scripts/diag_qfq_seams.py，治愈 scripts/heal_qfq_seams.py）。对策：距上次
    全量重拉超过 data.reanchor_days（默认 7 天）则放弃增量、强制全量重锚——
    任何漏网接缝的存活期上界 = 一个重锚周期。标记仅在全量路径完成时记录
    （增量合并不记——增量恰是风险操作）。标记缺失=按陈旧处理：存量文件首次
    访问触发一次全量重锚，幂等自愈。reanchor_days<=0 关闭（回到加固前行为）。
    """
    days = float(cfg.get("reanchor_days", 7))
    if days <= 0:
        return True
    t = _load_anchor_times(cache_dir).get(key)
    return t is not None and (time.time() - float(t)) < days * 86400.0


def _normalize(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """统一列名为 backtrader 友好格式。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    # 只保留标准列，按日期升序
    cols = ["date", "open", "high", "low", "close", "volume"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("date").reset_index(drop=True)
    # 丢弃 volume==0 噪声行（akshare 源偶发；真停牌是缺整行不是 vol=0）。
    # 只丢精确 0；NaN volume（若有）保留。真实成交必有 volume>0。
    if "volume" in df.columns:
        df = df[df["volume"] != 0].reset_index(drop=True)
    return df


def _normalize_futures(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """期货日线标准化：与 _normalize 同构，但列集扩为含 hold(持仓量)/settle(结算价)。

    期货主力/连续序列需持仓量（选主力、风控）与结算价（TSMOM 用结算价比收盘更贴近
    保证金盯市口径），故不能沿用只留 OHLCV 的 _normalize。列名规范：
    [date, open, high, low, close, volume, hold, settle]，按日期升序、丢 volume==0。
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    cols = ["date", "open", "high", "low", "close", "volume", "hold", "settle"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("date").reset_index(drop=True)
    if "volume" in df.columns:
        df = df[df["volume"] != 0].reset_index(drop=True)
    return df


def _fetch_futures_daily_raw(symbol: str, start: str | None = None,
                             end: str | None = None) -> pd.DataFrame:
    """向 akshare 拉取期货日线（新浪源 futures_zh_daily_sina），返回已 _normalize_futures 的 df。

    symbol: 如 "RB0" / "AU0"（0 结尾表主力连续）。新浪源固定返回全部历史，
    无法按 start/end 增量（无该参数），故拉全量后在 _normalize_futures 后按 [start,end] 截断。
    """
    try:
        import akshare as ak
    except ImportError as e:
        raise DataFetchError("未安装 akshare，请 pip install akshare") from e

    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or df.empty:
            raise DataFetchError(f"akshare 返回空数据: {symbol}")
        df = _normalize_futures(df, "date")
        # 新浪源返回较长历史，默认只保留近 5 年，避免数据过长（与旧实现一致）
        if start is None:
            cutoff = datetime.now() - timedelta(days=5 * 365)
            df = df[df["date"] >= pd.Timestamp(cutoff)].reset_index(drop=True)
        else:
            if start:
                df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
            if end:
                df = df[df["date"] <= pd.Timestamp(end)].reset_index(drop=True)
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"拉取期货 {symbol} 失败: {e}") from e
    return df


def _fetch_stock_daily_raw(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """实际向 akshare 拉取一段 A 股日线（新浪源优先，东财兜底），返回已 _normalize 的 df。"""
    try:
        import akshare as ak
    except ImportError as e:
        raise DataFetchError("未安装 akshare，请 pip install akshare") from e

    # 新浪源需要带市场前缀：6开头=sh，其余(sz) = sz
    prefix = "sh" if symbol.startswith("6") else "sz"

    # 优先用新浪源（列名已是英文标准格式，且比东财源稳定）
    try:
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}",
                                 start_date=start.replace("-", ""),
                                 end_date=end.replace("-", ""), adjust=adjust)
        if df is None or df.empty:
            raise DataFetchError(f"新浪源返回空: {symbol}")
        df = _normalize(df, "date")
    except DataFetchError:
        raise
    except Exception as sina_err:
        # 新浪失败则回退东财源（列名为中文，需映射）
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start.replace("-", ""),
                                    end_date=end.replace("-", ""), adjust=adjust)
            if df is None or df.empty:
                raise DataFetchError(f"东财源返回空: {symbol}")
            col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                       "最高": "high", "最低": "low", "成交量": "volume"}
            df = df.rename(columns=col_map)
            df = _normalize(df, "date")
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(
                f"拉取 A 股 {symbol} 失败（新浪:{sina_err} / 东财:{e}）") from e
    return df


def get_stock_daily(symbol: str, start: str = "2020-01-01", end: str | None = None,
                    adjust: str = "qfq", read_only: bool = False,
                    skip_missing: bool = False) -> pd.DataFrame:
    """
    拉取 A 股日线。带本地 parquet 缓存 + 增量更新。

    缓存策略：
    - 有效期内（max_cache_days）：直接读缓存，不碰网络（最快路径）
    - 过期但距上次全量未超过 full_refresh_days：增量补新数据，带重叠日校验——
      两段 close 在重叠日不一致（除权→qfq 重定锚）则该股全量重拉；空窗
      （周末/节假日/停牌）touch 返回缓存不重拉（审计#2/#13，2026-09-24 修）
    - 超过 full_refresh_days / 无缓存 / 检出复权基准漂移：全量重拉（刷新 qfq 基准）

    read_only: True = cache-only 模式（场景级进程并行用）：文件存在则直接读、
        **绝不拉网、绝不写 parquet、跳过 migrate**。文件缺失才回退拉取（此时该 worker
        是唯一写者，无并发写竞态）。默认 False=旧行为。设计见 plans/curried-prancing-sketch 点3。

    symbol: 6位股票代码，如 "000001"
    start/end: 日期字符串 YYYY-MM-DD
    adjust: 复权类型 qfq/hfq/""(不复权)
    返回: DataFrame[date, open, high, low, close, volume]
    """
    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    max_days = cfg.get("max_cache_days", 1)
    full_refresh_days = cfg.get("full_refresh_days", 30)

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # 缓存键不含 end（增量模式下 end 始终是今天），避免跨次 miss 重复拉
    key = f"stock_{symbol}_{adjust}_{start}"
    path = _cache_path(cache_dir, key)

    # 严格 cache-only（P1-4 钉住快照运行）：命中即读（跳过 TTL——冻结数据的"过期"是常态，
    # 钉住语义=接受冻结状态），未命中 fail-loud。绝不拉网、绝不写、绝不增量。
    if cfg.get("strict_cache"):
        if not path.exists():
            raise _snapshot.StrictCacheMiss(
                f"严格缓存模式：快照缺 {path.name}（拒绝拉网/落盘；"
                f"检查快照完整性，或去掉 --snapshot / QUANT_STRICT_CACHE 改常规模式）")
        return pd.read_parquet(path)

    # read_only：cache-only。命中则读，跳过一切过期/增量/写逻辑；未命中回退全量拉取
    # （回退时该 worker 是唯一写者，主进程预热已让多数标的存在，仅预热失败的个别标的会拉）
    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        # skip_missing=True（只读诊断用）：精确 key 未命中且不允许拉网时，回退到同 symbol 任意
        # 已有缓存里日期最早、结尾最晚的一档，按 [start,end] 裁剪返回；完全不碰网络。
        # 用于缓存键 start 敏感导致的不必要 miss（如诊断脚本请求的 start 与文件名不符）。
        # 找不到可覆盖档 → 返回空 df（该股缺失，跳过），绝不在只读诊断下拉网。
        if skip_missing:
            import glob
            pat = f"{cache_dir}/stock_{symbol}_{adjust}_*.parquet"
            cands = sorted(glob.glob(pat))
            best = None
            for p in cands:
                try:
                    df = pd.read_parquet(p)
                except Exception:
                    continue
                if df.empty:
                    continue
                lo = pd.Timestamp(df["date"].min())
                if best is None or (lo < best[0]):
                    best = (lo, df)
            if best is not None:
                df = best[1]
                return df[(df["date"] >= pd.Timestamp(start)) &
                          (df["date"] <= pd.Timestamp(end or "2199-12-31"))]
            return pd.DataFrame()
        # 文件缺失 → 回退拉取。多 worker 可能同时缺同一文件（预热失败的个别标的），
        # 写用临时文件 + 原子 rename 兜底，防并发写产生半截损坏 parquet（硬约束：线程安全）
        df = _fetch_stock_daily_raw(symbol, start, end, adjust)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(path, df)
        _record_anchor(cache_dir, key)  # 整文件全量写入=一次完整重锚
        return df

    # 1) 快路径：缓存未过期，直接读
    if not _need_refresh(path, max_days):
        return pd.read_parquet(path)

    # 2) 增量路径：有缓存且距上次全量未超期 → 只补新数据（带重叠校验，审计#2/#13）。
    #    锚定新鲜门（2026-10-23 接缝加固）：距上次全量重锚超 reanchor_days 也放弃
    #    增量走全量——除权后源端历史再生成期内，增量段内部锚切换是重叠校验的盲区，
    #    周期性全量重锚把漏网接缝的存活期压到 ≤ 一个重锚周期（见 _anchor_fresh）。
    if (path.exists() and not _need_full_refresh(path, full_refresh_days)
            and _anchor_fresh(cache_dir, key, cfg)):
        try:
            cached = pd.read_parquet(path)
            if not cached.empty:
                last_date = pd.Timestamp(cached["date"].iloc[-1]).strftime("%Y-%m-%d")
                # 已是最新（今天/最近交易日），无需拉取
                if last_date >= end:
                    return cached
                # 只拉新数据，但起点往前多压 10 个日历日（≥5 个交易日重叠）做校验：
                # qfq 前复权随除权事件整体重定基准——akshare 对全列除以当日 qfq 因子后才
                # 按日期切片，两次拉取间若有除权，增量段（"今天锚定"）与缓存段（"上次
                # 锚定"）在重叠日的价格不一致；直接拼接会在接缝处留下 ≈ 股息率的永久
                # 收益缺口（审计#2）。重叠日 close 对不上 → 该股整体全量重拉重定基准。
                inc_start = (pd.Timestamp(last_date) - timedelta(days=10)).strftime("%Y-%m-%d")
                try:
                    inc = _fetch_stock_daily_raw(symbol, inc_start, end, adjust)
                except DataFetchError as e:
                    # 空增量窗（周末/节假日/停牌/退市）：双源明确"返回空" → 非错误，
                    # touch 标记后返回缓存（审计#13：原实现该 raise 被外层 except 吞掉
                    # → 意外落到全量路径重拉全史，数小时+封IP风险）。网络类错误
                    # （非"返回空"）继续外抛 → 走外层原全量重试路径，不被误当空窗。
                    if "返回空" not in str(e):
                        raise
                    inc = None
                rebase = False
                if inc is not None and not inc.empty:
                    ov_dates = sorted(set(cached["date"]) & set(inc["date"]))
                    if not ov_dates:
                        # 增量段与缓存零重叠：锚定身份无从校验（源覆盖/端点行为变化），
                        # 宁全量重锚不可盲拼（2026-10-23 接缝加固）
                        rebase = True
                    else:
                        c_ref = cached.set_index("date")["close"]
                        i_ref = inc.set_index("date")["close"]
                        rel = ((i_ref[ov_dates] - c_ref[ov_dates]).abs()
                               / c_ref[ov_dates].abs())
                        if bool((rel > 1e-6).any()):
                            rebase = True  # 除权/复权基准漂移 → 不拼接，落全量路径
                if not rebase:
                    if inc is not None and not inc.empty:
                        merged = pd.concat([cached, inc], ignore_index=True)
                        merged = merged.drop_duplicates(subset=["date"], keep="last")
                        merged = merged.sort_values("date").reset_index(drop=True)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        _atomic_write_parquet(path, merged)  # 审计#14：非原子写 → 原子写
                        return merged
                    # 增量拉取为空（如周末/节假日/停牌），刷新 mtime 标记本次检查过
                    path.touch()
                    return cached
                # rebase=True → 落到下方全量路径整体重定基准
        except Exception:
            # 增量失败则回退全量，保证不因增量逻辑中断
            pass

    # 3) 全量路径：无缓存 / 已超 full_refresh_days / 增量失败兜底 / 检出复权基准漂移 /
    #    重锚超期（2026-10-23 接缝加固）。全量=单源单锚整段一致，天然无接缝。
    df = _fetch_stock_daily_raw(symbol, start, end, adjust)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(path, df)  # 审计#14：全仓日线路径统一原子写，防半截 parquet
    _record_anchor(cache_dir, key)   # 全量重锚完成时刻（供 _anchor_fresh 周期性强制）
    return df


def get_futures_daily(symbol: str, start: str | None = None, end: str | None = None,
                      read_only: bool = False) -> pd.DataFrame:
    """
    拉取期货主力连续日线（新浪源，代码以 0 结尾表主力）。

    symbol: 如 "RB0"(螺纹钢主力) / "AU0"(黄金主力)

    与 get_stock_daily 同范式的本地缓存 + 增量更新：
    - 有效期内直读缓存（最快路径）
    - 过期但距上次全量未超 full_refresh_days：只增量补 last_date+1→end 的新数据
    - 超过 full_refresh_days 或无缓存：全量重拉
    read_only: True = cache-only（命中即读、不拉网不写）；未命中才回退拉取。

    返回: DataFrame[date, open, high, low, close, volume, hold, settle]
          （列集比旧版多 hold/settle；消费方若只需 OHLCV，取这几列即可，向后兼容）
    """
    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    max_days = cfg.get("max_cache_days", 1)
    full_refresh_days = cfg.get("full_refresh_days", 30)

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # 缓存键不含 end（增量模式下 end 始终是今天），schema 增列无需迁移，键仍 futures_{symbol}
    key = f"futures_{symbol}"
    path = _cache_path(cache_dir, key)

    # 严格 cache-only（P1-4）：命中即读、未命中 fail-loud（原 read_only 未命中回退拉取
    # 的语义只属 worker 预热兜底，严格模式不承担该语义——快照必须不可写）。
    if cfg.get("strict_cache"):
        if not path.exists():
            raise _snapshot.StrictCacheMiss(f"严格缓存模式：快照缺 {path.name}")
        return pd.read_parquet(path)

    # read_only：cache-only
    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        df = _fetch_futures_daily_raw(symbol, start, end)
        _atomic_write_parquet(path, df)
        return df

    # 1) 快路径：缓存未过期直读
    if not _need_refresh(path, max_days):
        return pd.read_parquet(path)

    # 2) 增量路径：有缓存且未超 full_refresh_days → 只补 last_date+1→end
    if path.exists() and not _need_full_refresh(path, full_refresh_days):
        try:
            cached = pd.read_parquet(path)
            if not cached.empty:
                last_date = pd.Timestamp(cached["date"].iloc[-1]).strftime("%Y-%m-%d")
                if last_date >= end:
                    return cached
                # 新浪源无 start/end 增量参数，只能拉全量再截断（受 full_refresh_days 约束）
                inc = _fetch_futures_daily_raw(symbol, last_date, end)
                if inc is not None and not inc.empty:
                    merged = pd.concat([cached, inc], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["date"], keep="last")
                    merged = merged.sort_values("date").reset_index(drop=True)
                    _atomic_write_parquet(path, merged)
                    return merged
                path.touch()  # 无新数据，刷新 mtime 标记已检查
                return cached
        except Exception:
            pass  # 增量失败回退全量

    # 3) 全量路径：无缓存 / 超 full_refresh_days / 增量失败兜底
    df = _fetch_futures_daily_raw(symbol, start, end)
    _atomic_write_parquet(path, df)
    return df


def load_symbol(symbol: str, market: str = "stock", **kwargs) -> pd.DataFrame:
    """
    统一入口：按市场分发。
    market: "stock" / "futures"
    """
    if market == "stock":
        return get_stock_daily(symbol, **kwargs)
    elif market == "futures":
        return get_futures_daily(symbol, **kwargs)
    raise ValueError(f"未知市场: {market}")


# ============ 多因子选股：指数成分股 + 批量下载 ============

# 常用指数代码
INDEX_SYMBOLS = {
    "csi300": "000300",   # 沪深300（基准）
    "csi500": "000905",   # 中证500
    "csi1000": "000852",  # 中证1000
}
# csi_all = csi300∪csi500∪csi1000 逻辑并集（1800 只，三者零重叠 = 全市场频谱），
# 无单一 akshare 指数代码，在 get_index_constituents 内特判聚合，不进 INDEX_SYMBOLS。
# get_index_real_daily 不支持 csi_all（无真实指数行情），调用方须传真实指数名。
COMPOSITE_UNIVERSES = {"csi_all": ["csi300", "csi500", "csi1000"]}


def list_cached_symbols(start: str = "2016-11-24", adjust: str = "qfq",
                        cache_dir: str | None = None) -> list[str]:
    """枚举**已有日线缓存**的全部标的（只读磁盘目录，不拉网、不写盘）。

    用途（2026-10-30，M6 逐日动态训练池）：训练池不再是某个指数的静态成分股，而是"当日
    符合市值/流动性条件的全市场标的"。全市场标的清单因此不能来自指数接口，只能来自
    "缓存里到底有哪些股票"——这也正是严格 cache-only 模式下唯一可靠的口径（指数接口
    返回的是当前成分，与历史快照口径不同）。

    start/adjust 决定读哪个缓存键（必须与训练实际使用的键一致，见 serving_train_start）。
    返回按代码排序（§10 确定性），无缓存返回空列表。
    """
    cfg = _load_config()
    root = Path(__file__).resolve().parents[3]
    cache_root = root / (cache_dir or cfg.get("cache_dir", "data/cache"))
    if not cache_root.exists():
        return []
    prefix = f"stock_"
    suffix = f"_{adjust}_{start}.parquet"
    out = []
    for p in cache_root.glob(f"{prefix}*{suffix}"):
        name = p.name
        if not name.endswith(suffix):
            continue
        sym = name[len(prefix):-len(suffix)]
        if len(sym) == 6 and sym.isdigit():
            out.append(sym)
    return sorted(set(out))


def get_index_constituents(index_name: str, read_only: bool = False) -> list[str]:
    """
    获取指数成分股代码列表。

    index_name: "csi300" / "csi500" / "csi1000" / "csi_all"（= 三者并集 1800 只）
    返回: 6位股票代码列表

    read_only: True = cache-only（命中即读、不拉网不写）；未命中回退拉取。见 plans 点3。
    """
    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    max_days = cfg.get("max_cache_days", 1)

    # 复合并集 universe（csi_all）：聚合成分股并去重，缓存键 const_csi_all。
    # 无 akshare 单接口，递归拉三指数再并集。
    if index_name in COMPOSITE_UNIVERSES:
        path = _cache_path(cache_dir, f"const_{index_name}")
        # 严格 cache-only（P1-4）：命中即读、未命中 fail-loud，不递归拉子指数
        if cfg.get("strict_cache"):
            if not path.exists():
                raise _snapshot.StrictCacheMiss(f"严格缓存模式：快照缺 {path.name}")
            return pd.read_parquet(path)["symbol"].tolist()
        if read_only:
            if path.exists():
                return pd.read_parquet(path)["symbol"].tolist()
        elif not _need_refresh(path, max_days):
            return pd.read_parquet(path)["symbol"].tolist()
        seen: list[str] = []
        seen_set: set[str] = set()
        for sub in COMPOSITE_UNIVERSES[index_name]:
            for s in get_index_constituents(sub, read_only=read_only):
                if s not in seen_set:
                    seen_set.add(s)
                    seen.append(s)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(path, pd.DataFrame({"symbol": seen}))
        return seen

    code = INDEX_SYMBOLS.get(index_name)
    if code is None:
        raise ValueError(f"未知指数: {index_name}，可选: {list(INDEX_SYMBOLS)}")

    path = _cache_path(cache_dir, f"const_{index_name}")

    # 严格 cache-only（P1-4）：命中即读、未命中 fail-loud，绝不走拉取/陈旧缓存兜底
    if cfg.get("strict_cache"):
        if not path.exists():
            raise _snapshot.StrictCacheMiss(f"严格缓存模式：快照缺 {path.name}")
        return pd.read_parquet(path)["symbol"].tolist()

    if read_only:
        if path.exists():
            return pd.read_parquet(path)["symbol"].tolist()
        # 未命中回退拉取（预热应已写，仅异常态走到）
    elif not _need_refresh(path, max_days):
        return pd.read_parquet(path)["symbol"].tolist()

    try:
        import akshare as ak
        df = ak.index_stock_cons_csindex(symbol=code)
        if df is None or df.empty:
            raise DataFetchError(f"成分股返回空: {index_name}")
        symbols = df["成分券代码"].astype(str).str.zfill(6).tolist()
    except Exception as e:
        # 拉取失败（含超时/空返回）→ 陈旧缓存兜底：成分股日变更极小，
        # 宁可用旧一天的名单也不能让回测挂死/中断（2026-10-15 事故的根治）。
        # 无任何本地缓存才向上抛（预热过的环境不该走到这）。
        if path.exists():
            print(f"[WARN] {index_name} 成分股拉取失败（{type(e).__name__}: {e}），"
                  f"退回陈旧本地缓存", flush=True)
            return pd.read_parquet(path)["symbol"].tolist()
        raise DataFetchError(f"获取 {index_name} 成分股失败且无本地缓存: {e}") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(path, pd.DataFrame({"symbol": symbols}))
    return symbols


def _migrate_legacy_cache(cache_dir: str, symbol: str, adjust: str, start: str) -> None:
    """旧缓存键带 end 日期（stock_{sym}_{adj}_{start}_{end}），新键不带 end。
    若新键缺失但存在旧键文件，把最近一个旧文件重命名过来，避免重复全量拉取。"""
    root = Path(__file__).resolve().parents[3]
    cache_root = root / cache_dir
    if not cache_root.exists():
        return
    new_key = f"stock_{symbol}_{adjust}_{start}"
    new_path = cache_root / f"{new_key}.parquet"
    if new_path.exists():
        return
    # 找该标的所有旧缓存（不同 end），取 mtime 最新的
    prefix = f"stock_{symbol}_{adjust}_{start}_"
    legacy = sorted(cache_root.glob(f"{prefix}*.parquet"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if legacy:
        try:
            legacy[0].rename(new_path)
        except Exception:
            pass


def get_batch_daily(symbols: list[str], start: str = "2020-01-01",
                    adjust: str = "qfq", progress: bool = True,
                    read_only: bool = False, skip_missing: bool = False) -> dict[str, pd.DataFrame]:
    """
    批量拉取多只股票日线，返回 {symbol: DataFrame}。

    并发拉取（线程池，受 fetch_workers 限制，防 akshare/新浪源限流）。
    对失败的单只做容错（返回空 df），不因个别失败中断整体。
    依赖 get_stock_daily 的本地缓存 + 增量更新，二次运行会很快。

    read_only: True = cache-only（透传给 get_stock_daily，并跳过 _migrate_legacy_cache 的
        os.rename 以消除并发写竞态）。主进程预热后 worker 用此模式，详见 plans 点3。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    workers = max(1, int(cfg.get("fetch_workers", 8)))

    out = {}
    n = len(symbols)
    done = 0
    # 审计#15（2026-09-24）：失败清单落盘——原实现失败单只静默返回空 df，一次网络抖动
    # 让部分股从选股池/基准消失、重跑结果即变且无据可查。现收集失败明细写
    # reports/fetch_failures_<时间戳>.txt（返回语义不变：失败仍返回空 df 不中断整体）。
    failures: list[tuple[str, str]] = []

    def _one(sym: str) -> tuple[str, pd.DataFrame]:
        # 严格 cache-only（P1-4）下跳过旧缓存键迁移的 os.rename（快照目录不可写）
        if not read_only and not cfg.get("strict_cache"):
            _migrate_legacy_cache(cache_dir, sym, adjust, start)
        try:
            return sym, get_stock_daily(sym, start=start, adjust=adjust,
                                        read_only=read_only, skip_missing=skip_missing)
        except Exception as e:
            if progress:
                print(f"  {sym} 失败: {str(e)[:60]}")
            failures.append((sym, str(e)[:200]))  # CPython list.append 原子，线程安全
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            out[sym] = df
            done += 1
            if progress and (done % 100 == 0 or done == n):
                print(f"  [{done}/{n}] 已处理...")

    if failures:
        from datetime import datetime as _dt
        try:
            Path("reports").mkdir(exist_ok=True)
            mf_path = Path("reports") / f"fetch_failures_{_dt.now():%Y%m%d_%H%M%S}.txt"
            mf_path.write_text(
                "\n".join(f"{sym}\t{err}" for sym, err in failures), encoding="utf-8")
            print(f"[WARN] {len(failures)}/{n} 只拉取失败（已从池中剔除）→ 明细 {mf_path}；"
                  f"重跑可恢复，勿在失败清单非空时采信当日结果", flush=True)
        except OSError:
            print(f"[WARN] {len(failures)}/{n} 只拉取失败（明细落盘失败）", flush=True)
    # 键序确定性（2026-10-08，docs/quant_system/已知口径偏差.md §10）：上方 as_completed
    # 按线程完成序回填 → out 的键序逐调用随机。下游 ML 训练（walk_forward._train_ml_model）
    # 的 sym_data/current/特征行序、截面均值 np.nanmean 的求和序全部继承该序 → 行序/浮点和序
    # 逐 run 不同 → LightGBM 早停在平坦验证曲线处翻转（同数据同 seed 出不同模型，
    # 实测同口径跨 run 超额摆 17pp）。返回按输入 symbols 序重排，切断该随机源；
    # 修复后 (数据, seed) → 确定模型。注意：修复本身也会换一次数值路径（相对历史 run）。
    return {sym: out[sym] for sym in symbols}


def get_index_daily(symbols: list[str], price_dfs: dict[str, pd.DataFrame],
                    start: str = "2020-01-01") -> pd.DataFrame:
    """
    由成分股价格合成指数日净值（等权，起点归1），用作回测基准的近似。
    比 akshare 的指数行情接口更稳，且与成分股数据口径一致。
    """
    import numpy as np
    # 取所有股票的日收益率，对齐日期
    rets = {}
    for sym, df in price_dfs.items():
        if df.empty:
            continue
        r = df.set_index("date")["close"].pct_change()
        rets[sym] = r
    if not rets:
        return pd.DataFrame()
    ret_df = pd.DataFrame(rets)
    # 等权：每日截面均值收益
    daily_ret = ret_df.mean(axis=1)
    daily_ret = daily_ret[daily_ret.index >= pd.Timestamp(start)]
    # 第一行 pct_change 为 nan，填 0 作为起点，并丢弃仍为 nan 的行
    daily_ret = daily_ret.fillna(0.0)
    nav = (1 + daily_ret).cumprod()
    return pd.DataFrame({"date": nav.index, "nav": nav.values})


def _fetch_index_real_raw(code: str, start: str, end: str) -> pd.DataFrame:
    """向 akshare 拉取真实指数日线 OHLCV，返回已 _normalize 的 df。

    新浪源优先（stock_zh_index_daily，列名已英文、更稳，同 _fetch_stock_daily_raw 范式），
    东财 index_zh_a_hist 兜底（本环境东财源常连不上）。

    指数行情无复权概念（非个股），故无 adjust 参数。RSRS 等择时需真实 high/low 做回归，
    故必须用真实指数行情而非 get_index_daily 的成分股合成版（合成版只有 nav 无 high/low）。

    新浪源 stock_zh_index_daily 无 start/end 参数（固定返回全部历史），需在 _normalize 后
    按 [start,end] 截断；东财源带 start/end。code 需带市场前缀（6/0 开头指数=sh，3/9 开头=sz）。
    """
    try:
        import akshare as ak
    except ImportError as e:
        raise DataFetchError("未安装 akshare，请 pip install akshare") from e

    # 指数代码前缀：000300/000905/000852 是沪市指数(sh)，3xxxxx 是深市(sz)
    prefix = "sh" if code.startswith(("0", "6")) else "sz"

    # 优先新浪源（列名已英文标准格式，且比东财源稳定）
    try:
        df = ak.stock_zh_index_daily(symbol=f"{prefix}{code}")
        if df is None or df.empty:
            raise DataFetchError(f"新浪源指数返回空: {code}")
        df = _normalize(df, "date")
        # 新浪源无 start/end 参数，截断到请求区间
        if start or end:
            mask = pd.Series(True, index=df.index)
            if start:
                mask &= df["date"] >= pd.Timestamp(start)
            if end:
                mask &= df["date"] <= pd.Timestamp(end)
            df = df[mask].reset_index(drop=True)
    except DataFetchError:
        raise
    except Exception as sina_err:
        # 回退东财源（列名中文，需映射；带 start/end）
        try:
            df = ak.index_zh_a_hist(symbol=code, period="daily",
                                    start_date=start.replace("-", ""),
                                    end_date=end.replace("-", ""))
            if df is None or df.empty:
                raise DataFetchError(f"东财源指数返回空: {code}")
            col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                       "最高": "high", "最低": "low", "成交量": "volume"}
            df = df.rename(columns=col_map)
            df = _normalize(df, "date")
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(
                f"拉取指数行情 {code} 失败（新浪:{sina_err} / 东财:{e}）") from e
    return df


def get_index_real_daily(index_name: str, start: str = "2015-01-01", end: str | None = None,
                         read_only: bool = False) -> pd.DataFrame:
    """拉取真实指数日线 OHLCV（含 high/low），带本地 parquet 缓存 + 增量更新。

    与 get_stock_daily 同范式（缓存键/原子写/增量/read_only），但：
    - 指数无复权（无 adjust 维度），缓存键仅含 index_name
    - 默认 start 较早（2015）：RSRS z-score 标准化窗口 M=600≈2.4年交易日，需足够历史预热

    index_name: "csi300"/"csi500"/"csi1000"
    返回: DataFrame[date, open, high, low, close, volume]，升序。

    read_only: True = cache-only（命中即读、不拉网不写）；未命中回退拉取。同 get_stock_daily。
    """
    from datetime import datetime as _dt

    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    max_days = cfg.get("max_cache_days", 1)
    full_refresh_days = cfg.get("full_refresh_days", 30)

    code = INDEX_SYMBOLS.get(index_name)
    if code is None:
        raise ValueError(f"未知指数: {index_name}，可选: {list(INDEX_SYMBOLS)}")

    path = _cache_path(cache_dir, f"index_real_{index_name}")
    end = end or _dt.now().strftime("%Y-%m-%d")

    # 严格 cache-only（P1-4）：命中即读、未命中 fail-loud
    if cfg.get("strict_cache"):
        if not path.exists():
            raise _snapshot.StrictCacheMiss(f"严格缓存模式：快照缺 {path.name}")
        return pd.read_parquet(path)

    # read_only：cache-only
    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        df = _fetch_index_real_raw(code, start, end)
        _atomic_write_parquet(path, df)
        return df

    # 1) 快路径：缓存未过期直读
    if not _need_refresh(path, max_days):
        return pd.read_parquet(path)

    # 2) 增量路径：有缓存且未超 full_refresh_days → 只补 last_date+1→end
    if path.exists():
        cached = pd.read_parquet(path)
        last_date = cached["date"].max() if not cached.empty else None
        days_since = ((pd.Timestamp(end) - pd.Timestamp(last_date)).days
                      if last_date is not None else full_refresh_days + 1)
        if last_date is not None and days_since <= full_refresh_days and days_since >= 0:
            inc_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if inc_start <= end:
                try:
                    new = _fetch_index_real_raw(code, inc_start, end)
                    if new is not None and not new.empty:
                        merged = pd.concat([cached, new], ignore_index=True)
                        merged = merged.drop_duplicates(subset=["date"], keep="last")
                        merged = merged.sort_values("date").reset_index(drop=True)
                        _atomic_write_parquet(path, merged)
                        return merged
                    path.touch()  # 无新数据，刷新 mtime 标记已检查
                    return cached
                except Exception:
                    path.touch()
                    return cached  # 增量失败回退缓存，不中断回测
            path.touch()
            return cached

    # 3) 全量路径：无缓存 / 超过 full_refresh_days（也用于刷新历史）
    df = _fetch_index_real_raw(code, start, end)
    _atomic_write_parquet(path, df)
    return df


def get_stock_industry_map(read_only: bool = False) -> dict[str, str]:
    """
    获取 A 股 {股票代码: 申万一级行业名} 映射。
    用于行业中性化、行业暴露分析。

    数据源：akshare 申万一级行业指数（sw_index_first_info + index_component_sw）。
    申万源比东财稳定（东财 stock_board_industry_* 频繁被限流）。
    行业归属变化慢，缓存 30 天。
    返回: {symbol(6位): 行业名}，拉取失败的股票不含在内

    read_only: True = cache-only（命中即读、不拉网不写）；未命中回退拉取。见 plans 点3。
    """
    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    path = _cache_path(cache_dir, "industry_map_sw1")
    # 严格 cache-only（P1-4）：命中即读（忽略 30 天缓存期）、未命中 fail-loud
    if cfg.get("strict_cache"):
        if not path.exists():
            raise _snapshot.StrictCacheMiss(f"严格缓存模式：快照缺 {path.name}")
        df = pd.read_parquet(path)
        return dict(zip(df["symbol"].astype(str).str.zfill(6), df["industry"]))
    if path.exists():
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age <= timedelta(days=30) or read_only:
            df = pd.read_parquet(path)
            return dict(zip(df["symbol"].astype(str).str.zfill(6), df["industry"]))

    import akshare as ak
    industry_map: dict[str, str] = {}
    try:
        boards = ak.sw_index_first_info()
    except Exception as e:
        raise DataFetchError(f"获取申万一级行业列表失败: {e}") from e

    failed_boards: list[str] = []
    for _, row in boards.iterrows():
        code = row["行业代码"]      # 如 801010.SI
        name = row["行业名称"]      # 如 农林牧渔
        sw_code = code.split(".")[0]
        try:
            cons = ak.index_component_sw(symbol=sw_code)
        except Exception:
            failed_boards.append(name)
            continue
        if cons is None or cons.empty:
            failed_boards.append(name)
            continue
        for sym in cons["证券代码"].astype(str).str.zfill(6):
            if sym not in industry_map:
                industry_map[sym] = name

    # 审计#16（2026-09-24 修）：部分板块拉取失败时不再无条件写 30 天缓存——
    # 缺行业股在行业中性化中被静默错配，空洞 map 一旦缓存将错一个月。
    # 失败 >20% 视为拉取不完整：报错走调用方失败路径（跳过中性化），不落缓存。
    if failed_boards:
        print(f"[WARN] 行业映射 {len(failed_boards)}/{len(boards)} 个板块拉取失败: "
              f"{failed_boards}", flush=True)
    if len(failed_boards) > len(boards) * 0.2 or not industry_map:
        raise DataFetchError(
            f"行业映射拉取不完整（{len(failed_boards)}/{len(boards)} 板块失败，"
            f"仅得 {len(industry_map)} 只），不写缓存")

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(path, pd.DataFrame({"symbol": list(industry_map.keys()),
                                              "industry": list(industry_map.values())}))
    return industry_map


if __name__ == "__main__":
    # 自测：拉一支股票和一支期货
    print("拉取 A 股 000001 ...")
    s = get_stock_daily("000001")
    print(f"  行数={len(s)}  最新={s['date'].iloc[-1].date()}  close={s['close'].iloc[-1]}")

    print("拉取期货 RB0 ...")
    f = get_futures_daily("RB0")
    print(f"  行数={len(f)}  最新={f['date'].iloc[-1].date()}  close={f['close'].iloc[-1]}")