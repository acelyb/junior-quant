"""
数据快照层：不可变冻结 + 写时复制 + 清理（验证强化与实验防重复执行计划 P1-4，2026-10-21）。

背景（用户指令 2026-10-20）：取数缓存的例行刷新会回溯改写历史（qfq 前复权随除权重定
基准 / 成分股修订 / 双源全量重拉），同一配置的历史读数随刷新漂移（五天内三次实证，
见 docs/quant_system/已知口径偏差.md §8）。本层把"改写"变成"另存一份"：

- freeze(): 硬链接整目录复制（cp -al）把缓存状态冻结为 data/snapshots/<id>/。
  全部缓存写入均经 fetcher._atomic_write_parquet（mkstemp 唯一临时名 + os.replace
  原子替换）——替换只换目录项，旧 inode 经快照硬链接原样保留 = 天然写时复制，
  快照不可变（ext4 无块级去重，被改写文件保留旧数据块 ≈ 每代次 1-2GB，清理策略兜底）。
- freeze_barrier(): 挂在 _atomic_write_parquet 入口——每天第一次内容改写前先冻结旧
  状态（当日幂等，双检 + flock）。保证任何改写发生前其"改写前状态"已留存，
  快照代际连续：S_n = 第 n 次刷新前状态。
- 钉住回测: compare_ml --snapshot <id> 自动设置 QUANT_CACHE_DIR=快照目录 +
  QUANT_STRICT_CACHE=1（环境变量随进程树传播到 worker）→ 严格 cache-only：
  命中即读（跳过 TTL），未命中 fail-loud 报错退出，绝不拉网、绝不写快照。
  注意：现有 read_only 模式未命中会回退拉取（worker 预热兜底语义），与严格模式不同，
  严格模式才承担"不可变保证"，二者不得混用。
- 指纹兼容: 与原 scripts/world_fingerprint.py（P0-3）算法逐字一致——
  sha1(universe | const_{u}.parquet 内容 sha1 | 全缓存 (文件名,大小,mtime_ns) 排序摘要)。
  硬链接保留 mtime_ns → 对快照目录重算指纹 = 冻结时刻的指纹，历史登记行（如
  7f125bd0…）可直接作为钉住 run 的对照指纹。登记簿列名已改为 snapshot_fingerprint，
  指纹值与全部历史行兼容。
- gc(): 登记簿 reports/trial_registry.csv 仍引用的 + 最新 keep_newest 份保留，
  其余删除；检测到回测进程运行时拒绝执行。

环境变量（与既有 QUANT_CACHE_DIR 同约定，见 fetcher._load_config）：
  QUANT_SNAPSHOTS_ROOT  快照根目录（默认 data/snapshots；测试隔离用）
  QUANT_SNAPSHOT_BARRIER=0  紧急关闭改写前屏障（默认开）
  QUANT_STRICT_CACHE=1  严格 cache-only（compare_ml --snapshot 自动设置）
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# stat 摘要覆盖的缓存文件模式（与 P0-3 指纹算法逐字一致，勿改动——改了就断历史兼容）
FILE_PATTERNS = ("stock_*.parquet", "index_*.parquet", "fin_indicator_em_*.parquet")

FINGERPRINT_LOG_CSV = REPO_ROOT / "reports" / "cache_fingerprint_log.csv"


class StrictCacheMiss(RuntimeError):
    """严格缓存模式下缓存未命中（fail-loud，绝不静默拉网/落盘）。"""


class SnapshotError(RuntimeError):
    """快照操作失败（解析/冻结/清理）。"""


def snapshots_root() -> Path:
    return Path(os.environ.get("QUANT_SNAPSHOTS_ROOT")
                or (REPO_ROOT / "data" / "snapshots"))


def _resolve_dir(d: str | Path) -> Path:
    p = Path(d)
    return p if p.is_absolute() else REPO_ROOT / p


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_digest(cache: Path) -> tuple[str, int]:
    """(文件名, 大小, mtime_ns) 排序拼接 sha1。任何写入必改 mtime_ns → 必留痕。
    与 P0-3 逐字一致（含 pattern 顺序与行格式）。"""
    lines = []
    for pat in FILE_PATTERNS:
        for p in sorted(cache.glob(pat)):
            st = p.stat()
            lines.append(f"{p.name}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest(), len(lines)


def fingerprint(universe: str, cache_dir: str | Path = "data/cache") -> dict:
    """计算指定 universe 在指定缓存目录上的数据快照指纹（P0-3 算法，值兼容历史登记行）。"""
    cache = _resolve_dir(cache_dir)
    const_path = cache / f"const_{universe}.parquet"
    if not const_path.exists():
        raise FileNotFoundError(f"成分股缓存不存在: {const_path}")
    stat_digest, n_files = _stat_digest(cache)
    const_sha1 = _sha1_file(const_path)
    fp = hashlib.sha1(f"{universe}|{const_sha1}|{stat_digest}".encode()).hexdigest()
    return {
        "universe": universe,
        "snapshot_fingerprint": fp,
        "const_sha1": const_sha1,
        "cache_stat_sha1": stat_digest,
        "n_cache_files": n_files,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def record(fp: dict, log_csv: Path = FINGERPRINT_LOG_CSV) -> None:
    """追加一行到指纹日志（登记簿伴生表：批级留痕）。"""
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_csv.exists()
    import csv
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["computed_at", "universe", "snapshot_fingerprint", "n_cache_files"])
        w.writerow([fp["computed_at"], fp["universe"], fp["snapshot_fingerprint"],
                    fp["n_cache_files"]])


# ============ 冻结 ============

@contextmanager
def _flock(path: Path):
    """排它文件锁（跨进程跨线程；flock 按 open 文件描述互斥，同进程两个 fd 同文件也会互等，
    故屏障与冻结须用不同锁文件，见 freeze_barrier 注）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    try:
        yield f
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _const_map(cache: Path) -> dict[str, str]:
    """全部成分股缓存 {const 名: 内容 sha1}（通常 4-8 个小文件）。"""
    out = {}
    for p in sorted(cache.glob("const_*.parquet")):
        out[p.name] = _sha1_file(p)
    return out


def freeze(cache_dir: str | Path = "data/cache", reason: str = "") -> str:
    """把当前缓存状态冻结为不可变快照，返回快照 id。

    id = <YYYYmmdd-HHMMSS>-<指纹12>；指纹12 取 stat 摘要 + 全部 const 内容哈希的合成
    （universe 无关，标识快照本体；per-universe 指纹在 manifest 里按需可查）。
    内容级幂等：当前状态与某现有快照完全一致 → 复用该快照 id，不新增代际。
    """
    cache = _resolve_dir(cache_dir)
    if not cache.exists():
        raise SnapshotError(f"缓存目录不存在，无物可冻: {cache}")
    root = snapshots_root()
    stat_digest, n_files = _stat_digest(cache)
    cmap = _const_map(cache)
    # 内容级幂等：已有快照与当前状态完全一致（stat 摘要 + 全部 const 内容）→ 直接复用，
    # 状态未变不新增代际（同日重复冻结零成本；"刷新无实改"的日子也不产生空代际）。
    for m in list_snapshots():
        if (m.get("cache_stat_sha1") == stat_digest
                and m.get("const_sha1") == cmap):
            return m["snapshot_id"]
    body = hashlib.sha1((f"{stat_digest}|"
                        + ",".join(f"{k}:{v}" for k, v in sorted(cmap.items()))
                        ).encode("utf-8")).hexdigest()
    sid = f"{datetime.now():%Y%m%d-%H%M%S}-{body[:12]}"
    dest = root / sid
    with _flock(root / ".freeze.lock"):
        if dest.exists():
            return sid
        root.mkdir(parents=True, exist_ok=True)
        # cp -al 硬链接整目录：不复制数据块（inode 共享），mtime_ns 原样保留
        # → 快照重算指纹 = 冻结时刻指纹。写入侧全部经 mkstemp+os.replace 原子替换，
        # cp 期间并发的替换只会让快照拿到旧/新完整版本之一，绝无半截。
        r = subprocess.run(["cp", "-al", str(cache), str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise SnapshotError(f"cp -al 失败: {r.stderr.strip()[:200]}")
        manifest = {
            "snapshot_id": sid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source_cache_dir": str(cache),
            "cache_stat_sha1": stat_digest,
            "n_cache_files": n_files,
            "const_sha1": cmap,
            # per-universe 指纹（P0-3 算法值）：manifest_fp(manifest, universe) 按需重算
        }
        (root / f"{sid}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return sid


def manifest_fp(manifest: dict, universe: str) -> str:
    """从 manifest 组件重算 per-universe 指纹（P0-3 算法，与对快照目录直接重算逐字一致）。"""
    key = f"const_{universe}.parquet"
    if key not in manifest.get("const_sha1", {}):
        raise SnapshotError(
            f"快照 {manifest.get('snapshot_id')} 冻结时无 {key}，不能为 universe={universe} 重算指纹")
    return hashlib.sha1(
        f"{universe}|{manifest['const_sha1'][key]}|{manifest['cache_stat_sha1']}".encode()
    ).hexdigest()


# ============ 改写前屏障 ============

# 进程内"当日已过屏障"快路径缓存（无锁；跨日自动失效）。按缓存目录分键——
# 同进程可能写多个缓存目录（data/cache 与 data/cache_long），每个目录各自首写冻结。
_barrier_state: dict[str, date | None] = {}


def freeze_barrier(cache_dir: str | Path) -> None:
    """改写前屏障：每个缓存目录每天第一次内容改写前，先冻结该目录当前状态（当日幂等）。

    挂在 fetcher._atomic_write_parquet 入口（全部缓存写入的唯一咽喉）。三层快路径：
    进程内 (目录,日期) 缓存 → 无锁 sentinel 检查 → 锁内双检后冻结。屏障失败 fail-loud
    （冻结失败说明旧状态留不住，宁可中断也不静默改写历史）。

    sentinel 按缓存目录分文件（.last_freeze_<目录指纹8>）：幂等粒度=目录级而非根级——
    否则同日先写过 data/cache 再写 data/cache_long 时，后者会因前者的 sentinel
    被短路漏冻（首版端到端实测缺陷）。
    """
    if os.environ.get("QUANT_SNAPSHOT_BARRIER") == "0":
        return
    today = date.today()
    cache = _resolve_dir(cache_dir)
    barrier_key = str(cache.resolve())
    if _barrier_state.get(barrier_key) == today:
        return
    # 钉住/误配保护：被写目录本身在快照根下 → 不许把快照再冻成快照
    try:
        cache.resolve().relative_to(snapshots_root().resolve())
        return
    except ValueError:
        pass
    root = snapshots_root()
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / ("last_freeze_" + hashlib.sha1(barrier_key.encode()).hexdigest()[:8])
    if sentinel.exists():
        if sentinel.read_text(encoding="utf-8").split("\n")[0] == today.isoformat():
            _barrier_state[barrier_key] = today
            return
    # 屏障锁用独立文件（.barrier.lock ≠ freeze 内部的 .freeze.lock）：flock 按 open
    # 文件描述互斥——同进程 barrier 持 A 锁再进 freeze 拿 B 锁，若 A==B 即自锁（首次
    # 端到端测试实测）。并发安全不靠同一把锁：freeze 内部自有锁串行化真正拷贝，
    # 屏障锁只串行化"sentinel 判定+冻结"决策段，两个写进程不会重复冻结。
    with _flock(root / ".barrier.lock"):
        if sentinel.exists():
            if sentinel.read_text(encoding="utf-8").split("\n")[0] == today.isoformat():
                _barrier_state[barrier_key] = today
                return
        sid = freeze(cache, reason="barrier")
        sentinel.write_text(f"{today.isoformat()}\n{sid}\n", encoding="utf-8")
        _barrier_state[barrier_key] = today
    print(f"[数据快照] 改写前屏障已冻结 {sid}（写时复制，改写后旧状态仍完整保留）",
          flush=True)


# ============ 快照解析与钉住 ============

def list_snapshots() -> list[dict]:
    """全部快照 manifest，按冻结时间升序。"""
    root = snapshots_root()
    out = []
    for m in sorted(root.glob("*.manifest.json")):
        try:
            out.append(json.loads(m.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda x: x.get("created_at", ""))
    return out


def resolve_snapshot(id_or_prefix: str) -> Path:
    """按 id 或唯一前缀解析快照目录；歧义/未找到即报错退出。"""
    root = snapshots_root()
    matches = [d for d in root.iterdir()
               if d.is_dir() and (d.name == id_or_prefix
                                  or d.name.startswith(id_or_prefix)
                                  or id_or_prefix in d.name)]
    if not matches:
        avail = [d.name for d in root.iterdir() if d.is_dir()]
        raise SnapshotError(f"快照未找到: {id_or_prefix}；现有: {avail}")
    if len(matches) > 1:
        raise SnapshotError(f"快照前缀歧义: {id_or_prefix} → {[d.name for d in matches]}")
    return matches[0]


def pinned_fingerprint(universe: str, cache_dir: str | Path | None = None) -> dict:
    """钉住运行用：从快照 manifest 读指纹（不现算活动缓存——活动缓存前进会记错）。

    cache_dir = 生效快照目录（QUANT_CACHE_DIR）；manifest 缺失则回退对快照目录直接
    重算（算法一致，硬链接保 mtime，值相同）。
    """
    cache = _resolve_dir(cache_dir or os.environ.get("QUANT_CACHE_DIR") or "data/cache")
    mpath = snapshots_root() / f"{cache.name}.manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        return {
            "universe": universe,
            "snapshot_fingerprint": manifest_fp(manifest, universe),
            "const_sha1": manifest["const_sha1"].get(f"const_{universe}.parquet", ""),
            "cache_stat_sha1": manifest["cache_stat_sha1"],
            "n_cache_files": manifest["n_cache_files"],
            "computed_at": manifest["created_at"],
            "source": f"manifest:{manifest['snapshot_id']}",
        }
    return fingerprint(universe, cache)  # manifest 丢失时的等价回退


# ============ 清理 ============

def gc(keep_newest: int = 2, assume_yes: bool = False) -> list[str]:
    """删除登记簿不再引用且非最新的快照。返回被删除的快照 id 列表。

    保留规则：reports/trial_registry.csv 任一行引用的（按 per-universe 指纹匹配）
    + 最新 keep_newest 份。检测到回测进程运行时拒绝执行（防删正在读的快照）。
    """
    import csv
    root = snapshots_root()
    # 有回测进程在跑 → 拒绝（宁可不清也不删正在被读的快照）
    r = subprocess.run(["pgrep", "-f", "compare_ml|walk_forward"],
                        capture_output=True, text=True)
    if r.returncode == 0:
        raise SnapshotError("检测到回测进程在运行，拒绝执行清理（批后重试）")
    snaps = list_snapshots()
    if not snaps:
        return []
    referenced: set[str] = set()
    reg_csv = REPO_ROOT / "reports" / "trial_registry.csv"
    if reg_csv.exists():
        for row in csv.DictReader(open(reg_csv, encoding="utf-8")):
            fp = (row.get("snapshot_fingerprint") or row.get("world_fingerprint") or "").strip()
            if fp:
                referenced.add(fp)
    keep_ids = set()
    for m in snaps:
        # 引用判定：登记行指纹 = sha1(universe|const_sha1|stat)，universe 集合无法从指纹
        # 反解 → 用 manifest 预存的全部 const 名作 universe 候选逐一重算比对（保守）。
        for const_name in m.get("const_sha1", {}):
            u = const_name[len("const_"):-len(".parquet")]
            try:
                if manifest_fp(m, u) in referenced:
                    keep_ids.add(m["snapshot_id"])
                    break
            except SnapshotError:
                continue
    newest = [m["snapshot_id"] for m in snaps[-keep_newest:]]
    keep_ids.update(newest)
    victims = [m["snapshot_id"] for m in snaps if m["snapshot_id"] not in keep_ids]
    if not victims:
        return []
    if not assume_yes:
        print("将删除快照:", victims)
        raise SnapshotError("需 --yes 确认（dry-run 未删任何文件）")
    for sid in victims:
        shutil.rmtree(root / sid, ignore_errors=True)
        (root / f"{sid}.manifest.json").unlink(missing_ok=True)
    return victims
