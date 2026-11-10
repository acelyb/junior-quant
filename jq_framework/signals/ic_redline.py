"""signals/ic_redline.py — 滚动 OOS IC 红线（验证强化与实验防重复执行计划 P2-1，2026-10-22）。

服务模型预测力监控：每次重训时回溯评估"结果已完整"的历史 artifact 在其服务窗口内的
真实 OOS 截面 rank-IC——Spearman(模型预测分, 未来 horizon 日 close-to-close 收益)，逐日
截面、服务窗内取均值，追加 reports/signal_health.csv。滚动 K 窗均值 < 历史分布 P5 →
fail-loud（train_ml_daily 不落盘新 artifact + exit 1 + 保旧 artifact），与新鲜度红线
（ml_serving.max_age_days）同级。

时间对齐（为何评估"上上个"artifact）：h10 前向收益在评分日 u 之后第 horizon 个交易日
才完整——重训日 t 的上一份 artifact（data_end≈t−10）服务期 (t−10, t] 的结果尚未走完，
可评估的最新完整窗口 = 上上份 artifact 的服务期。缺口由多 artifact 追评机制自然补齐
（停机数日后的重训会连评多个窗口）。

参考分布自累积（同口径原则）：signal_health.csv 自身的 ic_mean 列按重训节奏（默认 10
交易日）一行行累积——参考分布与被监控读数同训练口径（服务端扩展窗）、同节奏、同截面
惯例，无需离线回测先验。参考行 = 有效行去掉最近 k_rolling 行（近期读数若同时进参考会
把 P5 拉向自己、门自我抵消），故武装条件是**有效行数 ≥ min_rows_arm + k_rolling**
（默认 8+6=14 行 ≈ 140 个交易日 ≈ 半年多）；未达前只记录不武装（引导期）。武装后每次
重训用参考行的 P5 作阈值。离线回测先验（walk_forward 逐窗 IC 落盘后钉快照重跑一次）
可作为后续补充，但训练口径差异（滚动 252 日 vs 服务端扩展窗）会让阈值偏松偏紧不可控，
默认不用。

评估口径（与训练/信号侧逐字同源，防"监控口径 ≠ 生产口径"的假红线）：
- 特征 = composite_ml._build_feature_matrix（NaN→当日截面列均值填）
- 标签 = factors.ic._future_return（horizon 日 close-to-close）
- 截面 = 传入的 valid 池（train_ml_daily 侧=universe∪基准跟随池[∪宽训练池]，与训练同池）
- artifact 因子装配契约 vs 当前 config 不一致 → 该行 status=contract_mismatch、ic 置空
  （旧 artifact 不能用新因子集忠实评估；不进参考分布、不触发门）

config ml_serving.ic_redline 可覆盖参数（enabled/k_rolling/min_rows_arm/csv_path）；
缺省用本模块默认值（批处理运行期间不写 config.yaml——场景 config_hash 逐场景合并自
磁盘配置，批中改动会撕裂批内 hash 一致性；配置块在批后补，见 train_ml_daily 注释）。

用法（由 train_ml_daily 在训练成功后、落盘前调用）：
    from jq_framework.signals.ic_redline import evaluate_oos_ic
    gate = evaluate_oos_ic(models_dir=..., horizon=..., valid=..., all_dates=...,
                           factors=..., series_caches=..., cfg=cfg)
    if gate["fired"]:
        return 1   # 不落盘，保旧 artifact，人工核查 reports/signal_health.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CSV = ROOT / "reports" / "signal_health.csv"
_COLUMNS = ["eval_date", "artifact", "serving_start", "serving_end", "n_dates",
            "n_syms_med", "ic_mean", "ic_std", "status", "armed", "rolling_k",
            "p5_ref", "fired"]

# 与新鲜度红线同级的最低截面规模：不足则该评分日不进 IC（截面太小 Spearman 无意义）
_MIN_CROSS_SECTION = 30


def _artifact_dirs(models_dir: str, horizon: int) -> list[Path]:
    """按 data_end 日期排序的全部 artifact 目录（ml_h{horizon}_YYYYMMDD）。"""
    root = Path(models_dir) if Path(models_dir).is_absolute() else ROOT / models_dir
    if not root.exists():
        return []
    prefix = f"ml_h{int(horizon)}_"
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix))


def _load_artifact(d: Path):
    """加载指定 artifact 目录（文件约定与 ml_model_store.load_model_artifact 一致，
    但按目录直取、不做新鲜度/契约拦截——契约由调用方按评估需要单独判）。"""
    import lightgbm as lgb

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("ensemble"):
        from jq_framework.backtest.walk_forward import _SeedEnsemble
        members = [lgb.Booster(model_file=str(d / f"model_s{s}.txt"))
                   for s in meta["seeds"]]
        model = _SeedEnsemble(members, list(meta["seeds"]))
    else:
        model = lgb.Booster(model_file=str(d / "model.txt"))
    return model, meta


def _date_pos(all_dates: list, target) -> int:
    """target 在交易日轴上的位置（bisect，允许 target 不在轴上时取左邻）。"""
    arr = pd.DatetimeIndex(pd.Timestamp(x) for x in all_dates)
    return int(arr.searchsorted(pd.Timestamp(target)))


def _evaluated_artifacts(csv_path: Path) -> set[str]:
    """signal_health.csv 里已有行的 artifact 目录名（幂等键：一个 artifact 一行）。"""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r.get("artifact", "") for r in csv.DictReader(f)}


def _serving_window_ics(valid: dict, all_dates: list, span: list,
                        factors: list, model, feature_names: list,
                        series_caches: dict, horizon: int) -> tuple[list[float], int]:
    """对服务窗 span 内每个评分日算截面 Spearman IC。返回 (ics, 截面规模中位数)。

    逐日截面构造镜像 walk_forward._build_train_xy 的采样日切片（含 date 行、iloc[:pos+1]），
    预测走 composite_score_ml（=信号侧同款），标签走 _future_return（=训练标签同款）。
    """
    from jq_framework.factors.composite_ml import composite_score_ml
    from jq_framework.factors.ic import _future_return

    sym_data = {s: df.set_index("date") for s, df in valid.items() if not df.empty}
    ics, nsyms = [], []
    for u in span:
        current = {}
        for sym, d in sym_data.items():
            pos = int(np.searchsorted(d.index.values, np.datetime64(u)))
            if pos < len(d) and d.index[pos] == u:
                current[sym] = d.iloc[:pos + 1]
        if len(current) < _MIN_CROSS_SECTION:
            continue
        score = composite_score_ml(current, factors, model, feature_names,
                                   series_caches=series_caches)
        if score.empty:
            continue
        frs = {}
        for sym, d in sym_data.items():
            if sym not in current:
                continue
            closes = d["close"]
            pos = int(np.searchsorted(closes.index.values, np.datetime64(u)))
            fr = _future_return(closes.index, closes, pos, horizon)
            if fr is not None and np.isfinite(fr):
                frs[sym] = fr
        common = [s for s in score.index
                  if s in frs and np.isfinite(score.get(s, np.nan))]
        if len(common) < _MIN_CROSS_SECTION:
            continue
        ic = stats.spearmanr(score[common].values,
                             np.array([frs[s] for s in common])).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
            nsyms.append(len(common))
    return ics, int(np.median(nsyms)) if nsyms else 0


def redline_gate(rows: list[dict], k_rolling: int, min_rows_arm: int) -> dict:
    """纯门逻辑（可单测）：rows=signal_health 全量行（status=ok 且 ic_mean 非 NaN 才有效）。

    参考分布 = 滚动窗**之前**的历史行（剔除最近 k_rolling 行——近期读数同时出现在两侧
    会把 P5 拉向自己、门自我抵消；参考须是先验）。armed=False 时 fired 恒 False。
    返回 {armed, fired, rolling_k, p5_ref, n_ref}。
    """
    def _val(r):
        try:
            v = float(r.get("ic_mean", ""))
        except (TypeError, ValueError):
            return None
        return v if (v is not None and np.isfinite(v)) else None

    vals = [v for r in rows if r.get("status") == "ok" for v in [_val(r)] if v is not None]
    ref = vals[:-k_rolling] if len(vals) > k_rolling else []
    if len(ref) < min_rows_arm:
        return {"armed": False, "fired": False, "rolling_k": None,
                "p5_ref": None, "n_ref": len(ref)}
    p5 = float(np.percentile(ref, 5))
    rolling = float(np.mean(vals[-k_rolling:]))
    return {"armed": True, "fired": bool(rolling < p5),
            "rolling_k": rolling, "p5_ref": p5, "n_ref": len(ref)}


def evaluate_oos_ic(models_dir: str, horizon: int, valid: dict, all_dates: list,
                    factors: list, series_caches: dict, cfg: dict | None = None,
                    k_rolling: int = 6, min_rows_arm: int = 8,
                    health_csv: str | Path | None = None, verbose: bool = True) -> dict:
    """回溯评估全部"结果已完整且未评过"的 artifact 服务窗口，追加 signal_health.csv。

    必须在 train_ml_daily 训练成功后、save_model_artifact 前调用（fired → 调用方
    exit 1 不落盘）。factors/series_caches 复用重训现场（零额外装配成本）。
    返回门状态 dict（含本次评估行数与 fired）。
    """
    # config 覆盖（ml_serving.ic_redline 块；缺省走参数默认值）
    if cfg is not None:
        icc = ((cfg.get("multifactor", {}) or {}).get("ml_serving", {}) or {}) \
            .get("ic_redline", {}) or {}
        if icc.get("enabled") is False:
            if verbose:
                print("[ic_redline] config 已禁用（ml_serving.ic_redline.enabled=false）",
                      flush=True)
            return {"fired": False, "armed": False, "disabled": True, "n_new_rows": 0}
        k_rolling = int(icc.get("k_rolling", k_rolling))
        min_rows_arm = int(icc.get("min_rows_arm", min_rows_arm))
        health_csv = icc.get("csv_path") or health_csv
    csv_path = Path(health_csv) if health_csv else DEFAULT_CSV
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    n_end = len(all_dates)
    end_t = all_dates[-1]
    evaluated = _evaluated_artifacts(csv_path)
    dirs = _artifact_dirs(models_dir, horizon)
    new_rows = []

    # 相邻 artifact 对 (A,B)：A 的服务窗 = (A.data_end, B.data_end]。最后一份 artifact
    # 的服务窗延伸到"现在"、结果未完整，不评——只评闭合窗。
    for a_dir, b_dir in zip(dirs, dirs[1:]):
        if a_dir.name in evaluated:
            continue
        a_meta = json.loads((a_dir / "meta.json").read_text(encoding="utf-8"))
        a_data_end = pd.Timestamp(a_meta["data_end"])
        b_meta = json.loads((b_dir / "meta.json").read_text(encoding="utf-8"))
        b_data_end = pd.Timestamp(b_meta["data_end"])

        # 结果完整性：服务窗末 + horizon 个交易日须已过去（全局轴判，逐标的仍按各自
        # 日历 NaN 过滤）
        pos_b = _date_pos(all_dates, b_data_end)
        if pos_b + horizon >= n_end:
            continue  # 尚未完整，留给后续重训追评

        # 服务窗交易日序列（全局轴 (a_end, b_end]）
        pos_a = _date_pos(all_dates, a_data_end)
        span = [all_dates[i] for i in range(pos_a + 1, pos_b + 1)]
        if len(span) < 3:
            row = {"eval_date": str(pd.Timestamp(end_t).date()), "artifact": a_dir.name,
                   "serving_start": str((a_data_end + pd.Timedelta(days=1)).date()),
                   "serving_end": str(b_data_end.date()), "n_dates": len(span),
                   "n_syms_med": 0, "ic_mean": "", "ic_std": "",
                   "status": "insufficient_span"}
            new_rows.append(row)
            if verbose:
                print(f"[ic_redline] {a_dir.name}: 服务窗仅 {len(span)} 日，记行不评",
                      flush=True)
            continue

        # 因子装配契约：旧 artifact 只能用与其训练时一致的因子集评估
        status, ics, nsyms = "ok", [], 0
        if cfg is not None:
            from jq_framework.signals.ml_model_store import _canonical, _factor_spec_list
            if _canonical(_factor_spec_list(cfg)) != _canonical(a_meta.get("factor_specs", [])):
                status = "contract_mismatch"
        if status == "ok":
            model, _ = _load_artifact(a_dir)
            feature_names = a_meta.get("feature_names") or \
                [f.name for f, _ in factors]
            ics, nsyms = _serving_window_ics(valid, all_dates, span, factors, model,
                                             feature_names, series_caches, horizon)
            if len(ics) < 3:
                status = "insufficient_dates"

        if status == "ok":
            row = {"eval_date": str(pd.Timestamp(end_t).date()), "artifact": a_dir.name,
                   "serving_start": str(pd.Timestamp(span[0]).date()),
                   "serving_end": str(pd.Timestamp(span[-1]).date()), "n_dates": len(ics),
                   "n_syms_med": nsyms,
                   "ic_mean": f"{float(np.mean(ics)):.6f}",
                   "ic_std": f"{float(np.std(ics, ddof=1)):.6f}" if len(ics) > 1 else "",
                   "status": "ok"}
            if verbose:
                print(f"[ic_redline] {a_dir.name}: 服务窗 {row['serving_start']}→"
                      f"{row['serving_end']} n={len(ics)} IC均值={row['ic_mean']}",
                      flush=True)
        else:
            row = {"eval_date": str(pd.Timestamp(end_t).date()), "artifact": a_dir.name,
                   "serving_start": str(pd.Timestamp(span[0]).date()),
                   "serving_end": str(pd.Timestamp(span[-1]).date()),
                   "n_dates": len(ics), "n_syms_med": nsyms,
                   "ic_mean": "", "ic_std": "", "status": status}
            if verbose:
                print(f"[ic_redline] {a_dir.name}: status={status}（记行不进参考分布）",
                      flush=True)
        new_rows.append(row)

    # 追加落盘（append 语义，文件不存在则带表头新建）
    if new_rows:
        exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_COLUMNS)
            if not exists:
                w.writeheader()
            for r in new_rows:
                w.writerow({k: r.get(k, "") for k in _COLUMNS})
    elif not csv_path.exists():
        # 无新行（首份 artifact 尚无闭合服务窗）也建表头文件：让引导期在
        # reports/ 下可见（运维可分辨"未开始积累"与"红线未接线"）
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_COLUMNS).writeheader()

    # 门核对（读取含历史的全量行；引用历史而非仅本次新行）
    # 首次运行 new_rows 为空 → csv 从未被创建，读取须容忍缺失（按空处理）：
    # 原实现在此 open 不存在的文件抛 FileNotFoundError，导致 IC 红线自上线以来
    # 首次运行必然崩溃、且崩溃点位于 train_ml_daily 落盘之前（新 artifact 一并丢失）。
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
    else:
        all_rows = []
    gate = redline_gate(all_rows, k_rolling, min_rows_arm)
    gate["n_new_rows"] = len(new_rows)
    if verbose:
        if gate["armed"]:
            print(f"[ic_redline] 门已武装：滚动{k_rolling}窗均值 "
                  f"{gate['rolling_k']:+.4f} vs 历史 P5 {gate['p5_ref']:+.4f}"
                  f"（参考 {gate['n_ref']} 行）"
                  f"{'——触发' if gate['fired'] else ''}", flush=True)
        else:
            print(f"[ic_redline] 引导期（有效行 {gate['n_ref']}/{min_rows_arm}），"
                  f"只记录不武装", flush=True)
    return gate
