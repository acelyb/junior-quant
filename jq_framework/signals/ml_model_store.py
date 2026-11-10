"""ml_model_store.py — ML 合成模型的持久化存取与生产因子装配（每日信号服务化，路线 b）。

背景（docs/stock/A股_ML断层交接.md；2026-09-24 用户拍板"业界也是持久化模型吧"）：
config weighting=ml 后，walk_forward 每窗口现场训练 LightGBM，而 signals/daily_portfolio.py
原本只走 static composite_score → 每日信号与回测断层。本模块把"训练好的模型 + 训练时冻结的
特征契约"落盘为 artifact 目录，供每日信号加载：

  {models_dir}/ml_h{horizon}_{data_end:%Y%m%d}/
      model.txt    LightGBM 模型文本（booster.save_model，截到 best_iteration）
      meta.json    训练契约：horizon/feature_names/factor_specs/训练窗/universe/diag

train/serve 零偏斜的保证（三层）：
  1. 训练器（scripts/train_ml_daily.py）直接复用 walk_forward._train_ml_model +
     _build_series_caches，不重写任何特征工程/训练逻辑；
  2. 服务侧 build_production_factors 逐字镜像 walk_forward.py L666-722 的因子装配
     （load_factor → ml.augmented_factors 展开 → industry_neutral → 基本面注入），
     calc_cross_section 产生的特征与训练时同口径；
  3. 加载时强校验（fail-loud SystemExit）：因子装配契约与当前 config 逐字一致
     （防"换因子集后旧模型静默打分"）+ 新鲜度（距 data_end 超过 max_age_days 个交易日
     即拒出信号，防长假/停跑后拿陈旧模型当当天信号）。

基准跟随规则镜像 research/attribution.py L158：csi1000 同池（训练池=universe），
其余对 config benchmark（csi300 宽基）——训练池与回测 all_syms 同构。

扩展窗语义（docs/quant_system/已知口径偏差.md §7）：walk_forward 训练是"自 fetch 起点
累积"的扩展窗（fetch 起点 = bt.start − 1.6×252 日历日），训练器沿用同语义；
2018 口径 A/B 的 fetch 起点 = 2016-11-24（config ml_serving.train_start 已钉住）。

注意：因子装配段与 walk_forward.py L666-722 是**受控复制**（回测热路径不引本模块，
避免改动在跑基线的代码）；两侧任何因子装配改动必须同步。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from jq_framework.factors.composite import load_factor

ROOT = Path(__file__).resolve().parent.parent


def _fail(msg: str):
    """fail-loud：服务侧任何契约破坏直接退出进程，不静默降级（空仓/旧模型都是危险输出）。"""
    print(f"[ML-SERVING FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def resolve_serving_universe(cfg: dict) -> str:
    """服务 universe：ml_serving.universe 显式优先，缺省回退 multifactor.universe。

    config 的 multifactor.universe 当前是 csi_all（回测宽池/场景覆盖用），生产每日信号
    口径是 csi1000——ml_serving.universe 必须显式钉住，防 config 漂移静默换信号池。
    """
    ms = cfg.get("multifactor", {}).get("ml_serving", {}) or {}
    return ms.get("universe") or cfg["multifactor"]["universe"]


def serving_train_start(cfg: dict) -> str:
    """训练/服务共用的日线拉取起点（缓存键的一部分）——训练器与每日信号必须取同一值。

    ml_serving.train_start 显式钉住；缺省镜像 walk_forward 的扩展窗公式
    （bt.start − 1.6×train_days 日历日）。

    为何信号侧也必须用这个起点（2026-10-29 修）：get_batch_daily 的缓存键是
    stock_{sym}_{adjust}_{start} 逐字拼接，起点不同即不同缓存文件。原实现信号侧
    不传 start（默认 2020-01-01）→ 读 qfq_2020-01-01 缓存，训练/回测读
    qfq_2016-11-24 缓存，两者锚一致但刷新节奏独立：信号侧停在 2026-10-09 而
    生产键已到 2026-10-28，信号凭空陈旧 13 个交易日且无任何告警。起点统一后
    两侧共用同一份文件，刷新一次即同时生效。
    """
    mf = cfg.get("multifactor", {}) or {}
    ms = mf.get("ml_serving", {}) or {}
    ts = ms.get("train_start")
    if ts:
        return str(ts)
    train_days = int((mf.get("walk_forward", {}) or {}).get("train_days", 252) or 252)
    return (pd.Timestamp(mf["backtest"]["start"])
            - pd.Timedelta(days=int(train_days * 1.6))).strftime("%Y-%m-%d")


def _serving_benchmark(cfg: dict, universe: str) -> str:
    """训练池基准跟随（镜像 research/attribution.py L158）：csi1000 同池，其余对宽基。"""
    if universe == "csi1000":
        return universe
    return cfg["multifactor"].get("benchmark", "csi300")


def _factor_spec_list(cfg: dict) -> list[dict]:
    """因子装配契约（有序）：生产 factors + ml.augmented_factors 展开。

    与 walk_forward.py L666-692 逐字同源（含 aug 展开序——ML 特征列序由它决定，
    顺序不同=契约破坏）。只做 spec 展开不实例化、不拉数据，供契约比对复用。
    返回 [{"module","class","params"}]（params 已含 aug 的 feature 键）。
    """
    mf = cfg["multifactor"]
    wf_cfg = mf.get("walk_forward", {})
    weighting = wf_cfg.get("weighting", "static")
    specs = [{"module": spec["module"], "class": spec["class"],
              "params": spec.get("params", {}), "weight": spec.get("weight", 1.0)}
             for spec in mf["factors"].values()]
    ml_cfg = wf_cfg.get("ml", {}) or {}
    aug = ml_cfg.get("augmented_factors") or {}
    if weighting == "ml" and aug:
        aug_specs = aug if isinstance(aug, dict) else {"alpha158": {"module": "factors.alpha158"}}
        for base, spec in aug_specs.items():
            spec = spec if isinstance(spec, dict) else {}
            module = spec.get("module", base)
            from importlib import import_module
            mod = import_module(module)
            feats = getattr(mod, "FEATURE_NAMES", [base] if base else [])
            for fn in feats:
                specs.append({"module": module, "class": spec.get("class", "Alpha158Factor"),
                              "params": {**(spec.get("params", {})), "feature": fn},
                              "weight": spec.get("weight", 1.0)})
    return specs


def _canonical(specs: list[dict]) -> list[str]:
    """装配契约的规范形（params 键序无关；weight 不进契约——ML 特征与权重无关，
    静态权重调整不该判死一个 ML artifact）。用于逐字比对。"""
    return [json.dumps({"module": s["module"], "class": s["class"], "params": s.get("params", {})},
                       sort_keys=True, ensure_ascii=False) for s in specs]


def build_production_factors(cfg: dict, syms: list[str] | None = None,
                             verbose: bool = False, read_only: bool = False) -> list[tuple]:
    """生产因子装配（镜像 walk_forward.py L666-722，train/serve 零偏斜的核心）。

    load_factor → aug 展开（weighting=ml 且配置时）→ 行业中性化 → 基本面注入。
    行业/基本面注入与 weighting 无关（回测两分支同装配）——daily_portfolio 的 static
    路径也走本函数，顺带修复审计#5"信号侧行业中性化/基本面注入缺失"断层。
    返回 [(factor_instance, weight)]，与 walk_forward 的 factors 同构。
    syms: 基本面注入的标的清单（None 时用调用方已加载的价格池键）。
    """
    mf = cfg["multifactor"]
    factors = [(load_factor(spec), spec.get("weight", 1.0))
               for spec in _factor_spec_list(cfg)]

    # 行业中性化注入（镜像 L694-706）
    if mf.get("industry_neutral", False):
        from jq_framework.data.fetchers.fetcher import get_stock_industry_map
        if verbose:
            print("拉取申万一级行业映射（用于行业中性化）...")
        try:
            industry_map = get_stock_industry_map(read_only=read_only)
            for factor, _ in factors:
                factor.industry_neutral = True
                factor.set_industry_map(industry_map)
        except Exception as e:
            if verbose:
                print(f"  行业映射拉取失败，跳过中性化: {e}")

    # 基本面因子数据注入（镜像 L708-722）
    has_fund = any(getattr(f, "is_fundamental", False) for f, _ in factors)
    if has_fund:
        from jq_framework.data.fetchers.fetcher_fundamental import build_fundamental_data
        if verbose:
            print("拉取基本面数据（估值+财务）注入基本面因子 ...")
        try:
            fund_data = build_fundamental_data(syms or [], progress=verbose, read_only=read_only)
            for factor, _ in factors:
                factor.set_fundamental_data(fund_data)
            if verbose:
                ok = sum(1 for v in fund_data.values() if v is not None and not v.empty)
                print(f"  基本面数据: {ok}/{len(syms or [])} 只股票就绪")
        except Exception as e:
            if verbose:
                print(f"  基本面数据拉取失败，基本面因子将返回 None: {e}")
    return factors


# ============ artifact 存取 ============

def _artifact_root(models_dir) -> Path:
    return Path(models_dir) if Path(models_dir).is_absolute() else ROOT / models_dir


def latest_artifact_dir(models_dir: str, horizon: int) -> Path | None:
    """最新 artifact 目录（按 data_end 日期命名的字典序）。无则 None。"""
    root = _artifact_root(models_dir)
    if not root.exists():
        return None
    prefix = f"ml_h{int(horizon)}_"
    cands = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix))
    return cands[-1] if cands else None


def save_model_artifact(model, feature_names: list[str], meta: dict,
                        models_dir: str = "models/ml_synthesis") -> Path:
    """落盘模型 + 训练契约。目录名含 data_end，一天多训互不覆盖（增量重训留痕）。

    num_iteration 截到 best_iteration：与 walk_forward 内存态 booster.predict 的默认
    行为对齐（早停模型只保留最佳轮数内的树），加载侧预测结果与回测侧逐字一致。
    best_iteration=0（未早停）时 num_iteration=None = 保存全部树，同为回测口径。

    多种子集成（P1-2，walk_forward._SeedEnsemble）：成员逐个落盘 model_s<seed>.txt，
    meta 记 seeds 列表；加载侧重组为集成，predict=成员均值（与回测侧逐字同口径）。
    单种子裸 Booster 路径不变（model.txt，历史 artifact 兼容加载）。
    """
    d = _artifact_root(models_dir) / f"ml_h{int(meta['horizon'])}_{pd.Timestamp(meta['data_end']):%Y%m%d}"
    d.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "models"):  # _SeedEnsemble（按 duck-type 判定，避免跨模块硬依赖）
        seeds = list(getattr(model, "seeds", []))
        for s, m in zip(seeds, model.models):
            best = int(getattr(m, "best_iteration", 0) or 0)
            m.save_model(str(d / f"model_s{s}.txt"), num_iteration=best if best > 0 else None)
        meta = {**meta, "ensemble": True, "seeds": seeds}
    else:
        best = int(getattr(model, "best_iteration", 0) or 0)
        model.save_model(str(d / "model.txt"), num_iteration=best if best > 0 else None)
    (d / "meta.json").write_text(
        json.dumps({**meta, "feature_names": list(feature_names),
                    "saved_at": datetime.now().isoformat(timespec="seconds")},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def load_model_artifact(cfg: dict, available_dates=None,
                        models_dir: str | None = None) -> tuple:
    """加载最新 artifact 并做 fail-loud 契约校验。返回 (model, feature_names, meta)。

    cfg: 当前 config（用于因子装配契约比对；horizon/models_dir/max_age_days 读 ml_serving 块）。
    available_dates: 信号日可用的交易日序列（价格数据日期并集）。给定时校验新鲜度：
        距 data_end 的交易日数 > max_age_days → SystemExit（fail-loud，拒出旧信号）。
    """
    import lightgbm as lgb

    ms = cfg.get("multifactor", {}).get("ml_serving", {}) or {}
    models_dir = models_dir or ms.get("models_dir", "models/ml_synthesis")
    horizon = int(ms.get("horizon", 10))

    d = latest_artifact_dir(models_dir, horizon)
    if d is None:
        _fail(f"无 ML artifact（{models_dir} 下无 ml_h{horizon}_* 目录）——"
              f"先跑 python scripts/train_ml_daily.py 训练落盘，weighting=ml 的每日信号"
              f"不接受无模型静默降级。")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("ensemble"):
        # 多种子集成 artifact（P1-2）：重组为 _SeedEnsemble，predict=成员均值。
        from jq_framework.backtest.walk_forward import _SeedEnsemble
        members = [lgb.Booster(model_file=str(d / f"model_s{s}.txt"))
                   for s in meta["seeds"]]
        model = _SeedEnsemble(members, list(meta["seeds"]))
    else:
        model = lgb.Booster(model_file=str(d / "model.txt"))

    # 契约校验 1：因子装配与当前 config 逐字一致（特征列序/口径防错位）
    cur = _canonical(_factor_spec_list(cfg))
    if cur != _canonical(meta.get("factor_specs", [])):
        n_cur, n_old = len(cur), len(meta.get("factor_specs", []))
        _fail(f"因子装配契约已变（当前 {n_cur} 项 vs artifact {n_old} 项）——config 因子集/"
              f"augmented_factors 与模型训练时不一致，旧模型对新特征打分会静默错位。"
              f"请重跑 scripts/train_ml_daily.py；artifact={d.name}")

    # 契约校验 2：新鲜度（距训练数据末的交易日数）
    if available_dates is not None:
        dates = sorted(pd.Timestamp(x) for x in available_dates)
        if dates:
            max_age = int(ms.get("max_age_days", 25))
            data_end = pd.Timestamp(meta["data_end"])
            stale = sum(1 for x in dates if x > data_end)
            if stale > max_age:
                _fail(f"ML 模型陈旧：距训练数据末（{meta['data_end']}）已 {stale} 个交易日，"
                      f"超过红线 {max_age}（ml_serving.max_age_days）。"
                      f"先重训 scripts/train_ml_daily.py 再出信号——fail-loud 防拿旧模型当新信号。")
            # 契约校验 3：行情不得落后于模型（反向陈旧，2026-10-29 修）。
            # 上面的 stale 判的是"模型比行情旧"，但行情比模型旧时 stale 恒为 0——旧缓存
            # 会让报告显示"距训练末 0 个交易日"这种看似最健康的状态，实际在用两周前的
            # 价格打分（当日实况：信号侧读 qfq_2020-01-01 陈旧键，行情停在 10-09 而模型
            # 训到 10-28）。行情末日本应 ≥ 训练末日（训练窗口的末日本身就取自行情），
            # 落后只能说明缓存没刷新。用日历日兜底：无交易日历可数，阈值取 7 天
            # （覆盖周末+单日停跑，而不至于把长假后的首次运行误杀——假期内模型也不会
            # 重训，data_end 同步停在节前）。
            lag_days = (data_end - dates[-1]).days
            max_lag = int(ms.get("max_price_lag_days", 7))
            if lag_days > max_lag:
                _fail(f"行情落后于模型：行情末 {dates[-1].date()} 比模型训练末日 "
                      f"{data_end.date()} 早 {lag_days} 个日历日（红线 {max_lag} 天）。"
                      f"通常是读到了未刷新的缓存键（缓存键含拉取起点，起点不一致即不同文件）。"
                      f"先刷新行情缓存（python scripts/train_ml_daily.py）再出信号——"
                      f"拿陈旧价格打分等于凭空延迟交易。")
            # 契约校验 4：整条链路（行情+模型）都不得落后于日历。上面两条判的都是
            # "行情与模型彼此是否对齐"，两者一起陈旧时同时失效（cache-only 运行或
            # 刷新整体失败）。阈值 20 个日历日：春节/国庆最长休市约 10 天，留一倍余量，
            # 只在真正长时间停跑时触发。
            all_lag = (pd.Timestamp.now().normalize() - dates[-1]).days
            if all_lag > int(ms.get("max_stale_calendar_days", 20)):
                _fail(f"行情与模型同时陈旧：最新行情 {dates[-1].date()} 距今 "
                      f"{all_lag} 个日历日。刷新链路大概率整体失效（网络/缓存目录不可写/"
                      f"定时任务未运行），请先排查再出信号。")
    return model, list(meta["feature_names"]), meta


def artifact_stale_days(cfg: dict, available_dates) -> int | None:
    """最新 artifact 距训练数据末的交易日数（无 artifact 返回 None）。供训练器节流判断。"""
    ms = cfg.get("multifactor", {}).get("ml_serving", {}) or {}
    d = latest_artifact_dir(ms.get("models_dir", "models/ml_synthesis"),
                            int(ms.get("horizon", 10)))
    if d is None:
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    data_end = pd.Timestamp(meta["data_end"])
    return sum(1 for x in available_dates if pd.Timestamp(x) > data_end)


def serving_horizon(cfg: dict) -> int:
    """服务 horizon（ml_serving.horizon，缺省 10 = walk_forward 默认 = 用户口径）。"""
    ms = cfg.get("multifactor", {}).get("ml_serving", {}) or {}
    return int(ms.get("horizon", 10))
