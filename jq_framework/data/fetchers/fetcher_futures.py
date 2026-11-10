"""
数据层（期货扩展）：主连对照、展期/期限结构(carry)、保证金规则、多品种批量。

与 data/fetchers/fetcher.py 分离：股票路径(get_stock_daily 等)绝不 import 本模块，
保证 A 股零回归。本模块只服务期货研究线，统一复用 fetcher 里的
_cache_path/_atomic_write_parquet/_load_config/read_only/增量范式。

对外接口（均有 parquet 缓存 + 增量/read_only，与 get_stock_daily 同款）：
- get_futures_contract_daily(symbol)  单真实合约日线（8 列含 hold/settle）
- get_futures_main_continuous(var, contract)  主力连续，X0/X88/X99 口径
- build_contract_series(var, month_pattern=None)  自建连续（按月试探枚举+hold 选主力+跳空调整），作 X0 对照/验证
- get_roll_yield_series(var, start_day)  逐日展期收益率序列（carry 原料）
- get_roll_yield_bar(var, date)  单日全交割月横截面（期限结构斜率）
- get_futures_rules(date=None)  品种规则表 {variety: {margin_ratio, multiplier}}
- get_batch_futures_daily(symbols, ...)  线程池批量，逐失败容错

主连口径说明（不在内部混用）：
- 新浪 X0：主流合约主导连续（项目现用）
- X88：主力平滑 / X99：指数（持仓量加权）
三者口径不同，默认都用 X0，X88/X99 仅作对照；自建 build_contract_series 是独立第三方验证。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 复用 fetcher 的缓存/读写/增量基础设施（私有点仅限此类内部，不动其数据接口语义）
from jq_framework.data.fetchers.fetcher import (
    DataFetchError,
    _atomic_write_parquet,
    _cache_path,
    _load_config,
    _need_full_refresh,
    _need_refresh,
    _normalize_futures,
)
# 共享限流/重试原语：M4 carry 批量回填（2018 起 ≈ 3.4 万次逐日调用）靠它防东财限流
from jq_framework.data.fetchers._ratelimit import Throttle, retry_backoff

# 主连口径：新浪连续合约代码后缀。默认主信号 X0，其余对照。
CONTINUOUS_CONTRACTS = {"X0", "X88", "X99"}

# carry 数据源缺口（实测 2026-09-18）：这 6 个品种东财 get_roll_yield 与新浪逐合约价
# 均无期限结构数据（东财 JSONDecodeError / 新浪只认主连代码无逐合约价）→ **无法算 carry**。
# get_roll_yield_series 对它们必须短路返回空，否则 default start(2018) 起逐日重拉
# 只是反复重试权威性空数据：每品种 ~2000 天 × retry_backoff(≈10s/天) ≈ 数小时/次拉取，
# 且 max_cache_days=1 的旧空缓存过期后每次 walk-forward 都会重拉一遍 → 病理级卡死。
# 只影响 carry 腿；TSMOM/低波等价格类因子不受影响。
NO_CARRY_SOURCE = {"I", "JM", "J", "M", "Y", "P"}


def _fetch_contract_daily_raw(symbol: str, start: str | None = None,
                              end: str | None = None) -> pd.DataFrame:
    """向 akshare 拉取单合约/主连日线（新浪源 futures_zh_daily_sina），返回 8 列 df。

    start/end 为可选截断；新浪源固定返回全部历史，无法按区间增量拉取。
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
        if start:
            df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
        if end:
            df = df[df["date"] <= pd.Timestamp(end)].reset_index(drop=True)
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"拉取期货 {symbol} 失败: {e}") from e
    return df


def _contract_cache(prefix: str, symbol: str):
    """返回 (cfg, path)。key 形如 {prefix}_{symbol}。"""
    cfg = _load_config()
    cache_dir = cfg.get("cache_dir", "data/cache")
    path = _cache_path(cache_dir, f"{prefix}_{symbol}")
    return cfg, path


def get_futures_contract_daily(symbol: str, start: str | None = None,
                               end: str | None = None,
                               read_only: bool = False) -> pd.DataFrame:
    """
    单真实合约（或主连）日线，8 列含 hold/settle。带缓存 + 增量 + read_only。

    symbol: 如 "RB2410"(真实合约) / "RB0"(主力)。start/end 截断；read_only=cache-only。
    返回: DataFrame[date, open, high, low, close, volume, hold, settle]
    """
    cfg, path = _contract_cache("futures", symbol)
    max_days = cfg.get("max_cache_days", 1)
    full_refresh_days = cfg.get("full_refresh_days", 30)

    # read_only：cache-only，未命中回退拉取（唯一写者，原子写防并发）
    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        df = _fetch_contract_daily_raw(symbol, start, end)
        _atomic_write_parquet(path, df)
        return df

    # 1) 快路径
    if not _need_refresh(path, max_days):
        return pd.read_parquet(path)

    # 2) 增量路径：有缓存且未超 full_refresh_days → 只补 last_date+1→end
    if path.exists() and not _need_full_refresh(path, full_refresh_days):
        try:
            cached = pd.read_parquet(path)
            if not cached.empty:
                last_date = pd.Timestamp(cached["date"].iloc[-1]).strftime("%Y-%m-%d")
                if end and last_date >= end:
                    return cached
                inc = _fetch_contract_daily_raw(symbol, last_date, end)
                if inc is not None and not inc.empty:
                    merged = pd.concat([cached, inc], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["date"], keep="last")
                    merged = merged.sort_values("date").reset_index(drop=True)
                    _atomic_write_parquet(path, merged)
                    return merged
                path.touch()
                return cached
        except Exception:
            pass

    # 3) 全量路径
    df = _fetch_contract_daily_raw(symbol, start, end)
    _atomic_write_parquet(path, df)
    return df


def get_futures_main_continuous(variety: str, contract: str = "X0",
                                start: str | None = None,
                                read_only: bool = False) -> pd.DataFrame:
    """
    主力连续日线（8 列含 hold/settle）。contract 指定口径后缀。

    variety: 品种代码，如 "RB" / "AU"（不带连续后缀）。
    contract: "X0"(新浪主导连续，稳健主信号) / "built"(自建换月拼接，build_contract_series)。

    说明：X88(主力平滑)/X99(指数,持仓量加权) 是东财 futures_hist_em 口径，
    与新浪 X0 定义不同；实测东财源在当前环境持续受限流(ConnectionError)，
    故本期不接 X88/X99 对照，避免引入不稳定/口径不一的双源。自建 continuous
    见 build_contract_series（真换月拼接），作为换月专项对照路径。
    返回: DataFrame[date, open, high, low, close, volume, hold, settle]
    """
    if contract == "built":
        return build_contract_series(variety, read_only=read_only)
    if contract != "X0":
        raise ValueError(
            f"主连口径 {contract} 不可用：本期仅支持新浪 X0 或自建 built。"
            f"X88/X99(东财) 属对照，勿用。")
    # 新浪连续后缀：X0 → 0（如 RB0/AU0）。契约记号映射到真实新浪符号。
    sina_suffix = "0"
    return get_futures_contract_daily(f"{variety}{sina_suffix}", start=start,
                                      read_only=read_only)


def resolve_contract(cfg: dict, contract: str | None = None) -> str:
    """主连口径解析（第零步激活 config 键）：显式入参 > config futures.contract > "X0"。

    此前 `futures.contract` 是死键（全仓无消费者），口径实际由各 CLI 默认值与
    硬编码决定；现在回测（run_futures_portfolio）与信号（daily_runner/
    futures_combination_signal）两侧统一经本函数取口径——一个口径一个来源。
    config 缺省/为空回退 "X0"，与历史行为逐字一致（零回归）。
    """
    if contract:
        return contract
    fut = (cfg or {}).get("futures", {}) or {}
    return str(fut.get("contract") or "X0")


def _month_contract_codes(variety: str, years_back: int = 7,
                          month_pattern: list[int] | None = None,
                          forward_years: int = 1) -> list[str]:
    """枚举候选主力合约代码（如 RB2105），覆盖 [今年-years_back, 今年+forward_years]。

    month_pattern 限定要试探的交割月（如 [1,5,9]），None → 全 12 月各试探。
    活跃月非固定模式 ⇒ 逐月试探枚举（CU 1-12 / AU 06,12 是各月集合的自然子集，天然覆盖）。

    forward_years（2026-10-28 修）：**必须向后看**。原实现只从当前年往回推，
    导致下一年度的合约不在候选池——而 1 月合约常在当年 10~12 月就成为主力
    （RB2701 在 2026-10 已是主力，hold 1,838,109 vs 系统错选的 RB2611 33,873）。
    后果：built 选到临近交割的次要合约 + 数据末端被截断。This bug repeats
    every year when the front contract rolls into the next calendar year.

    历史上该参数缺失还造成 2020 年前段污染：近 5 年候选池只有远月合约
    （2020 年选出 RB2101/1 月合约，hold 28,629 vs 真主力 RB2005 的 1,396,015），
    close 侥幸相关但 volume/hold 差 10~70 倍。2026-10-28 二次修正再补一年
    （5→6→7）：built 序列从 2019-01 起，6 年池覆盖到 2020，2019 整段仍被污染
    （CU/ZN/SC hold 中位只有基准的 0.01~0.04x）。

    请求量：now 前后共 (years_back+forward_years+1) 年 × 12 月，默认 8×12=96 次/品种。
    """
    now = datetime.now()
    months = month_pattern or list(range(1, 13))
    codes: list[str] = []
    for year in range(now.year + forward_years,
                     now.year - years_back - 1, -1):
        for m in months:
            codes.append(f"{variety}{str(year)[-2:]}{m:02d}")
    return codes


def _build_contract_series_allc(allc: pd.DataFrame) -> pd.DataFrame:
    """自建连续的纯核心：逐日选主力 → 换月跳空调整 → adj_close/adj_return。

    网络无关、可单测。入参 allc 为多合约日线拼接帧（含 date/close/hold/contract/... 列）。
    产出 DataFrame[date, close, adj_close, adj_return, hold, settle, main_contract, is_roll]
    （列缺失时该列自动剔除，兼容 X0 兜底帧）。

    语义（与新浪 X0 对照/验证，绝不改行为）：
      - 选主力：逐日跨合约取 hold 最大者；同日多合约保留持量最大一行
      - is_roll：连续两日 main_contract 变化处（首日为 True）
      - adj_close：换月日按两合约在前一换月点重叠收盘比 rebase，前复权
      - adj_return：在 adj_close 收益上直接拼接（消除换月日单日伪跳空）
    """
    hold_ok = pd.to_numeric(allc["hold"], errors="coerce").fillna(0)
    allc = allc.assign(_hold=hold_ok)
    best = (allc.sort_values("_hold", ascending=False)
                .drop_duplicates("date", keep="first")
                .sort_values("date").reset_index(drop=True))

    # volume 透传（OI/量确认突破探针，2026-10-26）：主力合约自身成交量，
    # **不做跨合约拼接调整**——换月日 volume 天然跳变（新合约自己的量），
    # 消费方（探针/策略）自行绕开 is_roll 日或用相对量（量比）口径。
    # hold 同语义（也是主力自身值），两列口径一致。
    has_volume = "volume" in best.columns and pd.to_numeric(
        best["volume"], errors="coerce").notna().any()
    vol_col = (pd.to_numeric(best["volume"], errors="coerce").fillna(0.0)
               if has_volume else None)

    best["main_contract"] = best["contract"]
    best["is_roll"] = best["contract"] != best["contract"].shift(1)

    adj_factor = 1.0
    adj_close = []
    old_contract = None
    for i, row in enumerate(best.itertuples(index=False)):
        code = row.contract
        if i == 0:
            adj_close.append(float(row.close))
            old_contract = code
            continue
        if code != old_contract:
            prev_date = best["date"].iloc[i - 1]
            prev_c = allc[(allc["date"] == prev_date) & (allc["contract"] == old_contract)]["close"]
            new_prev_c = allc[(allc["date"] == prev_date) & (allc["contract"] == code)]["close"]
            if not prev_c.empty and not new_prev_c.empty and float(new_prev_c.iloc[0]) > 0:
                adj_factor *= float(prev_c.iloc[0]) / float(new_prev_c.iloc[0])
            old_contract = code
        adj_close.append(float(row.close) * adj_factor)
    best["adj_close"] = adj_close

    best["adj_return"] = best["adj_close"].pct_change().fillna(0.0)

    keep = ["date", "close", "adj_close", "adj_return", "hold", "settle",
            "main_contract", "is_roll"]
    if vol_col is not None:
        best = best.assign(volume=vol_col)
        keep.insert(keep.index("hold") + 1 if "hold" in keep else len(keep), "volume")
    return best[[c for c in keep if c in best.columns]].reset_index(drop=True)


def build_contract_series(variety: str, read_only: bool = False,
                          month_pattern: list[int] | None = None,
                          years_back: int = 7,
                          start: str | None = None,
                          end: str | None = None) -> pd.DataFrame:
    """
    自建主力连续（真换月拼接）：逐日按持仓量 hold 最大选主力合约 → 换月跳空调整。

    对照/验证（--contract built）用，与新浪 X0 主连并存不替换。换月专项实现：
      - **逐月试探枚举**：候选合约覆盖 [今年-years_back, 今年+1]（month_pattern 可收敛）。
        years_back=7（2026-10-28 二次修正，原 5→6）：
        * 原 5：样本起点 2020-01 时，近 5 年候选池只到 2021 年，2020 年段被迫用
          远月合约（RB 选到 RB2101，持仓 255k vs 真主力 RB2010 的 1,433k），
          close 侥幸相关但 volume/hold 差 5~70 倍。
        * 中间态 6：补上了 2020，但 built 序列实际从 2019-01 起，**2019 整段仍在
          用 2020 年远月填充**（CU hold 中位 6,054 = 基准的 0.04x、ZN 0.02x、
          SC 0.01x，9/17 品种受影响）。修了尾和中段，漏了头。
        * 7 才覆盖到真实起点（2019 段 hold 中位回到 1.0~2.2x 基准）。
        影响面：2021-01 之后 17/17 品种 adj_return 逐位一致、hold 完全一致
        （只补 2019 段，不动评估窗口）。
        ⚠️ 结构性隐患：这个常数依赖"数据源回溯多久"与"样本起点"两个外部量，
        换数据源或延长样本时须重新核对首年 hold 是否达标（§10.4.7 教训一）。
      - **选主力**：逐日跨合约取 hold 最大者；换月日 = 新合约 hold 首次反超当日
      - **跳空调整**：换月日按两合约重叠价差 rebase，产出双份——
        `adj_close`（价格前复权，历史位移、观感真实）与 `adj_return`（直接在收益上
        拼接，避免换月日单日伪跳空）。**信号用 adj_return 版**（TSMOM 吃 pct_change）。
      - 返回帧带 `main_contract` / `is_roll` 列，供归因剔除换月日伪 PnL。

    返回: DataFrame[date, close, adj_close, adj_return, hold, settle, main_contract, is_roll]
    """
    # 1) 逐月枚举候选主力合约并逐合约拉取（单失败容错，防 2018 前源缺口中断）
    codes = _month_contract_codes(variety, years_back=years_back,
                                  month_pattern=month_pattern)
    frames: list[pd.DataFrame] = []
    for code in codes:
        try:
            df = get_futures_contract_daily(code, read_only=read_only)
            if df is not None and not df.empty and "hold" in df.columns:
                frames.append(df.assign(contract=code))
        except Exception:
            continue  # 单个合约缺失/源缺口跳过
    if not frames:
        # 第二步口径 fail-loud：无合约可拼 → 回退 X0 原始主连。静默兜底会让"built
        # 口径"在该品种悄悄退回含换月跳空的脏口径（强化计划第二步要防的静默降级），
        # 此处必须告警；回测/每日信号侧还会按 adj_close 缺失做逐品种口径标注。
        print(f"[build_contract_series] ⚠️ {variety} 无真实合约可拼，回退 X0 原始主连"
              f"（含换月跳空，非复权口径）", flush=True)
        return get_futures_main_continuous(variety, contract="X0", read_only=read_only)

    allc = pd.concat(frames, ignore_index=True)
    allc["date"] = pd.to_datetime(allc["date"])
    if start:
        allc = allc[allc["date"] >= pd.Timestamp(start)]
    if end:
        allc = allc[allc["date"] <= pd.Timestamp(end)]
    if allc.empty:
        print(f"[build_contract_series] ⚠️ {variety} 合约数据在截断区间内为空，"
              f"回退 X0 原始主连（含换月跳空，非复权口径）", flush=True)
        return get_futures_main_continuous(variety, contract="X0", read_only=read_only)

    # 2)-5) 纯核心（选主力/换月复权/adj_return），抽成网络无关纯函数便于单测
    out = _build_contract_series_allc(allc)

    # 6) 对照验证：与新浪 X0 主连对齐（交易日轴重合率 + close 相关，corr<0.99 告警）
    try:
        x0 = get_futures_main_continuous(variety, contract="X0", read_only=read_only)
        if x0 is not None and not x0.empty:
            st = _x0_vs_built_stats(out, x0)
            if st["n"] > 30 and not (st["corr"] >= 0.99 and st["overlap"] >= 0.95):
                print(f"[build_contract_series] {variety} 对照告警: "
                      f"与X0 corr={st['corr']:.4f} overlap={st['overlap']:.2%}（<0.99/95%）")
    except Exception:
        pass
    return out


def normalize_built_frame(df: pd.DataFrame) -> pd.DataFrame:
    """built 自建帧 → 引擎/信号兼容口径（纯函数，网络无关，可单测）。

    从 backtest/futures_walk_forward 的 built 分支抽出（第零步：回测侧与信号侧
    daily_runner 用同一函数归一化，一个口径一个来源）：
      - close ← adj_close（前复权价，信号吃复权口径）
      - settle 缺失/全空/全 0 时用 adj_return 累计重建（TSMOM 用 settle 定方向，
        须隔换月跳空连续，见 S4 承诺）
      - open/high/low/volume 缺列用 close 补（引擎只用 close/settle，占位即可）
    无 adj_close 的 X0 兜底帧（build_contract_series 无合约可拼时的回退）原样
    返回——两 schema 都兼容。None/空帧原样返回，调用方判空。
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if "adj_close" not in df.columns:
        return df
    df["close"] = df["adj_close"]
    # S4 兑现（2026-10-14 修正）：settle 一律按 adj_return 累计重建。原实现"settle
    # 缺失/全 0 才重建"的守卫在真实数据上永不触发（拼接 settle 恒有真值，仅零星
    # 源穴点为 0 → inf 收益），TSMOM(settle 优先)吃到未复权含换月跳空的原始拼接
    # settle——实测 RB 2021-11-16 换月日 settle 单日 −9%。连续性 > 结算价基差：
    # 定方向只需隔跳空的价水平（与 walk_forward built 分支注释的既定语义一致）。
    if "adj_return" in df.columns:
        df["settle"] = (1.0 + df["adj_return"]).cumprod()
    for c in ("open", "high", "low", "volume"):
        # volume 例外（2026-10-26）：build_contract_series 已透传真 volume（OI/量
        # 确认探针用），此处只在**缺列**时占位；有真值列不得覆盖（close 占位会
        # 污染量比口径）。既有调用方（引擎/信号）只用 close/settle，行为不变。
        if c == "volume" and "volume" in df.columns:
            continue
        if c not in df.columns:
            df[c] = df["close"]
    return df


def _x0_vs_built_stats(built_df: pd.DataFrame, x0_df: pd.DataFrame) -> dict:
    """built(自建换月) 与 X0(新浪主连) 两套独立数据的重合/相关统计。纯函数、网络无关、可单测。

    X0 与 built 是同一品种的**两路独立数据源**，互相印证数据基建正确性：
      - corr：共同窗口上 close 的 Pearson 相关（量纲一致可比）
      - overlap：共同日期窗内两口径交易日轴重合率（交 / 并，隔离覆盖期长短差异）
      - roll_days / roll_day_ret_consistency：built 换月日是否也落在 X0（换月点一致），
        换月日两口径收益一致性（built adj_return vs X0 同列收益符号/幅度）

    判据：corr≥0.99 且 overlap≥95% 视为数据基建正常；低于即告警（历史上 I/TA 的
    0.97/0.98 属已知对照差异——X0 主连在个别品种的复权基准与自建有系统差）。
    """
    out = {"corr": None, "overlap": None, "roll_days": 0, "n": 0,
           "roll_day_ret_consistency": None}
    if built_df is None or built_df.empty or x0_df is None or x0_df.empty:
        return out
    # 对齐到共同日期窗（max起/min止），隔离交易轴契合度与日期覆盖差异：
    #   corr/axis overlap 只评共同窗内两口径是否一致，不被 built/X0 覆盖期长短污染
    b_date, x_date = built_df["date"], x0_df["date"]
    lo, hi = max(b_date.min(), x_date.min()), min(b_date.max(), x_date.max())
    b_w = built_df[(b_date >= lo) & (b_date <= hi)]
    x_w = x0_df[(x_date >= lo) & (x_date <= hi)]
    m = b_w.merge(x_w[["date", "close"]].rename(columns={"close": "x0_close"}),
                  on="date", how="inner").dropna(subset=["close", "x0_close"])
    if m.empty:
        return out
    out["n"] = len(m)
    out["corr"] = float(m["close"].corr(m["x0_close"]))
    # 交易轴重合率：共同窗内两口径交易日轴交锋程度（该窗内 min(X0天数,built天数) 基准）
    axis_union = max(len(b_w), len(x_w))
    out["overlap"] = len(m) / axis_union if axis_union else 0.0
    if "is_roll" in built_df.columns:
        roll_mask = built_df["is_roll"].fillna(False)
        out["roll_days"] = int(roll_mask.sum())
        if "adj_return" in built_df.columns and out["roll_days"]:
            # 换月日收益一致性：built 换月日 adj_return 与 X0 同日 close 日收益符号一致率。
            # 主判据是 corr/overlap；此为换月点两口径对齐的旁证（报告非断言）。
            x0_daily = m["x0_close"].pct_change()            # 全轴 X0 日收益
            info = pd.DataFrame({"date": m["date"], "x0_ret": x0_daily})
            rd = built_df.loc[roll_mask, "date"]
            rr = info[info["date"].isin(rd.values)].merge(
                built_df.loc[roll_mask, ["date", "adj_return"]], on="date", how="inner")
            rr = rr.dropna(subset=["x0_ret", "adj_return"])
            if len(rr) > 1:
                a = rr["x0_ret"].to_numpy()
                b = rr["adj_return"].to_numpy()
                if len(a) == len(b):
                    out["roll_day_ret_consistency"] = float(
                        pd.Series((a > 0) == (b > 0)).mean())
    return out


def _trading_days(var: str, start_day: str, end_day: str | None) -> list[str]:
    """返回品种 var 在 [start_day, end_day] 间的交易日列表（YYYYMMDD，升序）。

    用 main continuous 的日期列做交易日历（新浪源已是交易日序列），避免逐日探测。
    """
    df = get_futures_main_continuous(var, contract="X0")
    if df is None or df.empty:
        return []
    mask = df["date"] >= pd.Timestamp(start_day)
    if end_day:
        mask &= df["date"] <= pd.Timestamp(end_day)
    days = df.loc[mask, "date"]
    return [d.strftime("%Y%m%d") for d in days]


def get_roll_yield_series(var: str, start_day: str | None = None,
                          end_day: str | None = None,
                          read_only: bool = False) -> pd.DataFrame:
    """
    逐交易日展期收益率序列（carry 原料，KMPV 2018 的商品 carry ≈ 展期收益率）。

    底层逐日 get_roll_yield(date, var)（已验证返回 (roll_yield, near_by, deferred)）。
    roll_yield 为年化展期收益率：负=contango(近月相对远月贴水/升水结构)，
    正=backwardation(近月升水)。carry 定向：sign(roll_yield) 即方向信号。

    返回: DataFrame[date, roll_yield, near_by, deferred]，date 为日期，升序。
    read_only: True = cache-only（命中即读、不拉网不写）。
    start_day: 起始日（YYYYMMDD）。None 时默认用 config futures.start（M4：曾写死最近 3 年，
    导致 2018-2023 前段 carry_map 空、tsmom_carry 退化为纯 TSMOM；现改读样本起点，缺省回退 3 年）。
    """
    cfg, path = _contract_cache("carry", var)

    # M4 源缺口短路：无 carry 源的品种（NO_CARRY_SOURCE）直接返回空，不逐日重拉。
    # 缺失缓存/缓存过期都会被 _need_refresh 判 True 触发 2018 起全量重拉——对无源品种
    # 只是重复试错权威性空数据（每品种数小时），必须在此短路。远端为空与本地空等价。
    if var in NO_CARRY_SOURCE:
        return pd.DataFrame(columns=["date", "roll_yield", "near_by", "deferred"])

    max_days = cfg.get("max_cache_days", 1)
    full_refresh_days = cfg.get("full_refresh_days", 30)

    if start_day is None:
        start_day = _carry_start_default(cfg)

    # read_only：cache-only
    if read_only:
        if path.exists():
            return pd.read_parquet(path)
        fallback = _fetch_roll_yield_series(var, start_day, end_day)
        _atomic_write_parquet(path, fallback)
        return fallback

    # 快路径
    if not _need_refresh(path, max_days):
        return pd.read_parquet(path)

    # 全量路径（carry 是逐日调用，量小，直接全量重拉时间序列最稳，不做增量）
    df = _fetch_roll_yield_series(var, start_day, end_day)
    _atomic_write_parquet(path, df)
    return df


def _carry_start_default(cfg: dict) -> str:
    """carry 默认起点：优先 config futures.start（M4 补齐，样本起点一致）；缺省回退最近 3 年
    （向后兼容——无 futures.start 配置时行为不变）。

    cfg 是 data 段（_contract_cache 返回），此处单独读完整 config 的 futures.start。
    """
    try:
        import yaml
        root = Path(__file__).resolve().parents[3]
        cfg_path = root / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        fut_start = (full.get("futures", {}) or {}).get("start")
        if fut_start:
            return pd.Timestamp(fut_start).strftime("%Y%m%d")
    except Exception:
        pass  # 读配置失败 → 回退 3 年，不因配置异常阻断
    return (datetime.now() - timedelta(days=3 * 365)).strftime("%Y%m%d")


def _fetch_roll_yield_series(var: str, start_day: str | None,
                             end_day: str | None) -> pd.DataFrame:
    """实际逐日拉取展期收益率，返回 DataFrame[date, roll_yield, near_by, deferred]。"""
    try:
        import akshare as ak
    except ImportError as e:
        raise DataFetchError("未安装 akshare，请 pip install akshare") from e

    if start_day is None:
        # 兜底最近 3 年（正常情况下 get_roll_yield_series 已填好 config futures.start）
        start_day = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y%m%d")
    days = _trading_days(var, start_day, end_day)
    rows = []
    # M4 批量回填（2018 起 ≈ 3.4 万次逐日调用）须节流 + 退避，防东财 JSONDecodeError/IP 限流
    th = Throttle(min_interval=0.1)
    for d in days:
        try:
            with th:
                ry = retry_backoff(
                    lambda: ak.get_roll_yield(date=d, var=var),
                    # 东财瞬时限流/JSON 解析失败属暂时性，值得退避重试；真·结构化空直接跳过
                    is_retryable=lambda e: True,
                    retries=3, base=2.0,
                )
            if not ry or not isinstance(ry, (tuple, list)) or len(ry) < 3:
                continue
            rows.append({"date": d, "roll_yield": float(ry[0]),
                         "near_by": str(ry[1]), "deferred": str(ry[2])})
        except Exception:
            continue  # 单日失败跳过（源偶发），不全盘中断
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "roll_yield", "near_by", "deferred"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_roll_yield_bar(var: str, date: str | None = None,
                       read_only: bool = False) -> pd.DataFrame:
    """
    单日全交割月展期收益率（期限结构横截面），作 carry 斜率/期限结构补充观察。

    底层 get_roll_yield_bar(type_method='date') 东财源常被限流（JSONDecodeError），
    故本函数以逐日 get_roll_yield_bar_series 的兼容层实现：返回单日 {date, roll_yield}，
    若源不可用返回空（调用方需容错，carry 主信号仍用 get_roll_yield_series）。
    """
    # 东财源不稳定：封装为 series 再取单日，缓存单日结果
    cfg, path = _contract_cache("rybar", f"{var}_{date or ''}")
    if read_only and path.exists():
        return pd.read_parquet(path)
    try:
        import akshare as ak
        row = ak.get_roll_yield(date=date or datetime.now().strftime("%Y%m%d"), var=var)
        df = pd.DataFrame([{"date": date or "", "roll_yield": float(row[0])}])
        _atomic_write_parquet(path, df)
        return df
    except Exception as e:
        # 源不可用返回空，调用方容错（不抛，carry 主信号在 get_roll_yield_series）
        return pd.DataFrame(columns=["date", "roll_yield"])


def get_futures_rules(date: str | None = None, read_only: bool = False) -> dict[str, dict]:
    """
    期货品种规则表：{variety: {margin_ratio(%), multiplier}}。

    底层 futures_rule(date)（已验证列：代码/交易保证金比例/合约乘数，均为 % 数字）。
    保证金比例用于风控的保证金核算与单品种保证金占用上限。
    缓存 1 天（交易所保证金/乘数很少变，按日刷新足够）。
    """
    cfg, path = _contract_cache("futures_rules", date or "latest")
    max_days = cfg.get("max_cache_days", 1)
    if path.exists() and (read_only or not _need_refresh(path, max_days)):
        _df = pd.read_parquet(path)
        return {row["variety"]: {"margin_ratio": row["margin_ratio"],
                                 "multiplier": row["multiplier"]}
                for row in _df.to_dict(orient="records")}

    try:
        import akshare as ak
        ak_date = date or datetime.now().strftime("%Y%m%d")
        df = ak.futures_rule(date=ak_date)
        if df is None or df.empty:
            raise DataFetchError(f"futures_rule 返回空: {ak_date}")
        rules = {}
        for _, r in df.iterrows():
            var = str(r["代码"]).strip()
            # 只取非期权（期权 代码 带 _O 或 品种名含期权，跳过）
            if "_O" in var or "期权" in str(r.get("品种", "")):
                continue
            try:
                margin = float(r["交易保证金比例"])
            except (TypeError, ValueError):
                continue
            try:
                mult = float(r["合约乘数"])
            except (TypeError, ValueError):
                mult = None
            rules[var] = {"margin_ratio": margin, "multiplier": mult}
        out = pd.DataFrame([{"variety": v, "margin_ratio": m["margin_ratio"],
                             "multiplier": m["multiplier"]}
                            for v, m in rules.items()])
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"获取期货规则表失败: {e}") from e

    _atomic_write_parquet(path, out)
    return {row["variety"]: {"margin_ratio": row["margin_ratio"],
                             "multiplier": row["multiplier"]}
            for row in out.to_dict(orient="records")}


def get_batch_futures_daily(symbols: list[str], progress: bool = True,
                            read_only: bool = False) -> dict[str, pd.DataFrame]:
    """
    批量拉取多品种期货日线，返回 {symbol: DataFrame}（8 列含 hold/settle）。

    线程池并发（受 fetch_workers 限制，防新浪源限流），对失败单品种容错（返回空 df）。
    透传 start/end 可选参数，主连符号如 "RB0"/"AU0" 或真实合约均可。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg = _load_config()
    workers = max(1, int(cfg.get("fetch_workers", 8)))
    out = {}
    n = len(symbols)
    done = 0

    def _one(sym: str) -> tuple[str, pd.DataFrame]:
        try:
            return sym, get_futures_daily(sym, read_only=read_only)
        except Exception as e:
            if progress:
                print(f"  {sym} 失败: {str(e)[:60]}")
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            out[sym] = df
            done += 1
            if progress and (done % 10 == 0 or done == n):
                print(f"  [{done}/{n}] 已处理...")
    # 键序确定性（2026-10-13，已知口径偏差 §10）：返回按输入 symbols 序重排
    # （与 fetcher.get_batch_daily 同修，切断完成序随机键序）。
    return {sym: out[sym] for sym in symbols}


# 为了使批量拉主连时能走到缓存/增量（而非直接调网络），这里 import 主模块的
# get_futures_daily（带缓存/增量/read_only 的官方入口），保证双路径一致。
# 放在底部 import 避免循环依赖（fetcher 不 import 本模块）。
from jq_framework.data.fetchers.fetcher import get_futures_daily  # noqa: E402


if __name__ == "__main__":
    print("=== fetcher_futures 自测 ===")
    print("1) 单真实合约 RB2410 8 列 ...")
    d = get_futures_contract_daily("RB2410", read_only=True)
    print(f"   cols={d.columns.tolist()} 行数={len(d)} 最新={d['date'].iloc[-1].date()}")
    print("2) 主连 RB0 (X0) ...")
    m = get_futures_main_continuous("RB", contract="X0", read_only=True)
    print(f"   cols={m.columns.tolist()} 行数={len(m)} 最新={m['date'].iloc[-1].date()}")
    print("3) carry 序列 RB ...")
    c = get_roll_yield_series("RB", start_day="20240901", read_only=True)
    print(f"   cols={c.columns.tolist()} 行数={len(c)}")
    if not c.empty:
        print(f"   最近3日 roll_yield: {c['roll_yield'].tail(3).tolist()}")
    print("4) 规则表 ...")
    try:
        r = get_futures_rules(read_only=True)
        rb = r.get("RB", {})
        print(f"   品种数={len(r)}  RB margin={rb.get('margin_ratio')}%  mult={rb.get('multiplier')}")
    except Exception as e:
        print(f"   规则表失败(可接受): {str(e)[:120]}")
