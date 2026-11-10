"""ic_ff_smoke.py — 资金流浅史(最近~120交易日)横截面IC检验(v2,修类型)

用手头已拉到的 csi1000 资金流(522只×120行)验证
"主力净流入占比(main_pct)对横截面未来收益有无预测力"。
有则值得上tushare深史;无则省事。

用字符串日期统一,按每票自身date序列位置索引,避免类型不匹配。
"""
from __future__ import annotations
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FF_DIR = "fundflow_offline/data_fundflow"

def load_panel() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(glob.glob(f"{FF_DIR}/ff_*.parquet")):
        df = pd.read_parquet(f)
        if df is None or df.empty:
            continue
        df["date"] = df["date"].astype(str)
        df = df.sort_values("date").reset_index(drop=True)
        out[f.split('/')[-1][3:-8]] = df
    return out

def summarize(ics: list[float], label: str, mean_n: int):
    if not ics:
        print(f"{label}: 无有效时点"); return
    ics = np.array(ics)
    n = len(ics)
    mean = ics.mean()
    sd = ics.std(ddof=1) if n > 1 else np.nan
    t = mean / (sd / np.sqrt(n)) if sd else np.nan
    win = (ics > 0).mean()
    print(f"{label}: 时点{n}(均{n:.0f}只) IC均值={mean:+.4f} ICIR={mean/sd if sd else float('nan'):+.3f} "
          f"t={t:+.2f} 胜率={win:.1%} 近3个采样IC={np.round(ics[-3:],3).tolist()}", flush=True)

def main():
    panel = load_panel()
    print(f"读到 {len(panel)} 只资金流", flush=True)
    if not panel:
        print("无数据"); return
    # 采样日: 各票共享日期交集(降序,取偏后期留出h未来)。不依赖某票轴。
    date_sets = [set(df['date']) for df in panel.values() if len(df)]
    common_days = sorted(date_sets[0].intersection(*date_sets[1:]))
    idx_map = {c: {d: i for i, d in enumerate(df['date'])} for c, df in panel.items()}

    for win in [1, 5, 10]:
        for h in [5, 10, 21]:
            ics = []
            mean_n = 0
            # 采样日: 每天一个时点,但要求该日之后还有>=h个交易日(用全局靠后采样)
            for d in common_days:
                # 该采样日需所有票都还能留h未来(用某票位置判断,近似)
                factor_map, ret_map = {}, {}
                for c, df in panel.items():
                    im = idx_map[c]
                    if d not in im:
                        continue
                    pos = im[d]
                    if pos + h >= df.shape[0]:
                        continue
                    win_slice = df['main_pct'].iloc[max(0, pos - win + 1):pos + 1]
                    if len(win_slice) < win or win_slice.isna().all():
                        continue
                    factor_map[c] = win_slice.astype(float).sum()
                    r0 = df['close'].iloc[pos]
                    rh = df['close'].iloc[pos + h]
                    if not (np.isfinite(r0) and np.isfinite(rh)) or r0 <= 0 or rh <= 0:
                        continue
                    ret_map[c] = rh / r0 - 1
                common = [c for c in factor_map if c in ret_map]
                if len(common) < 10:
                    continue
                fv = np.array([factor_map[c] for c in common], dtype=float)
                rv = np.array([ret_map[c] for c in common], dtype=float)
                m = np.isfinite(fv) & np.isfinite(rv)
                if m.sum() < 10:
                    continue
                rho, _ = spearmanr(fv[m], rv[m])
                if not np.isnan(rho):
                    ics.append(rho)
                    mean_n += m.sum()
            summarize(ics, f"main_pct {win}日累计 vs fut {h}日", mean_n)

if __name__ == "__main__":
    main()
