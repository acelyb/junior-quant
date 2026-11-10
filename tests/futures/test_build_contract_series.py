"""期货换月自建连续纯核心单元测试：_build_contract_series_allc。

守护重点（换月是"伪 PnL"最易藏身的工程桩子——主连拼接/缺口复权/is_roll）：
  1. 选主力：逐日取持量 hold 最大合约；同日多合约只留持量最大一行
  2. is_roll：main_contract 变化处标记（首日为 True）
  3. adj_close 前复权：换月日按重叠窗收盘比 rebase，历史价格位移、序列连续
  4. adj_return 消除伪跳空：换月日收益 = 新合约真实涨跌幅，而非 raw close 之比
  5. 换月后回落：复权因子收敛回 1.0（价格回到真实水平）

纯函数网络无关：手造多合约帧，逐字复现 build_contract_series 的选主力/复权逻辑。

运行：python -m unittest tests.futures.test_build_contract_series
"""
from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

from jq_framework.data.fetchers.fetcher_futures import (
    _build_contract_series_allc,
    _month_contract_codes,
    _x0_vs_built_stats,
)


def _overlap_frame() -> pd.DataFrame:
    """重叠窗换月场景（RB2601 到期 → RB2605）：
      - RB2601(老主力): 0102=100, 0105=110, 0106=120，0106 前 hold 大
      - RB2605(新合约): 0106 挂牌=210, 0107=200, 0108=210, 0109=220，0107 起 hold 反超
      - 0106 两合约重叠（都交易）→ 换月 rebase 锚存在
      换月日 0107 前复权 factor=120/210；raw close 从 120 突变 200，但 adj 连续。
    """
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-06",
                                "2026-01-07", "2026-01-08", "2026-01-09"]),
        "contract": ["RB2601", "RB2601", "RB2601", "RB2605",
                     "RB2605", "RB2605", "RB2605"],
        "close": [100.0, 110.0, 120.0, 210.0, 200.0, 210.0, 220.0],
        "hold": [5000, 5000, 5000, 100, 6000, 6000, 6000],
        "settle": [100.0, 110.0, 120.0, 210.0, 200.0, 210.0, 220.0],
    })


class MainContractSelectionTest(unittest.TestCase):
    def test_picks_highest_hold_per_day(self):
        # 0106 两合约都交易 → 选持量大者 RB2601
        out = _build_contract_series_allc(_overlap_frame())
        d6 = out[out["date"] == pd.Timestamp("2026-01-06")]
        self.assertEqual(d6["main_contract"].iloc[0], "RB2601")
        self.assertEqual(d6["close"].iloc[0], 120.0)

    def test_rolls_to_new_contract_when_hold_overtakes(self):
        # 0107 RB2605 hold(6000)>RB2601 → 换月
        out = _build_contract_series_allc(_overlap_frame())
        d7 = out[out["date"] == pd.Timestamp("2026-01-07")]
        self.assertEqual(d7["main_contract"].iloc[0], "RB2605")

    def test_n_rows_equals_trading_days(self):
        # 7 行输入（0106 双合约）→ 去重后 6 个交易日
        out = _build_contract_series_allc(_overlap_frame())
        self.assertEqual(len(out), 6)


class IsRollTest(unittest.TestCase):
    def test_is_roll_at_contract_change_only(self):
        out = _build_contract_series_allc(_overlap_frame())
        # 首日 0102 True；0105/0106 False（仍 RB2601）；0107 换月 True；0108/0109 False
        by_date = dict(zip(out["date"], out["is_roll"]))
        self.assertTrue(by_date[pd.Timestamp("2026-01-02")])
        self.assertFalse(by_date[pd.Timestamp("2026-01-05")])
        self.assertFalse(by_date[pd.Timestamp("2026-01-06")])
        self.assertTrue(by_date[pd.Timestamp("2026-01-07")])
        self.assertFalse(by_date[pd.Timestamp("2026-01-08")])
        self.assertFalse(by_date[pd.Timestamp("2026-01-09")])


class GapAdjustmentTest(unittest.TestCase):
    def test_adj_close_rebased_before_roll(self):
        # 0107 前复权：factor=120/210 → adj_close=200*120/210=114.2857
        out = _build_contract_series_allc(_overlap_frame())
        d7 = out[out["date"] == pd.Timestamp("2026-01-07")]
        self.assertAlmostEqual(d7["adj_close"].iloc[0], 200 * 120 / 210, places=6)

    def test_no_fake_gap_on_roll_day(self):
        # 换月日 adj_return = 新合约真实跌幅 200/210-1（-4.76%），
        # 而非 raw close 之比 200/120-1（+66.7% 伪跳空）。这就是复权点。
        out = _build_contract_series_allc(_overlap_frame())
        d6 = 120.0
        d7 = out[out["date"] == pd.Timestamp("2026-01-07")]
        self.assertAlmostEqual(d7["adj_return"].iloc[0], 200 / 210 - 1, places=6)
        self.assertNotAlmostEqual(d7["adj_return"].iloc[0], 200 / 120 - 1, places=2)

    def test_adj_return_next_day_new_contract_move(self):
        # 0108 adj_return = 210/200-1（新合约自身涨跌，复权因子已收敛）
        out = _build_contract_series_allc(_overlap_frame())
        d8 = out[out["date"] == pd.Timestamp("2026-01-08")]
        self.assertAlmostEqual(d8["adj_return"].iloc[0], 210 / 200 - 1, places=6)

    def test_factor_cumulative_after_roll(self):
        # 复权因子累积累乘、换月后仍作用（前复权位移的绝对水平，非回到 spot）。
        # 交易输入是 adj_return 而非水平（TSMOM 吃 pct_change），恒定因子下收益不变，故水平漂移为观感非 bug。
        # 0109 adj_close = 220 × (120/210) = 125.714（因子 120/210 持续生效）
        out = _build_contract_series_allc(_overlap_frame())
        d9 = out[out["date"] == pd.Timestamp("2026-01-09")]
        self.assertAlmostEqual(d9["adj_close"].iloc[0], 220 * 120 / 210, places=6)


class NoOverlapEdgeTest(unittest.TestCase):
    def test_no_overlap_preserves_close(self):
        # 无重叠窗（新合约换月前一日无报价）→ 不 rebase，adj_close=raw close
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06",
                                    "2026-01-07"]),
            "contract": ["RB2601", "RB2601", "RB2605", "RB2605"],
            "close": [100.0, 110.0, 200.0, 210.0],
            "hold": [5000, 3000, 6000, 6000],
            "settle": [100.0, 110.0, 200.0, 210.0],
        })
        out = _build_contract_series_allc(frame)
        self.assertAlmostEqual(out["adj_close"].iloc[2], 200.0, places=6)


class X0VsBuiltStatsTest(unittest.TestCase):
    """①b 独立数据交叉验证纯函数 _x0_vs_built_stats。

    守护：corr/overlap 只在**共同日期窗**内评（不被 built 覆盖期长于 X0 污染——
    这是 17 品种 overlap 从假 74% 修正到真 ~100% 的回归桩），
    并验证换月点收益一致性旁证可算。
    """
    def _built(self):
        # built 含 6 个交易日，is_roll 标记 0106/0107（两次换月）；
        # 多出 X0 覆盖前的 1 天（0102），验证共同窗隔离覆盖期差异
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06",
                                    "2026-01-07", "2026-01-08"]),
            "close": [100.0, 110.0, 120.0, 200.0, 210.0],
            "adj_return": [0.0, 0.10, 0.09, -0.0476, 0.05],
            "is_roll": [False, False, True, True, False],
        })

    def _x0(self):
        # X0 从 0105 才覆盖（01-02 不在 X0 → 共同窗自动从 0105 起）
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-06",
                                    "2026-01-07", "2026-01-08"]),
            "close": [112.0, 121.0, 198.0, 212.0],
        })

    def test_corr_overlap_on_common_window(self):
        st = _x0_vs_built_stats(self._built(), self._x0())
        # 共同窗 = 0105~0108（4 天），交=4，并=4 → overlap 100%
        self.assertEqual(st["n"], 4)
        self.assertAlmostEqual(st["overlap"], 1.0, places=6)
        # 0106 是换月日（在共同窗内被计入 roll_days）
        self.assertEqual(st["roll_days"], 2)

    def test_leading_coverage_does_not_penalize_overlap(self):
        """built 多出的覆盖前日期(0102)不拉低 overlap——回归假 74% 的桩。"""
        st = _x0_vs_built_stats(self._built(), self._x0())
        self.assertGreaterEqual(st["overlap"], 0.95,
                                "覆盖期差异不应污染共同窗重合率")

    def test_x0_ret_consistency_on_roll_day(self):
        # 两换月日(0106/0107)均在共同窗 → 换月点收益一致性可算
        st = _x0_vs_built_stats(self._built(), self._x0())
        self.assertIsNotNone(st["roll_day_ret_consistency"])
        self.assertLessEqual(st["roll_day_ret_consistency"], 1.0)
        self.assertGreaterEqual(st["roll_day_ret_consistency"], 0.0)

    def test_empty_inputs_return_null(self):
        st = _x0_vs_built_stats(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(st["corr"], None)
        self.assertEqual(st["overlap"], None)


class MonthContractCodesTest(unittest.TestCase):
    """候选合约枚举：向后看一年 + 覆盖样本起点。

    守护两个曾长期潜伏、后果严重的 bug（2026-10-28 修）：
      1. **尾视缺失**：原实现只从当前年往回推，下一年度合约不在候选池。
         而 1 月合约常在当年 10~12 月就成为主力 —— built 会选到临近交割的
         次要合约，且数据末端被截断。RB 实测：应为 RB2701（hold 1,838,109），
         实际选到 RB2611（hold 33,873），差 54 倍，数据停在 2026-10-14 而非 10-27。
         此 bug **每年主力跨年切换时复发**。
      2. **前段覆盖不足**：years_back=5 时样本起点 2020-01 的候选池只有 2021 年
         合约，2020 年段被迫用远月。RB 2020Q2 选到 RB2101（持仓 255k），
         真主力 RB2010 是 1,433k。close 侥幸相关（远月价格联动），
         但 volume/hold 差 5~70 倍，污染所有量能/OI 类因子。
      2b. **同上，修一次没修干净（2026-10-28 二次修正）**：5→6 补上了 2020，
         但序列实际从 2019-01 起，2019 整段仍被 2020 年远月填充
         （CU 0.04x / ZN 0.02x / SC 0.01x，9/17 品种）。6→7 才覆盖到真实起点。
    """

    def test_includes_next_calendar_year(self):
        """尾视：下一年度合约必须在候选池（防跨年主力切换时选错合约）。"""
        now = datetime.now()
        codes = _month_contract_codes("RB", years_back=7)
        nxt = str(now.year + 1)[-2:]
        self.assertTrue(any(c.startswith(f"RB{nxt}") for c in codes),
                        f"下一年度({now.year + 1})合约未进入候选池——"
                        f"主力跨年切换后 built 会选到次要合约并截断数据")

    def test_covers_sample_start_year(self):
        """前段：候选池须覆盖 built 序列的真实起点年（2019），否则该段量能不可信。

        2026-10-28 二次修正：years_back=6 只覆盖到 2020，而序列从 2019-01 起，
        2019 整段仍在用 2020 年远月填充（CU hold 中位 6,054 = 基准 0.04x）。
        所以这里守的是"首个有数据的年份"，不是某个写死的常数——换数据源
        导致起点前移时，本测试会跟着失败。
        """
        import inspect

        from jq_framework.data.fetchers import fetcher_futures
        sig = inspect.signature(fetcher_futures.build_contract_series)
        default_yb = sig.parameters["years_back"].default
        codes = _month_contract_codes("RB", years_back=default_yb)
        self.assertTrue(any(c.startswith("RB19") for c in codes),
                        "候选池未覆盖 2019 年（built 序列起点）——该段会用远月"
                        "合约填充，volume/hold 量级错误（CU/ZN/SC 曾低至 0.01x）")

    def test_default_covers_full_sample(self):
        """默认参数（build_contract_series 不传 years_back）须同时覆盖头尾。"""
        import inspect

        from jq_framework.data.fetchers import fetcher_futures
        sig = inspect.signature(fetcher_futures.build_contract_series)
        default_yb = sig.parameters["years_back"].default
        codes = _month_contract_codes("RB", years_back=default_yb)
        now = datetime.now()
        self.assertTrue(any(c.startswith(f"RB{str(now.year + 1)[-2:]}")
                            for c in codes), "默认参数缺尾视")
        self.assertTrue(any(c.startswith("RB19") for c in codes),
                        "默认参数覆盖不到 2019 年（样本起点）")

    def test_month_pattern_respected(self):
        """month_pattern 收敛交割月，但年覆盖不受影响。"""
        codes = _month_contract_codes("RB", years_back=7, month_pattern=[1, 5, 9])
        self.assertTrue(all(c[-2:] in ("01", "05", "09") for c in codes))
        self.assertTrue(any(c.startswith("RB19") for c in codes))
        nxt = str(datetime.now().year + 1)[-2:]
        self.assertTrue(any(c.startswith(f"RB{nxt}") for c in codes))

    def test_no_duplicate_codes(self):
        codes = _month_contract_codes("RB", years_back=7, month_pattern=[1, 5, 9])
        self.assertEqual(len(codes), len(set(codes)), "候选代码重复会导致重复拉取")


if __name__ == "__main__":
    unittest.main()
