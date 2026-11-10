"""第零步清障单测（docs/futures/期货CTA_强化计划.md）：契约键激活 + built 归一化 + 产物路径。

守护重点：
  1. resolve_contract：显式入参 > config futures.contract > "X0" 兜底（激活前该键
     全仓无消费者，口径由 CLI 默认值/硬编码决定；激活后回测与信号两侧同源）
  2. normalize_built_frame：built 自建帧 → 引擎兼容口径（close=adj_close、
     settle 用 adj_return 累计重建、缺列补 close）；无 adj_close 的 X0 兜底帧
     原样返回（两 schema 兼容）；None/空帧原样返回
  3. output_paths：产物文件名带策略名——pipeline 三策略串行互不覆盖（审计 #18）
  4. 第二步口径 fail-loud：built 模式逐品种标注 built / X0-fallback（判据 =
     归一化后帧有无 adj_close），fallback 品种回测日志醒目告警

纯函数、网络无关（引擎 fail-loud 用例 mock 数据源）。
运行：python -m unittest tests.futures.test_contract_config
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

# 私有库路径（研究脚本已移入 junior-quant-alpha）
import sys as _sys
from pathlib import Path as _Path
PRIVATE_PROBES = _Path(__file__).resolve().parents[2] / 'private'
for _d in ('probes', 'experiments'):
    _p = PRIVATE_PROBES / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
        break

from jq_framework.backtest.futures_walk_forward import output_paths
from jq_framework.data.fetchers.fetcher_futures import normalize_built_frame, resolve_contract


class ResolveContractTest(unittest.TestCase):
    """口径解析优先级：显式入参 > config futures.contract > "X0"。"""

    def test_missing_everything_falls_back_x0(self):
        # 无显式入参、无 config → X0（与历史行为一致，零回归）
        self.assertEqual(resolve_contract({}, None), "X0")
        self.assertEqual(resolve_contract(None, None), "X0")

    def test_config_value_used_when_no_explicit(self):
        self.assertEqual(resolve_contract({"futures": {"contract": "built"}}, None), "built")

    def test_explicit_overrides_config(self):
        # CLI --contract 显式入参优先（A/B 对照实验仍可强制指定口径）
        self.assertEqual(resolve_contract({"futures": {"contract": "built"}}, "X0"), "X0")
        self.assertEqual(resolve_contract({"futures": {"contract": "X0"}}, "built"), "built")

    def test_empty_config_value_falls_back_x0(self):
        # config 键存在但为空（""/None）→ X0，不产出口径为 "" 的脏值
        self.assertEqual(resolve_contract({"futures": {"contract": ""}}, None), "X0")
        self.assertEqual(resolve_contract({"futures": {"contract": None}}, None), "X0")
        self.assertEqual(resolve_contract({"futures": {}}, None), "X0")


def _built_frame() -> pd.DataFrame:
    """合成 built 风格帧：adj_close 带换月跳空复权、无 settle/open 等列。"""
    adj = [100.0, 101.0, 90.0, 89.0]
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4),
        "close": [100.0, 101.0, 99.0, 98.0],          # 原始 close 含换月跳空
        "adj_close": adj,
        "adj_return": [0.0] + [adj[i] / adj[i - 1] - 1 for i in range(1, 4)],
        "hold": [10.0, 10.0, 12.0, 12.0],
        "is_roll": [True, False, True, False],
    })


class NormalizeBuiltFrameTest(unittest.TestCase):
    def test_close_takes_adj_and_settle_rebuilt(self):
        out = normalize_built_frame(_built_frame())
        # close ← adj_close（前复权口径，信号吃复权）
        self.assertEqual(out["close"].tolist(), [100.0, 101.0, 90.0, 89.0])
        # settle 缺失 → adj_return 累计重建（隔换月跳空连续，TSMOM 定方向用）
        expected = np.cumprod([1.0 + r for r in
                               [0.0, 101.0 / 100.0 - 1, 90.0 / 101.0 - 1, 89.0 / 90.0 - 1]])
        np.testing.assert_allclose(out["settle"].to_numpy(), expected)

    def test_missing_ohlcv_cols_filled_with_close(self):
        out = normalize_built_frame(_built_frame())
        for c in ("open", "high", "low", "volume"):
            self.assertIn(c, out.columns)
            self.assertEqual(out[c].tolist(), out["close"].tolist())

    def test_real_settle_always_rebuilt_with_roll_jump_removed(self):
        # S4 修正（2026-10-14）：settle 有真实值也必须重建——原始拼接 settle 含换月
        # 跳空（实测 RB 2021-11-16 单日 −9%），"全 0 才重建"的守卫在真实数据上
        # 永不触发。重建后 settle = cumprod(1+adj_return)，隔跳空连续。
        df = _built_frame()
        df["settle"] = [100.0, 101.5, 90.5, 88.5]  # 含换月跳空的原始拼接 settle
        out = normalize_built_frame(df)
        expected = np.cumprod([1.0 + r for r in
                               [0.0, 101.0 / 100.0 - 1, 90.0 / 101.0 - 1, 89.0 / 90.0 - 1]])
        np.testing.assert_allclose(out["settle"].to_numpy(), expected)
        # 重建序列无 inf：源 settle 的 0 穴点不再进 settle 口径
        self.assertTrue(np.isfinite(out["settle"].to_numpy()).all())

    def test_x0_fallback_frame_passthrough(self):
        # 无 adj_close 的 X0 兜底帧（build 无合约可拼时的回退）→ 原样返回
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=3),
            "open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0], "close": [100.0, 101.0, 102.0],
            "volume": [10.0, 10.0, 10.0], "hold": [1.0, 1.0, 1.0],
            "settle": [100.0, 101.0, 102.0],
        })
        out = normalize_built_frame(df)
        self.assertEqual(out["close"].tolist(), [100.0, 101.0, 102.0])
        self.assertNotIn("adj_close", out.columns)

    def test_none_and_empty_passthrough(self):
        self.assertIsNone(normalize_built_frame(None))
        empty = pd.DataFrame()
        self.assertTrue(normalize_built_frame(empty).empty)

    def test_input_not_mutated(self):
        # 归一化内部 copy，不 mutate 缓存里的原帧
        df = _built_frame()
        before = df["close"].tolist()
        normalize_built_frame(df)
        self.assertEqual(df["close"].tolist(), before)


class OutputPathsTest(unittest.TestCase):
    """审计 #18：产物文件名带策略名，多策略串行互不覆盖。"""

    def test_filenames_carry_strategy(self):
        p = output_paths(Path("reports"), "tsmom_carry_cta")
        self.assertEqual(p["nav"].name, "futures_walk_forward_tsmom_carry_cta_nav.csv")
        self.assertEqual(p["turnover"].name,
                         "futures_walk_forward_tsmom_carry_cta_turnover.csv")
        self.assertEqual(p["cost"].name, "futures_walk_forward_tsmom_carry_cta_cost.csv")
        self.assertEqual(p["margin"].name, "futures_walk_forward_tsmom_carry_cta_margin.csv")

    def test_strategies_do_not_collide(self):
        # pipeline 三白名单策略串行跑，全部四类产物互不覆盖
        strats = ("tsmom_carry_cta", "tsmom_cta", "carry_cta")
        names = set()
        for s in strats:
            names.update(p.name for p in output_paths(Path("reports"), s).values())
        self.assertEqual(len(names), 4 * len(strats))


class PipelineBacktestWiringTest(unittest.TestCase):
    """审计 #18 接线：stage_backtest 把 --out-dir 透传子进程、--contract 只在显式时透传。

    拦截 _run 不真跑子进程（回测冒烟由 pipeline 全 stage 运行承担）。
    """

    def _capture_stage_backtest(self, **over) -> list[list[str]]:
        from types import SimpleNamespace
        import scripts.futures_pipeline as fp
        captured: list[list[str]] = []
        orig = fp._run
        fp._run = lambda cmd, cwd: captured.append(cmd)
        self.addCleanup(setattr, fp, "_run", orig)
        args = SimpleNamespace(
            config=Path("config.yaml"), contract=None, weight_method="vol_parity",
            attribution=False, train_days=None, test_days=None, step_days=None,
            out_dir=Path("reports/run_tag_x"))
        for k, v in over.items():
            setattr(args, k, v)
        fp.stage_backtest({}, args)
        return captured

    def test_out_dir_forwarded_all_three_strategies(self):
        captured = self._capture_stage_backtest()
        self.assertEqual(len(captured), 3)  # 三白名单策略各一条命令
        for cmd in captured:
            self.assertIn("--out-dir", cmd)
            self.assertEqual(cmd[cmd.index("--out-dir") + 1], "reports/run_tag_x")
        # 三策略名互异 → 引擎产物文件名带策略名，互不覆盖
        strats = [cmd[cmd.index("--strategy") + 1] for cmd in captured]
        self.assertEqual(len(set(strats)), 3)

    def test_contract_omitted_when_unset(self):
        # --contract 未指定 → 不透传（子进程各自读 config futures.contract）
        captured = self._capture_stage_backtest()
        self.assertTrue(all("--contract" not in cmd for cmd in captured))

    def test_contract_forwarded_when_explicit(self):
        captured = self._capture_stage_backtest(contract="built")
        for cmd in captured:
            self.assertEqual(cmd[cmd.index("--contract") + 1], "built")


class ContractSourceFailLoudTest(unittest.TestCase):
    """第二步口径 fail-loud：引擎逐品种标注 built / X0-fallback，fallback 醒目告警。

    mock build_contract_series：A 返回真 built 帧（含 adj_close/adj_return/is_roll）、
    B 返回 X0 兜底帧（build 无合约可拼时的回退形态，无 adj_close）——判据与
    run_futures_portfolio / daily_runner 同一根：归一化后有无 adj_close。
    """

    def _run(self, contract: str) -> tuple[dict, str]:
        import jq_framework.backtest.futures_walk_forward as wf
        idx = pd.date_range("2025-01-05", periods=120, freq="B")
        n = len(idx)
        drift = 0.01 * np.arange(n)
        built_A = pd.DataFrame({
            "date": idx, "close": 100.0 + drift,
            "adj_close": 100.0 + drift,
            "adj_return": [0.0] + [0.01] * (n - 1),
            "hold": 1000.0, "is_roll": [False] * n,
        })
        x0_B = pd.DataFrame({
            "date": idx, "open": 200.0 + drift, "high": 200.0 + drift,
            "low": 200.0 + drift, "close": 200.0 + drift,
            "settle": 200.0 + drift, "volume": 1000.0, "hold": 1000.0,
        })

        class _FakeStrat:
            factor_legs: dict = {}
            def on_bar(self, bars):
                from jq_framework.strategies.base import Signal
                return Signal(direction="long", price=float(bars["close"].iloc[-1]))

        orig = (wf.get_futures_main_continuous, wf.get_roll_yield_series,
                wf.get_futures_rules, wf.build_contract_series, wf.build_strategy)
        wf.get_futures_main_continuous = lambda v, contract="X0", read_only=False: (
            built_A if v == "A" else x0_B).copy()
        wf.get_roll_yield_series = lambda v, read_only=False: pd.DataFrame()
        wf.get_futures_rules = lambda read_only=False: {}
        wf.build_contract_series = lambda v, read_only=False: (
            built_A if v == "A" else x0_B).copy()
        wf.build_strategy = lambda sc: (_FakeStrat(), "fake")
        try:
            cfg = {"strategies": {"tsmom_cta": {"module": "jq_alpha.strategies.cta.tsmom",
                                                 "class": "TsmomStrategy", "params": {}}},
                   "futures": {"symbols": ["A0", "B0"], "rebalance_days": 1,
                               "margin": 0.1, "cost": {}, "risk": {}}}
            buf = io.StringIO()
            with redirect_stdout(buf):
                res = wf.run_futures_portfolio(cfg, "tsmom_cta", contract=contract,
                                               weight_method="equal", verbose=False)
            return res, buf.getvalue()
        finally:
            (wf.get_futures_main_continuous, wf.get_roll_yield_series,
             wf.get_futures_rules, wf.build_contract_series, wf.build_strategy) = orig

    def test_built_mode_marks_fallback_and_warns(self):
        res, log = self._run("built")
        self.assertEqual(res["contract_source"], {"A": "built", "B": "X0-fallback"})
        # fallback 品种必须点名告警（含品种与"无换月掩码剔除"后果）
        self.assertIn("[WARN]", log)
        self.assertIn("B", log)
        self.assertIn("换月跳空", log)

    def test_x0_mode_all_plain_no_warn(self):
        res, log = self._run("X0")
        self.assertEqual(res["contract_source"], {"A": "X0", "B": "X0"})
        self.assertNotIn("[WARN]", log)

    def test_built_true_series_marked_built(self):
        # 判据本身：归一化后有 adj_close → "built"（与 normalize_built_frame 兜底
        # 直通的行为对齐——X0 帧原样返回故无 adj_close）
        out = normalize_built_frame(pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=2),
            "close": [100.0, 101.0], "adj_close": [100.0, 101.0],
            "adj_return": [0.0, 0.01], "hold": [1.0, 1.0]}))
        self.assertIn("adj_close", out.columns)


if __name__ == "__main__":
    unittest.main()
