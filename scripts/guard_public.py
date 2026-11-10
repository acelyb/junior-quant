#!/usr/bin/env python3
"""公开仓库的 pre-commit 守卫：阻止私有 alpha 进入公开历史。

安装（在公开仓库根执行）：
    cp scripts/guard_public.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

拦截三类内容：
  1. 路径 —— 私有目录/文件（探针、实验、真实配置、研究文档）
  2. 内容 —— 具体读数（ΔIR / 夏普 / 年化 / 胜率 / IC t 值）
  3. commit message —— 研究结论措辞（否决/采纳/drop/keep）

设计原则：宁可误杀，不可漏放。误杀可以 --no-verify 绕过，
但绕过前请想清楚"这段内容暴露了我的研究结论吗"。
"""
from __future__ import annotations

import re
import subprocess
import sys

# ===== 1. 禁止提交的路径（正则，相对仓库根）=====
BLOCKED_PATHS = [
    (r"^private/", "私有 submodule"),
    (r"^docs/", "研究文档属于私有库"),
    (r"^config\.yaml$", "真实配置属于私有库"),
    (r"^research/.*\.csv$", "健康卡读数属于私有库"),
    (r"^scripts/futures_.*_probe\.py$", "研究探针属于私有库"),
    (r"^scripts/compare_.*\.py$", "对比实验属于私有库"),
    (r"^scripts/plot_.*\.py$", "绘图脚本属于私有库"),
    (r"^scripts/diag_.*\.py$", "诊断脚本属于私有库"),
    (r"^scripts/measure_.*\.py$", "度量实验属于私有库"),
    (r"^jq_framework/factors/(?!base\.py|composite|composite_ml|ic|ic_report|corr|"
     r"feature_engineering|universe_expansion|dynamic_pool|__init__|examples/)",
     "因子实现属于私有库"),
    (r"^jq_framework/strategies/(?!base|adapter|examples|__init__)", "策略实现属于私有库"),
]

# ===== 2. 禁止提交的内容（正则，文件正文）=====
BLOCKED_CONTENT = [
    (r"ΔIR\s*[−\-+]?\s*\d\.\d", "含 ΔIR 具体读数"),
    (r"夏普\s*[−\-+]?\s*0?\.\d", "含夏普具体读数"),
    (r"年化\s*[−\-+]?\s*\d+(\.\d+)?\s*%", "含年化具体读数"),
    (r"胜率\s*\d+(\.\d+)?\s*%", "含胜率具体读数"),
    (r"IC\s*[+\-]?0\.0\d+\s*\(\s*t\s*=", "含 IC 与 t 值"),
    (r"残差夏普\s*[−\-+]?\s*0?\.\d", "含残差检验读数"),
    (r"L1O\s*Δ?IR\s*[−\-+]?\s*\d", "含 L1O 读数"),
    (r"样本外\s*年化\s*[+\-]?\d", "含样本外年化"),
    (r"最大回撤\s*[−\-]?\d+(\.\d+)?\s*%", "含回撤读数"),
    # README 里出现过的裸读数格式（守卫最初漏掉的）
    (r"组合\s*IR\s*从\s*[−\-]?\d", "含组合 IR 读数"),
    (r"\bIR\s*[−\-]\d+\.\d+", "含裸 IR 数值"),
    (r"夏普\s*在\s*\d+(\.\d+)?\s*(上下|左右|附近)", "含夏普区间描述"),
    (r"[−\-]\d\.\d{2,}\s*掉到\s*[−\-]?\d", "含前后对比读数"),
]

# ===== 3. commit message 禁止措辞 =====
BLOCKED_MSG = [
    (r"否决|采纳|drop|keep\b", "含研究裁决措辞"),
    (r"ΔIR|夏普|胜率|年化\s*[+\-]?\d", "含绩效读数"),
    (r"失效|有效|优于|劣于|优于基线", "含比较结论"),
]

# 允许清单：这些文件即使命中内容规则也放行（框架自身含公式说明）
ALLOW_CONTENT = {
    "scripts/guard_public.py",
    "jq_framework/factors/composite.py",
    "jq_framework/factors/ic.py",
    "jq_framework/backtest/walk_forward.py",
}


def _staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def _staged_diff_name_only(path: str) -> str:
    """取暂存版本的内容（不是工作区）。"""
    return subprocess.run(["git", "show", f":{path}"],
                          capture_output=True, text=True).stdout


def main() -> int:
    errors: list[str] = []

    # --- 路径检查 ---
    for path in _staged_files():
        for pat, why in BLOCKED_PATHS:
            if re.search(pat, path):
                errors.append(f"  [路径] {path}\n         {why}")

    # --- 内容检查 ---
    for path in _staged_files():
        if path in ALLOW_CONTENT or not path.endswith((".py", ".md", ".yaml", ".yml")):
            continue
        content = _staged_diff_name_only(path)
        for pat, why in BLOCKED_CONTENT:
            m = re.search(pat, content)
            if m:
                line = content[:m.start()].count("\n") + 1
                errors.append(f"  [内容] {path}:{line}\n         {why} → {m.group(0)[:60]!r}")

    if errors:
        print("\n" + "=" * 70)
        print("  公开仓库守卫：拒绝提交")
        print("=" * 70)
        for e in errors:
            print(e)
        print("\n" + "=" * 70)
        print("  这些内容属于私有 alpha，应提交到 private/ (junior-quant-alpha)")
        print("  确需绕过：git commit --no-verify（请先确认不含研究结论）")
        print("=" * 70 + "\n")
        return 1

    return 0


def check_message(msg_path: str) -> int:
    """commit-msg hook 入口：检查 commit message。

    用法（.git/hooks/commit-msg）：
        exec python3 scripts/guard_public.py --msg-only "$1"
    """
    try:
        with open(msg_path, encoding="utf-8") as f:
            msg = f.read()
    except FileNotFoundError:
        return 0
    # 忽略注释行与空行
    body = "\n".join(l for l in msg.splitlines() if not l.startswith("#"))

    errors = []
    for pat, why in BLOCKED_MSG:
        m = re.search(pat, body)
        if m:
            errors.append(f"  [消息] {why} → {m.group(0)!r}")

    if errors:
        print("\n" + "=" * 70)
        print("  公开仓库守卫：commit message 含研究结论")
        print("=" * 70)
        for e in errors:
            print(e)
        print("\n  研究措辞属于私有库；公开仓库的 message 只描述改了什么")
        print("  确需绕过：git commit --no-verify")
        print("=" * 70 + "\n")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--msg-only":
        sys.exit(check_message(sys.argv[2]))
    sys.exit(main())
