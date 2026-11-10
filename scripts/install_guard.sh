#!/bin/sh
# 安装公开仓库守卫（pre-commit + commit-msg + message 模板）
# 在公开仓库根执行：sh scripts/install_guard.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1) pre-commit：路径 + 内容检查
printf '#!/bin/sh\nexec python3 scripts/guard_public.py\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# 2) commit-msg：message 检查
printf '#!/bin/sh\nexec python3 scripts/guard_public.py --msg-only "$1"\n' > .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg

# 3) message 模板
git config commit.template .gitmessage

echo "✓ 公开仓库守卫已安装"
echo "  .git/hooks/pre-commit   路径 + 内容检查"
echo "  .git/hooks/commit-msg   commit message 检查"
echo "  commit.template         .gitmessage"
echo
echo "  绕过（慎用）：git commit --no-verify"
