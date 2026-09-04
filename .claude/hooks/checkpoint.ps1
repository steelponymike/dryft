# Auto-checkpoint hook (Stop + SessionEnd).
# Commits all working-tree changes, but ONLY on claude/* or worktree-* session
# branches - never on main/master, never on detached HEAD. Failure is always
# silent (exit 0): a broken checkpoint must not break the session.
# NOTE: keep this file pure ASCII - PowerShell 5.1 misparses non-ASCII in BOM-less files.
$ErrorActionPreference = 'SilentlyContinue'
if ($env:CLAUDE_PROJECT_DIR) { Set-Location $env:CLAUDE_PROJECT_DIR }
$branch = git rev-parse --abbrev-ref HEAD
if ($LASTEXITCODE -ne 0 -or -not $branch) { exit 0 }
if ($branch -notlike 'claude/*' -and $branch -notlike 'worktree-*') { exit 0 }
git add -A | Out-Null
git diff-index --quiet HEAD --
if ($LASTEXITCODE -ne 0) {
    git commit -m "checkpoint: auto-commit ($branch)" | Out-Null
}
exit 0
