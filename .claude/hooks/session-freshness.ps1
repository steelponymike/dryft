# SessionStart hook: proves the checkout is CURRENT before the session writes anything,
# and says out loud when the safety nets are not actually running.
#
# WHY THIS EXISTS. Four incidents, one shape - a session is handed memory/STATUS.md with
# authority while nothing checks whether it is the NEWEST version that exists, then writes
# a whole-file save from it and reverts whoever wrote last:
#   1. 2026-07-29, peace-by-piece. A session built ~630 lines against a 5-commit-stale tree
#      and came close to reverting a live deploy. That repo's check-staleness.sh was the
#      first version of this guard; this hook supersedes it.
#   2. 2026-09-01/02, claude-setup. A session branch was merged into master at 11:34, then
#      a LATER session kept committing to that same already-merged branch. A third session
#      fast-forwarded master, never saw those commits, and rewrote STATUS.md on top. Both
#      sides held work the other lacked.
#   3. 2026-09-02. A save opened on a checkout NINE commits behind origin/master with a
#      two-day-old STATUS.md on disk. Caught by hand, barely.
#   4. spf-ops sat on a DETACHED HEAD for three months. checkpoint.ps1 exits silently on a
#      detached HEAD, so nothing was ever auto-committed - an entire memory/ folder
#      (CHANGELOG.md and SESSION-LOG.md existing nowhere in the repo) lived untracked.
#
# Loud ONLY when there is something to act on; a clean, current checkout prints one line.
# NOTE: keep this file pure ASCII - PowerShell 5.1 misparses non-ASCII in BOM-less files.
# Never fails a session: every path exits 0.

$ErrorActionPreference = 'SilentlyContinue'

try {
    # Resolve the directory the SESSION is actually in, not where it was launched.
    # Per the worktrees doc: "Hook paths don't follow the worktree. ${CLAUDE_PROJECT_DIR}
    # stays put... cwd follows Claude." In a `claude -w <name>` session this hook would
    # otherwise check the MAIN checkout and report it as current while the worktree the
    # session is really in went unchecked - a freshness guard that silently guards the
    # wrong directory is worse than none. The hook's stdin JSON carries the real cwd.
    # Guarded by IsInputRedirected so a hook invoked without stdin can never block startup.
    $root = $null
    try {
        if ([Console]::IsInputRedirected) {
            $raw = [Console]::In.ReadToEnd()
            if ($raw) {
                $j = $raw | ConvertFrom-Json
                if ($j.cwd) { $root = [string]$j.cwd }
            }
        }
    } catch { }
    if (-not $root) { $root = $env:CLAUDE_PROJECT_DIR }
    if (-not $root) { $root = (Get-Location).Path }
    Push-Location $root
    if ((git rev-parse --is-inside-work-tree 2>$null) -ne 'true') { Pop-Location; exit 0 }

    $warn = @()

    # ---- 1. Is the checkpoint safety net running for this session? ----
    $branch = (git rev-parse --abbrev-ref HEAD 2>$null)
    if ($branch -eq 'HEAD') {
        $warn += "HEAD IS DETACHED. checkpoint.ps1 does NOT auto-commit on a detached HEAD,"
        $warn += "so NOTHING written this session is being saved anywhere. Before doing any"
        $warn += "work:  git switch -c claude/<topic>"
    }
    elseif ($branch -notlike 'claude/*' -and $branch -notlike 'worktree-*') {
        $warn += "Branch '$branch' is not a claude/* branch, so the checkpoint hook will NOT"
        $warn += "auto-commit. Anything written must be committed deliberately."
    }

    # ---- 2. memory/ files git has never seen (the spf-ops failure) ----
    $orphans = @(git ls-files --others --exclude-standard -- memory/ 2>$null)
    if ($orphans.Count) {
        $warn += "UNTRACKED memory/ FILES - git has never seen these, so they are one"
        $warn += "'git clean' from gone and invisible to every other machine:"
        foreach ($o in $orphans) { $warn += "    $o" }
    }

    # ---- 3. Is there anywhere off-machine for this work to go at all? ----
    $remotes = @(git remote 2>$null)
    if (-not $remotes.Count) {
        $warn += "THIS REPO HAS NO GIT REMOTE. Every commit exists only on this machine and"
        $warn += "is not backed up anywhere. Do not treat committing as saving."
        Write-Output "[freshness] branch=$branch | NO REMOTE"
        if ($warn.Count) {
            Write-Output "!!! ---------------------------------------------------------------------"
            foreach ($w in $warn) { Write-Output "!!! $w" }
            Write-Output "!!! ---------------------------------------------------------------------"
        }
        Pop-Location; exit 0
    }

    # ---- 4. Is this checkout current? ----
    # Refresh only when the remote view is stale, so session start stays fast.
    $gitDir = (git rev-parse --git-common-dir 2>$null)
    $fresh = $false
    if ($gitDir) {
        $fetchHead = Join-Path $gitDir 'FETCH_HEAD'
        if (Test-Path $fetchHead) {
            if (((Get-Date) - (Get-Item $fetchHead).LastWriteTime).TotalMinutes -lt 30) { $fresh = $true }
        }
    }
    if (-not $fresh) {
        $env:GIT_TERMINAL_PROMPT = '0'
        git fetch origin --prune --quiet 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $fresh = $true }
    }

    # Default branch: origin/HEAD when set, else fall back rather than going silent.
    $def = $null
    if (git rev-parse --verify --quiet refs/remotes/origin/HEAD 2>$null) {
        $def = (git rev-parse --abbrev-ref origin/HEAD 2>$null)
    }
    if (-not $def -or $def -eq 'origin/HEAD') {
        if (git rev-parse --verify --quiet refs/remotes/origin/main 2>$null)        { $def = 'origin/main' }
        elseif (git rev-parse --verify --quiet refs/remotes/origin/master 2>$null)  { $def = 'origin/master' }
    }

    $line = "[freshness] branch=$branch"

    if ($fresh -and $def) {
        $behind = (git rev-list --count "HEAD..$def" 2>$null)
        $ahead  = (git rev-list --count "$def..HEAD" 2>$null)
        $line += " | vs ${def}: behind $behind, ahead $ahead"
        if ([int]$behind -gt 0 -and [int]$ahead -gt 0) { $line += "  -> DIVERGED" }

        # The one that actually costs content: memory files changed on the default branch
        # in commits this checkout cannot see. Writing a whole-file save from what was just
        # injected would revert them. Ahead-only is normal and stays quiet - no false alarms.
        if ([int]$behind -gt 0) {
            $missed = @(git diff --name-only "HEAD..$def" -- memory/ 2>$null)
            if ($missed.Count) {
                $warn += "MEMORY FILES CHANGED ON $def THAT THIS CHECKOUT CANNOT SEE:"
                foreach ($m in $missed) { $warn += "    $m" }
                $warn += "The STATUS.md injected above is NOT the newest version that exists."
                $warn += "Do NOT write any memory file until you reconcile. Union-merge, keep"
                $warn += "BOTH sides, and verify a marker from each survives:"
                $warn += "    git merge $def        # resolve by UNION, never by picking a side"
            } else {
                $warn += "This checkout is $behind commit(s) behind $def (no memory/ files"
                $warn += "affected). Run 'git pull --ff-only' before building or deploying."
            }
        }
    }
    elseif (-not $fresh) {
        $line += " | remote unreachable (offline?) - treat STATUS.md as possibly stale"
    }
    else {
        $line += " | no default branch found on origin - cannot check freshness"
    }

    Write-Output $line
    if ($warn.Count) {
        Write-Output "!!! ---------------------------------------------------------------------"
        foreach ($w in $warn) { Write-Output "!!! $w" }
        Write-Output "!!! ---------------------------------------------------------------------"
    }

    Pop-Location
} catch { }

exit 0
