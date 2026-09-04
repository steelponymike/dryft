# SessionStart hook: injects memory/STATUS.md into session context via stdout.
# Makes the "read STATUS.md first" protocol automatic instead of discipline-based.
# Emits a size banner and an end marker so a PARTIAL delivery cannot pass unnoticed:
# the warning rides at the TOP (it survives truncation and shows up in the preview a
# large payload gets reduced to), and the END marker rides at the BOTTOM (seeing it is
# proof the whole file arrived). Added 2026-08-06 after a session answered from 624 of
# 794 lines without registering that it had only part of the file.
# NOTE: keep this file pure ASCII - PowerShell 5.1 misparses non-ASCII in BOM-less files.
$ErrorActionPreference = 'SilentlyContinue'

# Resolve the directory the SESSION is actually in, not where it was launched.
# Per the worktrees doc: "Hook paths don't follow the worktree. ${CLAUDE_PROJECT_DIR} stays
# put... cwd follows Claude." So in a `claude -w <name>` session, CLAUDE_PROJECT_DIR still
# points at the MAIN checkout - injecting its STATUS.md would hand the session the wrong
# branch's file and say nothing about it. The hook's stdin JSON carries the real cwd.
# Guarded by IsInputRedirected so a hook invoked without stdin can never block session start.
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

$p = Join-Path $root 'memory/STATUS.md'
# Roughly where this file stops fitting in a single Read (25k-token cap). Tune if that moves.
$readLimitBytes = 45000
if (Test-Path $p) {
    $f     = Get-Item $p
    $days  = [int]((Get-Date) - $f.LastWriteTime).TotalDays
    $kb    = [int]($f.Length / 1024)
    # @(...).Count, NOT Measure-Object -Line: -Line silently skips blank lines and
    # under-reported this file by 51 (760 vs 811), which would tell a reader it was
    # finished at 94% of the file - the exact failure this banner exists to prevent.
    $lines = @(Get-Content $p).Count
    $endMarker = "=== END STATUS.md - all $lines lines delivered ==="

    Write-Output "=== memory/STATUS.md - auto-injected by SessionStart hook ==="
    Write-Output "SIZE: $lines lines / $kb KB. Working copy last touched ~$days days ago; the"
    Write-Output "frontmatter 'updated:' date is authoritative - flag it to Mike if over 5 days old."
    Write-Output "PROOF OF FULL DELIVERY: this injection ends with a line reading"
    Write-Output "  $endMarker"
    Write-Output "If you cannot see that exact line, you are holding a PARTIAL file."
    if ($f.Length -gt $readLimitBytes) {
        Write-Output ""
        Write-Output "!!! TRUNCATION RISK: STATUS.md is $kb KB, past the ~45 KB single-read limit.  !!!"
        Write-Output "!!! Treat it as PARTIAL until you have actually seen the END line above.     !!!"
        Write-Output "!!! Never conclude 'that is not in STATUS' from a partial read - either page !!!"
        Write-Output "!!! the rest (Read with offset=) or Grep STATUS.md for the term directly.    !!!"
    }
    Write-Output "============================================================================"
    Get-Content $p -Raw
    Write-Output $endMarker
}
exit 0
