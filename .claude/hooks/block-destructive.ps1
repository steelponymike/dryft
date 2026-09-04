# PreToolUse guard for Bash/PowerShell commands. Exit 2 = block the tool call.
# Two tiers:
#   - Always-block: history/force destroyers. Claude never runs these; if one is
#     genuinely needed, Mike runs it himself.
#   - Tree deletion: blocked EXCEPT when the path is under .claude/worktrees/
#     (the save protocol's husk sweep stays allowed).
$ErrorActionPreference = 'SilentlyContinue'
$raw = [Console]::In.ReadToEnd()
if (-not $raw) { exit 0 }
try { $payload = $raw | ConvertFrom-Json } catch { exit 0 }
$cmd = [string]$payload.tool_input.command
if (-not $cmd) { exit 0 }

# ---------------------------------------------------------------------------
# Match against a SANITISED copy of the command, not the raw string.
#
# The patterns below match ANYWHERE in the command text, so a commit message that
# merely DESCRIBES a blocked command was itself blocked - this fired on 2026-09-02
# on a message documenting a file-discard command, and again on 2026-09-03.
# A message is data, not an instruction to the shell.
#
# Deliberately narrow, because over-stripping would blind the guard:
#   - Quoted -m / --message arguments are always stripped: by definition a message.
#   - Heredoc bodies are stripped ONLY for commands that consume a heredoc as TEXT
#     (git commit/tag/notes, gh pr/issue). A heredoc fed to a SHELL still executes,
#     so 'bash <<EOF ... rm -rf / ... EOF' must remain fully visible to the guard.
# ---------------------------------------------------------------------------
$scan = $cmd
if ($scan -match '(?s)\bgit\s+(commit|tag|notes)\b' -or $scan -match '(?s)\bgh\s+(pr|issue)\b') {
    $scan = [regex]::Replace($scan, '(?sm)<<-?\s*([''"]?)([A-Za-z_][A-Za-z0-9_]*)\1.*?^\s*\2\s*$', ' <HEREDOC> ')
}
# No backslash escapes in these patterns on purpose: a literal '\\' has been silently
# collapsed to '\' more than once while writing this file through a shell heredoc, which
# made the pattern invalid, made [regex]::Replace throw, and left $scan un-sanitised -
# i.e. the guard quietly reverted to its old behaviour. Keep them backslash-free.
$scan = [regex]::Replace($scan, '(?s)(?:-m|--message)(?:=|\s+)"[^"]*"', ' -m <MSG> ')
$scan = [regex]::Replace($scan, '(?s)(?:-m|--message)(?:=|\s+)''[^'']*''', ' -m <MSG> ')

$alwaysBlock = @(
    'git\s+reset\s+--hard',
    'git\s+clean(?=[^|;&]*\s-[A-Za-z]*f)',
    'git\s+checkout\s+--\s',
    'git\s+push[^|;&]*--force(?!-with-lease)',
    'git\s+push[^|;&]*\s-f\b',
    # PowerShell -match is CASE-INSENSITIVE, so the old '-D' pattern also blocked the
    # SAFE '-d' (which refuses to delete anything unmerged). (?-i) forces case
    # sensitivity so only the force-delete is caught. --force is blocked separately:
    # 'branch --delete --force' is the same destructive act spelled out, and it used
    # to slip through entirely.
    '(?-i)git\s+branch[^|;&]*\s-[A-Za-z]*D\b',
    'git\s+branch[^|;&]*--force',
    'git\s+worktree\s+remove[^|;&]*--force',
    'git\s+stash\s+(drop|clear)'
)
$treeDelete = @(
    '(?<!git\s)\brm\s+-[A-Za-z]*[rR]',
    'Remove-Item\s[^|;&]*-Recurse',
    '\brmdir\s[^|;&]*-Recurse',
    '\brd\s+/s',
    '\brmdir\s+/s',
    '\bdel\s+/s'
)

foreach ($p in $alwaysBlock) {
    if ($scan -match $p) {
        [Console]::Error.WriteLine("BLOCKED (block-destructive hook): matches always-block pattern '$p'. If this operation is genuinely needed, Mike runs it himself.")
        exit 2
    }
}
# Normalise separators with a plain string Replace (char 92 = backslash) so the regex
# itself needs no backslash - see the note above.
$worktreeSweep = $scan.Replace([string][char]92, '/') -match '\.claude/+worktrees'
foreach ($p in $treeDelete) {
    if ($scan -match $p) {
        if ($worktreeSweep) { continue }
        [Console]::Error.WriteLine("BLOCKED (block-destructive hook): recursive deletion outside .claude/worktrees/ (pattern '$p'). Announce the deletion and let Mike confirm or run it himself.")
        exit 2
    }
}
exit 0
