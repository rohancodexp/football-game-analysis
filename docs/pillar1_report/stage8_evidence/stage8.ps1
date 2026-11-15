# Stage 8 full-video regression. Writes only under results\stage8 (gitignored). Never edits code or git.
$ErrorActionPreference = 'Continue'
$repo = 'C:\Users\lokes\Desktop\capstone\football-game-analysis'
$old  = 'C:\Users\lokes\Desktop\capstone\main_4ad4329'   # untouched export of pre-Stage-5 main (4ad4329)
Set-Location $repo
$py  = Join-Path $repo '.venv\Scripts\python.exe'
$R   = Join-Path $repo 'results\stage8'
if (Test-Path $R) { Write-Host 'results\stage8 already exists - refusing to overwrite evidence.' -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Path "$R\logs" | Out-Null
$L = "$R\logs"
$data = 'datasets\infactory_soccer_events\data'

function Run-Step {
    param([string]$Name, [string[]]$Cmd)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $out = & $py -B @Cmd 2>&1 | ForEach-Object { $s = "$_"; Write-Host $s; $s }
    $code = $LASTEXITCODE
    $out | Set-Content -Encoding UTF8 "$L\$Name.log"
    $line = "$Name exit=$code seconds=$([int]((Get-Date) - $t0).TotalSeconds)"
    Add-Content -Encoding UTF8 "$L\summary.txt" $line
    if ($code -eq 0) { Write-Host $line -ForegroundColor Green } else { Write-Host $line -ForegroundColor Red }
}

# Environment and provenance
$envlog = @()
$envlog += "date: $(Get-Date -Format o)"
$envlog += "head: $(git rev-parse HEAD)"
$envlog += "branch: $(git branch --show-current)"
$envlog += "origin/main: $(git rev-parse origin/main)"
$envlog += "git status lines: $((git status --short | Measure-Object).Count)"
$envlog += "python: $(& $py --version 2>&1)"
$envlog += (& $py -m pip freeze 2>&1 | ForEach-Object { "pip: $_" })
foreach ($v in '1','3') { $h = Get-FileHash "$data\$v.mp4" -Algorithm SHA256; $envlog += "video $v.mp4 sha256: $($h.Hash.ToLower())" }
$envlog | Set-Content -Encoding UTF8 "$L\environment.txt"
$envlog | Select-Object -First 6 | ForEach-Object { Write-Host $_ }

# 1. Full test suite on merged main
Run-Step 'tests' @('-m', 'unittest', 'discover', '-p', 'test_*.py')

# 2. Per video: old code (baseline), new code (candidate), validator, regression gate
foreach ($v in '3', '1') {
    Run-Step "old_v$v"      @("$old\run.py", '--video', "$data\$v.mp4", '--out', "$R\old_v$v", '--render', '--team-save-crops')
    Run-Step "new_v$v"      @('run.py',      '--video', "$data\$v.mp4", '--out', "$R\new_v$v", '--render', '--team-save-crops')
    Run-Step "validate_v$v" @('validate_tracking_outputs.py', '--video', "$data\$v.mp4", '--run-dir', "$R\new_v$v", '--out', "$R\validation_v$v.json", '--require-render')
    Run-Step "regress_v$v"  @('validate_pillar1_regression.py', '--baseline', "$R\old_v$v", '--candidate', "$R\new_v$v", '--out', "$R\regression_v$v.json")
}

Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
Get-Content "$L\summary.txt"
Write-Host "`nDone. Tell Claude 'stage 8 done'." -ForegroundColor Yellow
