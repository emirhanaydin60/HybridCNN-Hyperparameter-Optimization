param(
    [string]$PythonPath = "C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repositoryRoot

if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "Python interpreter not found: $PythonPath"
}

$studyRoot = Join-Path $repositoryRoot "results\igwo_comparison\paper_config_8x15_random_local"
New-Item -Path $studyRoot -ItemType Directory -Force | Out-Null
foreach ($seed in @(42, 43, 44)) {
    if (Test-Path (Join-Path $studyRoot "seed_$seed")) {
        throw "Refusing to overwrite an existing random-local seed study"
    }
}

$claimPath = Join-Path $studyRoot "supervisor_start_claim.json"
$claimStream = [System.IO.File]::Open(
    $claimPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
$claimWriter = [System.IO.StreamWriter]::new($claimStream)
$claimWriter.Write(
    (@{
        status = "starting"
        protected_pids = @{ seed_42 = 20104; seed_43 = 9112; seed_44 = 2412 }
        started_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4)
)
$claimWriter.Dispose()

$stdoutPath = Join-Path $studyRoot "supervisor.stdout.log"
$stderrPath = Join-Path $studyRoot "supervisor.stderr.log"
$supervisorScript = Join-Path $repositoryRoot "scripts\supervise_igwo_8x15_after_adaptive.py"
$arguments = @(
    "-u",
    $supervisorScript,
    "--expected-pids",
    "42=20104,43=9112,44=2412",
    "--python-path",
    $PythonPath
)
$process = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $repositoryRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

@{
    status = "started"
    supervisor_pid = $process.Id
    started_at = (Get-Date).ToString("o")
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    supervisor_log = (Join-Path $studyRoot "supervisor.log")
    status_file = (Join-Path $studyRoot "supervisor_status.json")
} | ConvertTo-Json -Depth 4 |
    Set-Content -Path $claimPath -Encoding UTF8

[PSCustomObject]@{
    supervisor_pid = $process.Id
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    supervisor_log = (Join-Path $studyRoot "supervisor.log")
    status_file = (Join-Path $studyRoot "supervisor_status.json")
} | Format-List
