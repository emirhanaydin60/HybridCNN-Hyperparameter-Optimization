param(
    [string]$PythonPath = "C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repositoryRoot

if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "Python interpreter not found: $PythonPath"
}

$configPath = Join-Path $repositoryRoot "results\main_experiment\config_used.json"
$config = Get-Content $configPath -Raw | ConvertFrom-Json
if (
    $config.dataset -ne "cifar10" -or
    $config.model_name -ne "hybrid_cnn" -or
    $config.population_size -ne 8 -or
    $config.iteration_count -ne 15 -or
    $config.search_epochs -ne 6 -or
    $config.final_epochs -ne 50
) {
    throw "Archived CIFAR-10 GWO configuration differs from the expected paper protocol"
}

$studyRoot = Join-Path $repositoryRoot "results\igwo_comparison\paper_config_4x15"
foreach ($seed in @(42, 43, 44)) {
    $baselineRun = $seed - 41
    $baselinePath = Join-Path $repositoryRoot "results\main_experiment\GWO\run_$('{0:d2}' -f $baselineRun)\summary.json"
    if (-not (Test-Path $baselinePath -PathType Leaf)) {
        throw "Existing GWO baseline for seed $seed is missing: $baselinePath"
    }
    $baseline = Get-Content $baselinePath -Raw | ConvertFrom-Json
    if ($baseline.run_seed -ne $seed -or $baseline.algorithm -ne "gwo") {
        throw "Unexpected GWO baseline at $baselinePath"
    }
    $seedStudy = Join-Path $studyRoot "seed_$seed"
    if (Test-Path $seedStudy) {
        throw "Refusing to overwrite an existing seed study: $seedStudy"
    }
}

$activeRunners = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object {
        $_.CommandLine -match "run_igwo_existing_gwo_seed.py|run_igwo_comparison.py"
    }
if ($activeRunners) {
    throw "An IGWO/GWO runner is already active; refusing to start duplicate studies"
}

$launchLogDirectory = Join-Path $studyRoot "launch_logs"
New-Item -Path $launchLogDirectory -ItemType Directory -Force | Out-Null
$gpuLockPath = Join-Path $studyRoot "gpu_training.lock"
[System.IO.File]::WriteAllBytes($gpuLockPath, [byte[]](0))

$launches = @()
foreach ($seed in @(42, 43, 44)) {
    $seedStudy = Join-Path $studyRoot "seed_$seed"
    $stdoutPath = Join-Path $launchLogDirectory "seed_$seed.stdout.log"
    $stderrPath = Join-Path $launchLogDirectory "seed_$seed.stderr.log"
    $arguments = @(
        "-u",
        "scripts\run_igwo_existing_gwo_seed.py",
        "--seed",
        "$seed",
        "--study-dir",
        $seedStudy,
        "--gpu-lock",
        $gpuLockPath
    )
    $process = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $repositoryRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    $launches += [PSCustomObject]@{
        seed = $seed
        pid = $process.Id
        study_directory = $seedStudy
        training_log = (Join-Path $seedStudy "CIFAR10\IGWO\run_01\training.log")
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        gpu_lock = $gpuLockPath
    }
}

$launches |
    ConvertTo-Json -Depth 4 |
    Set-Content -Path (Join-Path $studyRoot "process_launch.json") -Encoding UTF8
$launches | Format-Table -AutoSize
