param(
    [string]$PythonPath = "C:\Users\emirh\anaconda3\envs\pytorch-v1\python.exe"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repositoryRoot

if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "Python interpreter not found: $PythonPath"
}

$adaptiveRoot = Join-Path $repositoryRoot "results\igwo_comparison\paper_config_6x15_adaptive"
$studyRoot = Join-Path $repositoryRoot "results\igwo_comparison\paper_config_8x15_random_local"
$seeds = @(42, 43, 44)
foreach ($launchArtifact in @("launch_claim.json", "process_launch.json")) {
    if (Test-Path (Join-Path $studyRoot $launchArtifact)) {
        throw "A prior random-local launch artifact exists; refusing a duplicate launch"
    }
}

foreach ($seed in $seeds) {
    $adaptiveSeedRoot = Join-Path $adaptiveRoot "seed_$seed"
    $manifestPath = Join-Path $adaptiveSeedRoot "study_manifest.json"
    $runRoot = Join-Path $adaptiveSeedRoot "CIFAR10\IGWO\run_01"
    if (-not (Test-Path $manifestPath -PathType Leaf)) {
        throw "Adaptive seed $seed has no completion manifest: $manifestPath"
    }
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    $summaryPath = Join-Path $runRoot "summary.json"
    $summary = if (Test-Path $summaryPath -PathType Leaf) {
        Get-Content $summaryPath -Raw | ConvertFrom-Json
    } else {
        $null
    }
    $requiredFiles = @(
        "metrics.json",
        "history.json",
        "confusion_matrix.json",
        "igwo_trace.json",
        "config_used.json",
        "best_model.pth",
        "training.log"
    )
    if (
        $manifest.status -ne "completed" -or
        $manifest.seed -ne $seed -or
        $manifest.igwo_config.population_size -ne 6 -or
        $manifest.igwo_config.iteration_count -ne 15 -or
        $manifest.local_search_candidates_per_iteration -ne 2 -or
        $null -eq $summary -or
        $summary.run_seed -ne $seed -or
        $summary.config.population_size -ne 6 -or
        $summary.config.iteration_count -ne 15
    ) {
        throw "Adaptive seed $seed is not a verified successful 6x15 run"
    }
    foreach ($fileName in $requiredFiles) {
        $path = Join-Path $runRoot $fileName
        if (-not (Test-Path $path -PathType Leaf) -or (Get-Item $path).Length -eq 0) {
            throw "Adaptive seed $seed is missing a required output: $path"
        }
    }
    $confusion = Get-Content (Join-Path $runRoot "confusion_matrix.json") -Raw | ConvertFrom-Json
    if ($confusion.total -ne 10000 -or $confusion.matrix.Count -ne 10) {
        throw "Adaptive seed $seed has an invalid CIFAR-10 confusion matrix"
    }
    $trace = Get-Content (Join-Path $runRoot "igwo_trace.json") -Raw | ConvertFrom-Json
    if (
        $trace.configured_iterations -ne 15 -or
        $trace.local_search_candidates_per_iteration -ne 2
    ) {
        throw "Adaptive seed $seed has an invalid final IGWO trace"
    }
}

$activeRunners = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object {
        $_.CommandLine -match "run_igwo_existing_gwo_seed.py" -and
        $_.CommandLine -match [regex]::Escape($adaptiveRoot)
    }
if ($activeRunners) {
    throw "Adaptive IGWO processes are still active; refusing to launch the new study"
}

foreach ($seed in $seeds) {
    $seedStudy = Join-Path $studyRoot "seed_$seed"
    if (Test-Path $seedStudy) {
        throw "Refusing to overwrite an existing random-local seed study: $seedStudy"
    }
}

$claimPath = Join-Path $studyRoot "launch_claim.json"
New-Item -Path $studyRoot -ItemType Directory -Force | Out-Null
$claimStream = [System.IO.File]::Open(
    $claimPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
$claimWriter = [System.IO.StreamWriter]::new($claimStream)
$claimWriter.Write(
    (@{ status = "launching"; seeds = $seeds; started_at = (Get-Date).ToString("o") } | ConvertTo-Json -Depth 3)
)
$claimWriter.Dispose()

$launchLogDirectory = Join-Path $studyRoot "launch_logs"
New-Item -Path $launchLogDirectory -ItemType Directory -Force | Out-Null
$gpuLockPath = Join-Path $adaptiveRoot "gpu_training.lock"
$configPath = Join-Path $repositoryRoot "results\main_experiment\config_used.json"
$launches = @()
foreach ($seed in $seeds) {
    $seedStudy = Join-Path $studyRoot "seed_$seed"
    $stdoutPath = Join-Path $launchLogDirectory "seed_$seed.stdout.log"
    $stderrPath = Join-Path $launchLogDirectory "seed_$seed.stderr.log"
    $arguments = @(
        "-u",
        "scripts\run_igwo_random_local_seed.py",
        "--seed",
        "$seed",
        "--config",
        $configPath,
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
    $launches |
        ConvertTo-Json -Depth 4 |
        Set-Content -Path (Join-Path $studyRoot "process_launch.json") -Encoding UTF8
}

$launches |
    ConvertTo-Json -Depth 4 |
    Set-Content -Path (Join-Path $studyRoot "process_launch.json") -Encoding UTF8
@{
    status = "started"
    seeds = $seeds
    started_at = (Get-Date).ToString("o")
    process_launch = (Join-Path $studyRoot "process_launch.json")
} | ConvertTo-Json -Depth 3 |
    Set-Content -Path $claimPath -Encoding UTF8
$launches | Format-Table -AutoSize
