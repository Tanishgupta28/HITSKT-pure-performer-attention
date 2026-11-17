param([ValidateSet('development','production')][string]$Mode = 'development', [switch]$Demo)
$ErrorActionPreference = 'Stop'
$platformRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimeDirectory = Join-Path $platformRoot '.platform-runtime'
$apiExecutable = Join-Path $platformRoot 'backend\.venv\Scripts\python.exe'
$nextExecutable = Join-Path $platformRoot 'frontend\node_modules\next\dist\bin\next'
if (!(Test-Path -LiteralPath $apiExecutable) -or !(Test-Path -LiteralPath $nextExecutable)) {
    throw 'Install the backend and frontend dependencies first. See docs/platform/README.md.'
}
foreach ($platformPort in @(8000, 3000)) {
    if (Get-NetTCPConnection -LocalPort $platformPort -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $platformPort is already in use. Stop that service before launching another copy."
    }
}
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
$nodeExecutable = (Get-Command node.exe).Source
$webArguments = @('node_modules/next/dist/bin/next','dev','--hostname','127.0.0.1','--port','3000')
if ($Mode -eq 'production') {
    if (!(Test-Path -LiteralPath (Join-Path $platformRoot 'frontend\.next\standalone\server.js'))) {
        throw 'Run npm.cmd run build in frontend before using production mode.'
    }
    $webArguments = @('scripts/start.mjs')
}
$priorDemoSetting = $env:DEMO_ENABLED
try {
    $env:DEMO_ENABLED = if ($Demo) { 'true' } else { 'false' }
    if ($Demo) {
        & $apiExecutable (Join-Path $platformRoot 'backend/tools/demo_dataset.py') --seed-only
        if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the local sample accounts.' }
    }
    $apiProcess = Start-Process -FilePath $apiExecutable -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory (Join-Path $platformRoot 'backend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeDirectory 'backend.log') -RedirectStandardError (Join-Path $runtimeDirectory 'backend-error.log') -PassThru
} finally {
    $env:DEMO_ENABLED = $priorDemoSetting
}
$webProcess = Start-Process -FilePath $nodeExecutable -ArgumentList $webArguments -WorkingDirectory (Join-Path $platformRoot 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeDirectory 'frontend.log') -RedirectStandardError (Join-Path $runtimeDirectory 'frontend-error.log') -PassThru
# Process.Path can briefly be null after Start-Process; record the executables
# supplied to the launcher so the stop helper can still verify ownership.
@(
    @{id=$apiProcess.Id; started_at=$apiProcess.StartTime.ToUniversalTime().ToString('o'); executable=$apiExecutable}
    @{id=$webProcess.Id; started_at=$webProcess.StartTime.ToUniversalTime().ToString('o'); executable=$nodeExecutable}
) | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDirectory 'processes.json')
Write-Output 'Lumen is starting at http://localhost:3000. Logs: .platform-runtime. Stop with scripts/stop-platform.ps1.'
