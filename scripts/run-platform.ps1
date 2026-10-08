param([ValidateSet('development','production')][string]$Mode = 'development')
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
$apiProcess = Start-Process -FilePath $apiExecutable -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory (Join-Path $platformRoot 'backend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeDirectory 'backend.log') -RedirectStandardError (Join-Path $runtimeDirectory 'backend-error.log') -PassThru
$webProcess = Start-Process -FilePath $nodeExecutable -ArgumentList $webArguments -WorkingDirectory (Join-Path $platformRoot 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeDirectory 'frontend.log') -RedirectStandardError (Join-Path $runtimeDirectory 'frontend-error.log') -PassThru
@($apiProcess, $webProcess) | ForEach-Object {
    @{id=$_.Id; started_at=$_.StartTime.ToUniversalTime().ToString('o'); executable=$_.Path}
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDirectory 'processes.json')
Write-Output 'Lumen is starting at http://localhost:3000. Logs: .platform-runtime. Stop with scripts/stop-platform.ps1.'
