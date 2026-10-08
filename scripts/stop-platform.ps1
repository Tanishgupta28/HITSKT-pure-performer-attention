$ErrorActionPreference = 'Stop'
$platformRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$processFile = Join-Path $platformRoot '.platform-runtime\processes.json'
if (!(Test-Path -LiteralPath $processFile)) {
    Write-Output 'No platform processes were recorded by run-platform.ps1.'
    exit 0
}
$records = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
function Stop-PlatformTree([int]$ownedProcessId) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ownedProcessId"
    foreach ($child in $children) { Stop-PlatformTree -ownedProcessId $child.ProcessId }
    Stop-Process -Id $ownedProcessId -ErrorAction SilentlyContinue
}
foreach ($record in $records) {
    $platformProcess = Get-Process -Id $record.id -ErrorAction SilentlyContinue
    if (!$platformProcess) { continue }
    $recordedStartTicks = ([datetime]$record.started_at).ToUniversalTime().Ticks
    if ($platformProcess.StartTime.ToUniversalTime().Ticks -ne $recordedStartTicks -or $platformProcess.Path -ne $record.executable) {
        Write-Warning "Process $($record.id) no longer matches the recorded platform process; left running."
        continue
    }
    Stop-PlatformTree -ownedProcessId $platformProcess.Id
}
Write-Output 'Recorded platform processes stopped. MongoDB and other services were left running.'
