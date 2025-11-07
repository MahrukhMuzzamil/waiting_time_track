param(
    [string]$RtspUrl,
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [string]$Browser = ""
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$configPath = Join-Path $projectRoot 'config.json'

if ([string]::IsNullOrWhiteSpace($RtspUrl)) {
    if (Test-Path $configPath) {
        try {
            $cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
            if ($cfg -and $cfg.rtsp_url) {
                $RtspUrl = [string]$cfg.rtsp_url
            }
        } catch {
            Write-Warning "Failed to read config.json: $($_.Exception.Message)"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($RtspUrl)) {
    throw "Provide -RtspUrl or populate rtsp_url in config.json."
}

$venvActivate = Join-Path $projectRoot '.venv\Scripts\Activate.ps1'
if (!(Test-Path $venvActivate)) {
    throw "Virtual environment not found. Run 'python -m venv .venv' and install requirements first."
}

$serverScript = @"
Set-Location -LiteralPath "$projectRoot"
. "$venvActivate"
`$env:RTSP_URL = "$RtspUrl"
`$env:PORT = "$Port"
if (-not `$env:CONF_THRESHOLD) { `$env:CONF_THRESHOLD = "0.4" }
python server.py
"@

$encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($serverScript))

Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-NoExit",
    "-EncodedCommand", $encoded
) -WindowStyle Minimized | Out-Null

Start-Sleep -Seconds 5

if (-not $NoBrowser) {
    $url = "http://localhost:$Port/video_ai"
    if ([string]::IsNullOrWhiteSpace($Browser)) {
        Start-Process $url | Out-Null
    } else {
        Start-Process -FilePath $Browser -ArgumentList $url | Out-Null
    }
    Write-Host "Opened guard dashboard at $url"
} else {
    Write-Host "Server started on http://localhost:$Port"
}


