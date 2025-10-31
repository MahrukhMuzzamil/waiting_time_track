param(
  [string]$TaskName = "ClinicWaitTimeApp",
  [string]$User = "$env:UserName"
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Split-Path -Parent $root
$bat = Join-Path $root 'run_clinic.bat'

if (!(Test-Path $bat)) { throw "Batch file not found: $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c \"$bat\""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

try {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
} catch {}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Run clinic wait-time app on logon"
Write-Host "Registered scheduled task '$TaskName'."


