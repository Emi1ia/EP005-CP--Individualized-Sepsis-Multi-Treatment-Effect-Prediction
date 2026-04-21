param(
  [string]$DataRoot = "c:\Users\emili\sepsis_project\data",
  [string]$PhysioNetUsername = $env:PHYSIONET_USERNAME,
  [string]$PhysioNetPassword = $env:PHYSIONET_PASSWORD,
  [switch]$IncludeMimicEd,
  [switch]$SkipRestricted
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Downloader = Join-Path $ScriptDir "download_physionet.py"

if (!(Test-Path -LiteralPath $Downloader)) {
  throw "Downloader script not found: $Downloader"
}

function Invoke-PhysioNetDownload {
  param(
    [string]$BaseUrl,
    [string]$Dest,
    [bool]$Restricted
  )

  $args = @(
    $Downloader,
    "--base-url", $BaseUrl,
    "--dest", $Dest
  )

  if ($Restricted) {
    if ([string]::IsNullOrWhiteSpace($PhysioNetUsername) -or [string]::IsNullOrWhiteSpace($PhysioNetPassword)) {
      throw "Missing PhysioNet credentials. Set PHYSIONET_USERNAME/PHYSIONET_PASSWORD or pass -PhysioNetUsername/-PhysioNetPassword."
    }
    $args += @("--username", $PhysioNetUsername, "--password", $PhysioNetPassword)
  }

  Write-Host "`n[download] $BaseUrl -> $Dest"
  python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Download failed for $BaseUrl"
  }
}

# Open dataset: PhysioNet Challenge 2019 (sync/repair)
Invoke-PhysioNetDownload `
  -BaseUrl "https://physionet.org/files/challenge-2019/1.0.0/training/" `
  -Dest (Join-Path $DataRoot "challenge-2019\training") `
  -Restricted $false

if (-not $SkipRestricted) {
  # Credentialed dataset: MIMIC-IV v3.1 hosp/icu
  Invoke-PhysioNetDownload `
    -BaseUrl "https://physionet.org/files/mimiciv/3.1/hosp/" `
    -Dest (Join-Path $DataRoot "mimiciv\3.1\hosp") `
    -Restricted $true

  Invoke-PhysioNetDownload `
    -BaseUrl "https://physionet.org/files/mimiciv/3.1/icu/" `
    -Dest (Join-Path $DataRoot "mimiciv\3.1\icu") `
    -Restricted $true

  if ($IncludeMimicEd) {
    Invoke-PhysioNetDownload `
      -BaseUrl "https://physionet.org/files/mimic-iv-ed/2.2/" `
      -Dest (Join-Path $DataRoot "mimic-iv-ed\2.2") `
      -Restricted $true
  }
} else {
  Write-Host "`nSkipping restricted datasets (MIMIC-IV/MIMIC-IV-ED)."
}

Write-Host "`nAll requested dataset downloads completed."
