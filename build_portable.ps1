# build_portable.ps1 - Build a portable ZIP (Windows, onedir)

param(
  [string]$AppPy   = "app.py",
  [string]$AppName = "Teken Frame",
  [string]$Icon    = "assets\app.ico",  # optional; ignored if not found
  [string]$Version = "v1.0.0"
)

$ErrorActionPreference = "Stop"

# 1) Activate venv if exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
}

# 2) Ensure deps via python -m pip
$py = ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

& $py -m pip install --upgrade pip
& $py -m pip install "flet==0.28.3" "flet-cli==0.28.3" exifread pyinstaller

# 3) Sanity checks
if (!(Test-Path $AppPy)) { throw "File not found: $AppPy" }

# 4) Clean previous build/dist (optional)
if (Test-Path ".\build") { Remove-Item -Recurse -Force ".\build" }
if (Test-Path ".\dist")  { Remove-Item -Recurse -Force ".\dist"  }

# 5) Build (onedir) with flet pack - only include assets that exist
$packArgs = @(
  $AppPy,
  "--name", $AppName,
  "--product-name", $AppName,
  "--onedir",
  "--hidden-import", "flet",
  "--hidden-import", "exifread"
)

# Optional icon
if (Test-Path $Icon) {
  $packArgs += @("--icon", $Icon)
} else {
  Write-Host "Icon not found ($Icon) - skipping." -ForegroundColor Yellow
}

# Include image folder if exists (you use DRONE_IMG = 'image/...'):
if (Test-Path ".\image") {
  $packArgs += @("--add-data", "image;image")
} else {
  Write-Host "image folder not found - skipping." -ForegroundColor Yellow
}

# Include assets folder only if it exists:
if (Test-Path ".\assets") {
  $packArgs += @("--add-data", "assets;assets")
} else {
  Write-Host "assets folder not found - skipping." -ForegroundColor Yellow
}

# Include exiftool if present
if (Test-Path ".\exiftool-13.30_64\exiftool.exe") {
  $packArgs += @("--add-binary", "exiftool-13.30_64\exiftool.exe;.")
} elseif (Test-Path ".\exiftool.exe") {
  $packArgs += @("--add-binary", "exiftool.exe;.")
} else {
  Write-Host "exiftool.exe not found in project - relying on PATH/auto-discovery." -ForegroundColor Yellow
}

Write-Host "Running: flet pack" -ForegroundColor Cyan
& flet pack @packArgs

# 6) Create portable ZIP
$stamp   = Get-Date -Format "yyyyMMdd_HHmm"
$destDir = Join-Path "dist" $AppName
if (!(Test-Path $destDir)) { throw "Build output folder not found: $destDir" }
$zipName = "${AppName}_Portable_${Version}_$stamp.zip"

if (Test-Path $zipName) { Remove-Item $zipName -Force }
Compress-Archive -Path (Join-Path $destDir "*") -DestinationPath $zipName -Force

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "EXE folder: $destDir"
Write-Host "ZIP file : $zipName"
