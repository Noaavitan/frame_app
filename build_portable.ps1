# build_portable.ps1 — Build a portable ZIP (Windows, onedir)

param(
  [string]$AppPy   = "app.py",
  [string]$AppName = "Teken Frame",
  [string]$Icon    = "assets\app.ico",   # אופציונלי; יידלג אם לא קיים
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

# 4) Clean previous build/dist
if (Test-Path ".\build") { Remove-Item -Recurse -Force ".\build" }
if (Test-Path ".\dist")  { Remove-Item -Recurse -Force ".\dist"  }

# 5) Build (onedir) with flet pack — include only resources that exist
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

# Include image folder (DRONE_IMG)
if (Test-Path ".\image") {
  $packArgs += @("--add-data", "image;image")
} else {
  Write-Host "image folder not found - skipping." -ForegroundColor Yellow
}

# Include assets folder if exists
if (Test-Path ".\assets") {
  $packArgs += @("--add-data", "assets;assets")
} else {
  Write-Host "assets folder not found - skipping." -ForegroundColor Yellow
}

# Pack exiftool.exe ליד ה-EXE (resolve_exiftool_path יאתר אותו)
if (Test-Path ".\exiftool-13.30_64\exiftool.exe") {
  $packArgs += @("--add-binary", "exiftool-13.30_64\exiftool.exe;.")
} elseif (Test-Path ".\exiftool.exe") {
  $packArgs += @("--add-binary", "exiftool.exe;.")
} else {
  Write-Host "exiftool.exe not found in project - relying on PATH/auto-discovery." -ForegroundColor Yellow
}

# Locate flet CLI in venv; fallback to global
$fletExe = Join-Path (Split-Path $py) "flet.exe"
if (!(Test-Path $fletExe)) { $fletExe = "flet" }

Write-Host "Running: flet pack ..." -ForegroundColor Cyan
& $fletExe pack @packArgs

# 6) Create portable ZIP (pick the newest dist folder)
if (!(Test-Path ".\dist")) { throw "Build failed: dist folder not created." }

$distDirs = Get-ChildItem "dist" -Directory | Sort-Object LastWriteTime -Descending
if ($distDirs.Count -eq 0) { throw "Build failed: no subfolders in dist." }

# Prefer exact $AppName folder if exists; otherwise take the newest
$destDir = Join-Path "dist" $AppName
if (!(Test-Path $destDir)) {
  $destDir = $distDirs[0].FullName
  Write-Host "Note: Using newest dist folder: $destDir" -ForegroundColor Yellow
}

# Mirror _internal\image → image (and assets if exist) so נתיבים יחסיים יעבדו
$internalImage = Join-Path $destDir "_internal\image"
$publicImage   = Join-Path $destDir "image"
if (Test-Path $internalImage) {
  if (!(Test-Path $publicImage)) {
    Copy-Item -Recurse -Force $internalImage $publicImage
    Write-Host "Copied _internal\image → image" -ForegroundColor Green
  }
}

$internalAssets = Join-Path $destDir "_internal\assets"
$publicAssets   = Join-Path $destDir "assets"
if (Test-Path $internalAssets) {
  if (!(Test-Path $publicAssets)) {
    Copy-Item -Recurse -Force $internalAssets $publicAssets
    Write-Host "Copied _internal\assets → assets" -ForegroundColor Green
  }
}

$stamp   = Get-Date -Format "yyyyMMdd_HHmm"
$zipName = "${AppName}_Portable_${Version}_$stamp.zip"

if (Test-Path $zipName) { Remove-Item $zipName -Force }
Compress-Archive -Path (Join-Path $destDir "*") -DestinationPath $zipName -Force

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "EXE folder: $destDir"
Write-Host "ZIP file : $zipName"
