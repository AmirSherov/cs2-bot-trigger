$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'build'))
$stageRoot = [IO.Path]::GetFullPath((Join-Path $buildRoot 'portable-stage'))
if ([IO.Path]::GetDirectoryName($stageRoot) -ne $buildRoot) { throw 'Unsafe staging path' }
if (Test-Path -LiteralPath $stageRoot) { Remove-Item -LiteralPath $stageRoot -Recurse -Force }
$packageRoot = Join-Path $stageRoot 'gpu-trigger-portable'
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
foreach ($name in @('gpu-trigger.exe','person-seg-320.onnx','person-seg-fast-320.onnx','onnxruntime.dll','onnxruntime_providers_shared.dll','DirectML.dll')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $packageRoot -ErrorAction Stop
}
$crtRoot = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Redist\MSVC\14.50.35710\x64\Microsoft.VC145.CRT'
foreach ($name in @('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll')) {
    Copy-Item -LiteralPath (Join-Path $crtRoot $name) -Destination $packageRoot -ErrorAction Stop
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'licenses') -Destination (Join-Path $packageRoot 'licenses') -Recurse
Set-Content -LiteralPath (Join-Path $packageRoot 'run-live.cmd') -Encoding ASCII -Value @('@echo off','"%~dp0gpu-trigger.exe" --live','if errorlevel 1 pause')
Set-Content -LiteralPath (Join-Path $packageRoot 'run-dry.cmd') -Encoding ASCII -Value @('@echo off','"%~dp0gpu-trigger.exe"','if errorlevel 1 pause')
Set-Content -LiteralPath (Join-Path $packageRoot 'README.txt') -Encoding UTF8 -Value @(
    'Полностью распакуйте архив. Запустите run-live.cmd для кликов или run-dry.cmd для проверки.',
    'В игре: F8 — включить, удержание Mouse4 — разрешить реакцию, F10 — завершить.',
    'Область по умолчанию 640x640; --roi 960 возвращает прежний размер области.',
    'Быстрая модель person-seg-fast-320.onnx используется автоматически.',
    'AVX2 используется при поддержке CPU; --no-avx2 включает резервную обработку.',
    'Сохраните DLL и модели рядом с EXE. Python и CUDA для запуска не нужны.',
    'При уменьшении ROI возможны ошибки распознавания; 320x320 не рекомендуется.',
    'Медианные замеры — не гарантированный предел задержки в игре.'
)
$checksums = Get-ChildItem -LiteralPath $packageRoot -File -Recurse | ForEach-Object {
    (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $_.FullName.Substring($packageRoot.Length + 1)
}
Set-Content -LiteralPath (Join-Path $packageRoot 'SHA256SUMS.txt') -Value $checksums -Encoding ASCII
$temporaryZip = Join-Path $buildRoot 'gpu-trigger-portable-new.zip'
Compress-Archive -LiteralPath $packageRoot -DestinationPath $temporaryZip -CompressionLevel Optimal -Force
Move-Item -LiteralPath $temporaryZip -Destination (Join-Path $projectRoot 'gpu-trigger-portable.zip') -Force
Write-Output 'Portable package rebuilt.'

