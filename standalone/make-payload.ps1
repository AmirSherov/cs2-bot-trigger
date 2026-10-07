$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $PSScriptRoot 'build'
$crtRoot = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Redist\MSVC\14.50.35710\x64\Microsoft.VC145.CRT'
$payloadFiles = @(
    @{ Name = 'gpu-trigger.exe'; Path = (Join-Path $projectRoot 'gpu-trigger.exe') },
    @{ Name = 'person-seg-320.onnx'; Path = (Join-Path $projectRoot 'person-seg-320.onnx') },
    @{ Name = 'person-seg-fast-320.onnx'; Path = (Join-Path $projectRoot 'person-seg-fast-320.onnx') },
    @{ Name = 'onnxruntime.dll'; Path = (Join-Path $projectRoot 'onnxruntime.dll') },
    @{ Name = 'onnxruntime_providers_shared.dll'; Path = (Join-Path $projectRoot 'onnxruntime_providers_shared.dll') },
    @{ Name = 'DirectML.dll'; Path = (Join-Path $projectRoot 'DirectML.dll') }
)
foreach ($name in @('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll')) {
    $payloadFiles += @{ Name = $name; Path = (Join-Path $crtRoot $name) }
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $projectRoot 'licenses') -File) {
    $payloadFiles += @{ Name = ('license-' + $file.Name); Path = $file.FullName }
}
$rows = @()
$resources = @('#include <windows.h>')
$fingerprints = @()
$id = 101
foreach ($file in $payloadFiles) {
    $hash = (Get-FileHash -LiteralPath $file.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $size = (Get-Item -LiteralPath $file.Path).Length
    $packed = Join-Path $buildRoot ($id.ToString() + '.bin')
    & (Join-Path $buildRoot 'pack.exe') $file.Path $packed
    if ($LASTEXITCODE -ne 0) { throw "Compression failed: $($file.Name)" }
    $rows += ('    { L"' + $file.Name + '", ' + $id + ', ' + $size + 'ULL, "' + $hash + '" },')
    $resources += ($id.ToString() + ' RCDATA "' + ($packed.Replace('\','/')) + '"')
    $fingerprints += ($file.Name + ':' + $hash)
    ++$id
}
$sha = [System.Security.Cryptography.SHA256]::Create()
try { $versionBytes = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($fingerprints -join "`n")) }
finally { $sha.Dispose() }
$version = ([BitConverter]::ToString($versionBytes).Replace('-','').ToLowerInvariant()).Substring(0,24)
$header = @('#pragma once', '#include <cstdint>', 'struct Payload { const wchar_t* name; int resource; uint64_t size; const char* sha256; };',
    ('inline constexpr wchar_t CacheVersion[] = L"' + $version + '";'), 'inline constexpr Payload Payloads[] = {') + $rows + @('};')
Set-Content -LiteralPath (Join-Path $buildRoot 'manifest.h') -Value $header -Encoding ASCII
Set-Content -LiteralPath (Join-Path $buildRoot 'payload.rc') -Value $resources -Encoding ASCII
Set-Content -LiteralPath (Join-Path $buildRoot 'cache-version.txt') -Value $version -Encoding ASCII
Write-Output "Embedded $($payloadFiles.Count) files. Cache version: $version"
