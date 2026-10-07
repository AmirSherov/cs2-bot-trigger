param([switch]$Extended)
$ErrorActionPreference = 'Stop'
$program = Join-Path $PSScriptRoot 'gpu-trigger.exe'
$cases = @(
    @{ File = 'person-near.png'; Expected = 1 },
    @{ File = 'person-group.png'; Expected = 1 },
    @{ File = 'wall.png'; Expected = 0 },
    @{ File = 'sky-wall.png'; Expected = 0 },
    @{ File = 'legs-gap-corrected.png'; Expected = 0 }
)
foreach ($case in $cases) {
    $file = Join-Path (Join-Path $PSScriptRoot 'tests') $case.File
    $result = & $program --image $file --benchmark 5 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Runtime failed: $($case.File): $result" }
    $match = [regex]::Match(($result -join "`n"), 'center_hit=(\d)')
    if (-not $match.Success -or [int]$match.Groups[1].Value -ne $case.Expected) {
        throw "Detection mismatch: $($case.File): $result"
    }
    Write-Output "PASS $($case.File) center_hit=$($case.Expected)"
}
Write-Output '5 image integration checks passed. No mouse input was generated.'
if ($Extended) {
    foreach ($roi in @(640,960)) {
        foreach ($case in $cases) {
            $file = Join-Path (Join-Path $PSScriptRoot 'tests') $case.File
            $reference = $null
            foreach ($variant in @('fast-avx2','fast-scalar','full-avx2','full-scalar')) {
                $arguments = @('--image',$file,'--benchmark','3','--roi',$roi.ToString())
                if ($variant -like 'full-*') { $arguments += @('--model',(Join-Path $PSScriptRoot 'person-seg-320.onnx')) }
                if ($variant -like '*-scalar') { $arguments += '--no-avx2' }
                $result = & $program @arguments 2>&1
                if ($LASTEXITCODE -ne 0) { throw "Runtime failed: $($case.File) $variant" }
                $text = $result -join "`n"
                $match = [regex]::Match($text,'center_hit=(\d) person_confidence=([0-9.]+) mask_probability=([0-9.]+)')
                if (-not $match.Success -or [int]$match.Groups[1].Value -ne $case.Expected) { throw "Mismatch: ROI=$roi $($case.File) $variant $text" }
                $values = @([double]::Parse($match.Groups[2].Value,[Globalization.CultureInfo]::InvariantCulture),[double]::Parse($match.Groups[3].Value,[Globalization.CultureInfo]::InvariantCulture))
                if ($null -eq $reference) { $reference=$values }
                elseif ([Math]::Abs($values[0]-$reference[0]) -gt .002 -or [Math]::Abs($values[1]-$reference[1]) -gt .002) { throw "Numerical mismatch: ROI=$roi $($case.File) $variant" }
            }
            Write-Output "PASS parity ROI=$roi $($case.File): full/compact and AVX2/scalar"
        }
    }
    Write-Output '40 backend parity checks passed without generating mouse input.'
}
