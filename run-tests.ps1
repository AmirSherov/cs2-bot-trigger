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
