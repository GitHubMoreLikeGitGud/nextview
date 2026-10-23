#Requires -Version 7.0
<#
.SYNOPSIS
    Smoke test for a deployed NextView endpoint.

.DESCRIPTION
    Checks GET /api/health (opencv 5.0.0), GET / (HTML containing "NextView" with a CSP header) and,
    unless -HealthOnly, GET /app.js, /api/presets, /api/board.png and /api/evidence (baked figures present),
    then POST /api/session followed by a few POST /api/step calls. Prints latencies (client and the handler's own).
    The URL defaults to api_url in infra/deployed.json. Throws (exit code 1) on any failed check.
#>
[CmdletBinding()]
param(
    [string]$Url,
    [switch]$HealthOnly,
    [string]$Preset = 'wide_robot_handheld',
    [int]$Steps = 3,
    [int]$ReadyTimeoutS = 60
)

$ErrorActionPreference = 'Stop'
if (-not $Url) {
    $deployed = Join-Path $PSScriptRoot 'deployed.json'
    if (-not (Test-Path -LiteralPath $deployed)) { throw 'no -Url given and infra/deployed.json not found' }
    $Url = (Get-Content -Raw -LiteralPath $deployed | ConvertFrom-Json).api_url
}
$Url = $Url.TrimEnd('/')
$failures = [System.Collections.Generic.List[string]]::new()

function Invoke-Timed([string]$Method, [string]$Path, $Body) {
    $params = @{ Uri = "$Url$Path"; Method = $Method; SkipHttpErrorCheck = $true; TimeoutSec = 40 }
    if ($null -ne $Body) {
        $params.Body = ($Body | ConvertTo-Json -Depth 20 -Compress)
        $params.ContentType = 'application/json'
    }
    # The stage is throttled (2 req/s, burst 5): retry 429 and 5xx like the web UI (0.5, 1, 2, 4 s).
    $backoff = @(0.5, 1, 2, 4)
    $retries = 0
    for ($try = 0; ; $try++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $resp = Invoke-WebRequest @params
        $sw.Stop()
        $code = [int]$resp.StatusCode
        if (($code -eq 429 -or $code -ge 500) -and $try -lt $backoff.Count -and [string]$resp.Content -notmatch '"INTERNAL"') {
            $retries++
            Start-Sleep -Milliseconds ([int](1000 * $backoff[$try]))
            continue
        }
        break
    }
    [pscustomobject]@{
        Status = $code
        Ms = [math]::Round($sw.Elapsed.TotalMilliseconds)
        Content = [string]$resp.Content
        Headers = $resp.Headers
        Retries = $retries
    }
}

function Test-Check([bool]$Ok, [string]$Label) {
    if ($Ok) { Write-Host "  PASS  $Label" -ForegroundColor Green }
    else { Write-Host "  FAIL  $Label" -ForegroundColor Red; $failures.Add($Label) }
}

Write-Host "Smoke test: $Url"

# A new stage can return 404 for a few seconds while auto-deploy finishes.
$deadline = (Get-Date).AddSeconds($ReadyTimeoutS)
do {
    $health = Invoke-Timed GET '/api/health'
    if ($health.Status -eq 200) { break }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)

$hb = $null
try { $hb = $health.Content | ConvertFrom-Json } catch { }
Test-Check ($health.Status -eq 200) "GET /api/health -> $($health.Status) in $($health.Ms) ms"
Test-Check ($hb -and $hb.ok -eq $true -and $hb.opencv -eq '5.0.0') "health body: $($health.Content)"

$root = Invoke-Timed GET '/'
Test-Check ($root.Status -eq 200 -and $root.Content -match 'NextView') "GET / -> $($root.Status) in $($root.Ms) ms, contains NextView"
Test-Check ([bool]$root.Headers['content-security-policy']) 'GET / has a content-security-policy header'

$warm = 1..5 | ForEach-Object { (Invoke-Timed GET '/api/health').Ms }
Write-Host "  warm /api/health ms: $($warm -join ', ')"

if (-not $HealthOnly) {
    $js = Invoke-Timed GET '/app.js'
    Test-Check ($js.Status -eq 200 -and [string]$js.Headers['content-type'] -match 'javascript') "GET /app.js -> $($js.Status) ($($js.Headers['content-type']))"
    $presets = Invoke-Timed GET '/api/presets'
    $pb = $null
    try { $pb = $presets.Content | ConvertFrom-Json } catch { }
    Test-Check ($presets.Status -eq 200 -and $pb -and $pb.presets.Count -ge 6) "GET /api/presets -> $($presets.Status), $($pb.presets.Count) presets"
    $board = Invoke-Timed GET '/api/board.png?kind=screen'
    Test-Check ($board.Status -eq 200 -and [string]$board.Headers['content-type'] -eq 'image/png') "GET /api/board.png?kind=screen -> $($board.Status) ($($board.Headers['content-type']))"
    $ev = Invoke-Timed GET '/api/evidence'
    $eb = $null
    try { $eb = $ev.Content | ConvertFrom-Json } catch { }
    $nFig = if ($eb) { @($eb._figures).Count } else { 0 }
    Test-Check ($ev.Status -eq 200 -and $eb -and $eb.available -eq $true -and $nFig -ge 10) "GET /api/evidence -> $($ev.Status), available $($eb.available), $nFig figures"
    if ($nFig -gt 0) {
        $fig = Invoke-Timed GET ('/evidence/' + [uri]::EscapeDataString(@($eb._figures)[0]))
        Test-Check ($fig.Status -eq 200 -and [string]$fig.Headers['content-type'] -eq 'image/png') "GET /evidence/$(@($eb._figures)[0]) -> $($fig.Status)"
    }
}

if (-not $HealthOnly) {
    $session = Invoke-Timed POST '/api/session' @{ mode = 'sim'; preset = $Preset; seed = 42 }
    $sb = $null
    try { $sb = $session.Content | ConvertFrom-Json } catch { }
    Test-Check ($session.Status -eq 200 -and $sb -and $sb.state) "POST /api/session -> $($session.Status) in $($session.Ms) ms (retries $($session.Retries))"
    if ($sb -and $sb.state) {
        $state = $sb.state
        for ($i = 1; $i -le $Steps; $i++) {
            $step = Invoke-Timed POST '/api/step' @{ state = $state; choose_candidate = $null }
            $stb = $null
            try { $stb = $step.Content | ConvertFrom-Json } catch { }
            Test-Check ($step.Status -eq 200 -and $stb -and $stb.state) "POST /api/step #$i -> $($step.Status) in $($step.Ms) ms (server $($stb.latency_ms.total) ms, rule $($stb.decision.rule), retries $($step.Retries))"
            if ($stb -and $stb.state) { $state = $stb.state } else { break }
        }
    }
}

if ($failures.Count -gt 0) { throw "smoke test failed: $($failures.Count) check(s)" }
Write-Host 'Smoke test passed.' -ForegroundColor Green
