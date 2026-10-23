# Helpers shared by deploy.ps1, teardown.ps1 and smoke_test.ps1 (dot-sourced).
# Credentials come from the caller's environment (AWS_PROFILE or the standard AWS_* variables).

$script:NotFoundPattern = 'ResourceNotFoundException|NoSuchEntity|RepositoryNotFoundException|NotFoundException|NoSuchBucket|\(404\)|Not Found|does not exist'
$script:DeniedPattern = 'AccessDenied|UnauthorizedOperation|not authorized'

function Invoke-AwsRaw {
    # Runs the AWS CLI with JSON output. Returns ExitCode, Out (stdout) and Err (stderr) without throwing.
    $errFile = [System.IO.Path]::GetTempFileName()
    $prev = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $out = & aws @args --region $script:Region --output json 2> $errFile
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    $err = Get-Content -Raw -LiteralPath $errFile -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $errFile -ErrorAction SilentlyContinue
    [pscustomobject]@{ ExitCode = $code; Out = (@($out) -join "`n"); Err = "$err".Trim() }
}

function Invoke-Aws {
    # Runs the AWS CLI and returns parsed JSON (or $null for empty output); throws on failure.
    $r = Invoke-AwsRaw @args
    if ($r.ExitCode -ne 0) { throw "aws $($args[0]) $($args[1]) failed: $($r.Err)" }
    if ([string]::IsNullOrWhiteSpace($r.Out)) { return $null }
    return $r.Out | ConvertFrom-Json
}

function Get-AwsOrNull {
    # Like Invoke-Aws, but returns $null when the resource does not exist.
    $r = Invoke-AwsRaw @args
    if ($r.ExitCode -eq 0) {
        if ([string]::IsNullOrWhiteSpace($r.Out)) { return $null }
        return $r.Out | ConvertFrom-Json
    }
    if ($r.Err -match $script:NotFoundPattern) { return $null }
    throw "aws $($args[0]) $($args[1]) failed: $($r.Err)"
}

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Note([string]$Message) {
    Write-Host "    $Message"
}

function ConvertTo-FileUri([string]$Path) {
    return 'file://' + ($Path -replace '\\', '/')
}

function New-TempJson($Object, [string]$Text) {
    # Writes JSON (from an object or raw text) to a temp file and returns its path.
    $path = Join-Path ([System.IO.Path]::GetTempPath()) ("nv-" + [guid]::NewGuid().ToString('N') + '.json')
    if ($null -eq $Text -or $Text -eq '') { $Text = $Object | ConvertTo-Json -Depth 10 }
    [System.IO.File]::WriteAllText($path, $Text, [System.Text.UTF8Encoding]::new($false))
    return $path
}

function Expand-Template([string]$Path, [hashtable]$Values) {
    $text = Get-Content -Raw -LiteralPath $Path
    foreach ($k in $Values.Keys) { $text = $text.Replace("{{$k}}", [string]$Values[$k]) }
    return $text
}
