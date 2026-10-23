#Requires -Version 7.0
<#
.SYNOPSIS
    Removes everything deploy.ps1 created, in reverse order.

.DESCRIPTION
    Order: HTTP API, Lambda function (its invoke permission goes with it), log group, S3 bucket
    (emptied only after a prompt), ECR repository, IAM role. Refuses to run during the judging
    window (2026-10-27 to 2026-11-09) unless -AllowDuringJudging is given.
    -DryRun lists what would be deleted without changing anything.
#>
[CmdletBinding()]
param(
    [string]$Region = 'ap-southeast-1',
    [string]$Name = 'opencv-hackathon-nextview',
    [string]$RoleName = 'opencv-hackathon-lambda-role',
    [switch]$DryRun,
    [switch]$Yes,
    [switch]$KeepBucket,
    [switch]$AllowDuringJudging
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$today = (Get-Date).Date
if (-not $DryRun -and -not $AllowDuringJudging -and $today -ge [datetime]'2026-10-27' -and $today -le [datetime]'2026-11-09') {
    throw 'Refusing to tear down during the judging window (2026-10-27 to 2026-11-09). Pass -AllowDuringJudging to override.'
}

$Account = (Invoke-Aws sts get-caller-identity).Account
$Bucket = "$Name-$Account"
$LogGroup = "/aws/lambda/$Name"

# Discover
$apis = @((Invoke-Aws apigatewayv2 get-apis).Items | Where-Object { $_.Name -eq $Name })
$fn = Get-AwsOrNull lambda get-function-configuration --function-name $Name
$logs = @((Invoke-Aws logs describe-log-groups --log-group-name-prefix $LogGroup).logGroups |
    Where-Object { $_.logGroupName -eq $LogGroup })
$bucketExists = (Invoke-AwsRaw s3api head-bucket --bucket $Bucket).ExitCode -eq 0
$repo = Get-AwsOrNull ecr describe-repositories --repository-names $Name
$role = Get-AwsOrNull iam get-role --role-name $RoleName

Write-Step "Teardown plan (account $Account, region $Region)"
Write-Note "HTTP API:  $(if ($apis.Count) { ($apis | ForEach-Object { "$($_.ApiId) $($_.ApiEndpoint)" }) -join '; ' } else { '(none)' })"
Write-Note "Lambda:    $(if ($fn) { $fn.FunctionArn } else { '(none)' })"
Write-Note "Log group: $(if ($logs.Count) { $LogGroup } else { '(none)' })"
Write-Note "S3 bucket: $(if ($bucketExists) { "$Bucket$(if ($KeepBucket) { ' (kept: -KeepBucket)' })" } else { '(none)' })"
Write-Note "ECR repo:  $(if ($repo) { $repo.repositories[0].repositoryUri } else { '(none)' })"
Write-Note "IAM role:  $(if ($role) { $role.Role.Arn } else { '(none)' })"

if ($DryRun) { Write-Step 'Dry run: nothing deleted.'; return }

if (-not $Yes) {
    $answer = Read-Host "Type '$Name' to delete these resources"
    if ($answer -ne $Name) { Write-Step 'Aborted.'; return }
}

foreach ($a in $apis) {
    Write-Step "Deleting HTTP API $($a.ApiId)"
    Invoke-Aws apigatewayv2 delete-api --api-id $a.ApiId | Out-Null
}

if ($fn) {
    Write-Step "Deleting Lambda function $Name"
    Invoke-Aws lambda delete-function --function-name $Name | Out-Null
}

if ($logs.Count) {
    Write-Step "Deleting log group $LogGroup"
    Invoke-Aws logs delete-log-group --log-group-name $LogGroup | Out-Null
}

if ($bucketExists -and -not $KeepBucket) {
    $listing = Invoke-Aws s3api list-objects-v2 --bucket $Bucket --max-items 1000
    $count = if ($listing -and $listing.Contents) { @($listing.Contents).Count } else { 0 }
    $deleteBucket = $true
    if ($count -gt 0) {
        $empty = $Yes
        if (-not $Yes) { $empty = (Read-Host "Bucket $Bucket holds $count+ object(s). Empty it? [y/N]") -match '^(y|yes)$' }
        if ($empty) {
            Write-Step "Emptying s3://$Bucket"
            $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            & aws s3 rm "s3://$Bucket" --recursive --region $Region | Out-Null
            $rmOk = ($LASTEXITCODE -eq 0)
            $ErrorActionPreference = $prev
            if (-not $rmOk) { throw "emptying s3://$Bucket failed" }
        } else {
            Write-Note 'bucket kept (not emptied)'
            $deleteBucket = $false
        }
    }
    if ($deleteBucket) {
        Write-Step "Deleting bucket $Bucket"
        Invoke-Aws s3api delete-bucket --bucket $Bucket | Out-Null
    }
}

if ($repo) {
    Write-Step "Deleting ECR repository $Name (with its images)"
    Invoke-Aws ecr delete-repository --repository-name $Name --force | Out-Null
}

if ($role) {
    Write-Step "Deleting IAM role $RoleName"
    foreach ($p in @((Invoke-Aws iam list-role-policies --role-name $RoleName).PolicyNames)) {
        if ($p) { Invoke-Aws iam delete-role-policy --role-name $RoleName --policy-name $p | Out-Null }
    }
    foreach ($p in @((Invoke-Aws iam list-attached-role-policies --role-name $RoleName).AttachedPolicies)) {
        if ($p) { Invoke-Aws iam detach-role-policy --role-name $RoleName --policy-arn $p.PolicyArn | Out-Null }
    }
    Invoke-Aws iam delete-role --role-name $RoleName | Out-Null
}

Write-Step 'Teardown complete.'
