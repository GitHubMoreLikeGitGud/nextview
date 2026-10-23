#Requires -Version 7.0
<#
.SYNOPSIS
    Idempotent deploy of NextView: ECR image, Lambda (container), HTTP API, logs and S3 bucket.

.DESCRIPTION
    Every create is preceded by a describe/get, so re-running keeps the same API URL and the
    same NEXTVIEW_STATE_KEY. AWS credentials come from the environment (for example AWS_PROFILE).
    Writes infra/deployed.json and finishes with infra/smoke_test.ps1.

.EXAMPLE
    pwsh infra/deploy.ps1
.EXAMPLE
    pwsh infra/deploy.ps1 -SmokeHealthOnly
#>
[CmdletBinding()]
param(
    [string]$Region = 'ap-southeast-1',
    [string]$Name = 'opencv-hackathon-nextview',
    [string]$RoleName = 'opencv-hackathon-lambda-role',
    [ValidateRange(128, 10240)][int]$MemoryMb = 2048,
    [ValidateRange(3, 900)][int]$TimeoutS = 30,
    [int]$LogRetentionDays = 14,
    [double]$ThrottleRate = 2,
    [int]$ThrottleBurst = 5,
    [string]$ImageTag,
    [switch]$SkipBuild,
    [switch]$SkipSmoke,
    [switch]$SmokeHealthOnly
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$tempFiles = [System.Collections.Generic.List[string]]::new()

try {
    # 1. Account and region
    Write-Step 'Caller identity'
    $Account = (Invoke-Aws sts get-caller-identity).Account
    Write-Note "account $Account, region $Region"
    $Bucket = "$Name-$Account"
    $LogGroup = "/aws/lambda/$Name"

    # 2. Execution role
    Write-Step "IAM role $RoleName"
    $role = Get-AwsOrNull iam get-role --role-name $RoleName
    $roleCreated = $false
    if (-not $role) {
        $role = Invoke-Aws iam create-role --role-name $RoleName `
            --assume-role-policy-document (ConvertTo-FileUri (Join-Path $PSScriptRoot 'trust.json')) `
            --description 'NextView Lambda execution role'
        $roleCreated = $true
        Write-Note 'created'
    } else {
        Write-Note 'exists'
    }
    $RoleArn = $role.Role.Arn

    $policyText = Expand-Template (Join-Path $PSScriptRoot 'role-policy.template.json') @{
        REGION = $Region; ACCOUNT = $Account; FUNCTION = $Name; BUCKET = $Bucket
    }
    $policyFile = New-TempJson -Text $policyText; $tempFiles.Add($policyFile)
    $r = Invoke-AwsRaw iam put-role-policy --role-name $RoleName --policy-name nextview-inline `
        --policy-document (ConvertTo-FileUri $policyFile)
    if ($r.ExitCode -eq 0) {
        Write-Note 'inline policy nextview-inline (logs on its own group, S3 reports/traces/eval) applied'
    } elseif ($r.Err -match $script:DeniedPattern) {
        Write-Warning "put-role-policy denied; falling back to the AWSLambdaBasicExecutionRole managed policy (no S3 access): $($r.Err)"
        Invoke-Aws iam attach-role-policy --role-name $RoleName `
            --policy-arn 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole' | Out-Null
    } else {
        throw "put-role-policy failed: $($r.Err)"
    }

    # 3. ECR repository
    Write-Step "ECR repository $Name"
    $repo = Get-AwsOrNull ecr describe-repositories --repository-names $Name
    if (-not $repo) {
        $repo = Invoke-Aws ecr create-repository --repository-name $Name --image-tag-mutability MUTABLE
        $RepoUri = $repo.repository.repositoryUri
        Write-Note 'created'
    } else {
        $RepoUri = $repo.repositories[0].repositoryUri
        Write-Note 'exists'
    }
    Invoke-Aws ecr put-lifecycle-policy --repository-name $Name `
        --lifecycle-policy-text (ConvertTo-FileUri (Join-Path $PSScriptRoot 'ecr-lifecycle.json')) | Out-Null
    Write-Note 'lifecycle: keep the 3 most recent images'
    Invoke-Aws ecr put-image-scanning-configuration --repository-name $Name `
        --image-scanning-configuration scanOnPush=true | Out-Null
    Write-Note 'basic vulnerability scan on push'

    # 4. Build and push
    $baseDigest = ((Get-Content -LiteralPath (Join-Path $RepoRoot 'Dockerfile')) |
        Select-String -Pattern '^FROM\s+\S+@(sha256:[0-9a-f]{64})' | Select-Object -First 1).Matches.Groups[1].Value
    if (-not $ImageTag) {
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $sha = (& git -C $RepoRoot rev-parse --short HEAD 2>$null)
        $gitOk = ($LASTEXITCODE -eq 0)
        $dirty = (& git -C $RepoRoot status --porcelain 2>$null)
        $ErrorActionPreference = $prev
        if (-not $gitOk) { $sha = 'nocommit' }
        $ImageTag = $sha
        if (-not $gitOk -or $dirty) { $ImageTag = "$sha-$((Get-Date).ToUniversalTime().ToString('yyyyMMddHHmmss'))" }
    }
    $registry = $RepoUri.Split('/')[0]
    if (-not $SkipBuild) {
        Write-Step 'Bake the Evidence tab (results/figures + headline numbers)'
        $py = @((Join-Path $RepoRoot '.venv/Scripts/python.exe'), (Join-Path $RepoRoot '.venv/bin/python')) |
            Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if (-not $py) { $py = 'python' }
        & $py (Join-Path $RepoRoot 'scripts/bake_evidence.py')
        if ($LASTEXITCODE -ne 0) { throw 'scripts/bake_evidence.py failed' }

        Write-Step "Build and push ${RepoUri}:$ImageTag"
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry | Out-Null
        $loginOk = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = $prev
        if (-not $loginOk) { throw 'docker login to ECR failed' }

        & docker buildx build --platform linux/amd64 --provenance=false --sbom=false `
            --build-arg "GIT_SHA=$ImageTag" -t "${RepoUri}:$ImageTag" --load $RepoRoot
        if ($LASTEXITCODE -ne 0) { throw 'docker build failed' }
        & docker push "${RepoUri}:$ImageTag"
        if ($LASTEXITCODE -ne 0) { throw 'docker push failed' }
    } else {
        Write-Step "Skipping build; using ${RepoUri}:$ImageTag"
    }
    $img = Invoke-Aws ecr describe-images --repository-name $Name --image-ids "imageTag=$ImageTag"
    $ImageDigest = $img.imageDetails[0].imageDigest
    $mediaType = $img.imageDetails[0].imageManifestMediaType
    if ($mediaType -match 'index|list') { throw "pushed image is a manifest index ($mediaType); Lambda needs a single-platform manifest" }
    $ImageUri = "$RepoUri@$ImageDigest"
    Write-Note "image $ImageDigest ($mediaType, $([math]::Round($img.imageDetails[0].imageSizeInBytes / 1MB)) MB compressed)"

    # 5. Log group
    Write-Step "Log group $LogGroup"
    $r = Invoke-AwsRaw logs create-log-group --log-group-name $LogGroup
    if ($r.ExitCode -ne 0 -and $r.Err -notmatch 'ResourceAlreadyExistsException') { throw "create-log-group failed: $($r.Err)" }
    Invoke-Aws logs put-retention-policy --log-group-name $LogGroup --retention-in-days $LogRetentionDays | Out-Null
    Write-Note "retention $LogRetentionDays days"

    # 6. S3 bucket
    Write-Step "S3 bucket $Bucket"
    $r = Invoke-AwsRaw s3api head-bucket --bucket $Bucket
    if ($r.ExitCode -ne 0) {
        if ($r.Err -notmatch '404|Not Found') { throw "head-bucket failed (bucket name taken or no access): $($r.Err)" }
        Invoke-Aws s3api create-bucket --bucket $Bucket --create-bucket-configuration "LocationConstraint=$Region" | Out-Null
        Write-Note 'created'
    } else {
        Write-Note 'exists'
    }
    Invoke-Aws s3api put-public-access-block --bucket $Bucket --public-access-block-configuration `
        'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true' | Out-Null
    Invoke-Aws s3api put-bucket-encryption --bucket $Bucket `
        --server-side-encryption-configuration (ConvertTo-FileUri (Join-Path $PSScriptRoot 'bucket-encryption.json')) | Out-Null
    $bpFile = New-TempJson -Text (Expand-Template (Join-Path $PSScriptRoot 'bucket-policy.template.json') @{ BUCKET = $Bucket })
    $tempFiles.Add($bpFile)
    Invoke-Aws s3api put-bucket-policy --bucket $Bucket --policy (ConvertTo-FileUri $bpFile) | Out-Null
    Invoke-Aws s3api put-bucket-lifecycle-configuration --bucket $Bucket `
        --lifecycle-configuration (ConvertTo-FileUri (Join-Path $PSScriptRoot 'lifecycle.json')) | Out-Null
    Write-Note 'public access blocked, SSE-S3, TLS-only policy, reports/ and traces/ expire after 30 days'

    # 7. Lambda function
    Write-Step "Lambda function $Name"
    $fn = Get-AwsOrNull lambda get-function-configuration --function-name $Name
    $vars = [ordered]@{}
    if ($fn -and $fn.Environment -and $fn.Environment.Variables) {
        foreach ($p in $fn.Environment.Variables.PSObject.Properties) { $vars[$p.Name] = $p.Value }
    }
    if (-not $vars['NEXTVIEW_STATE_KEY']) {
        $keyBytes = [byte[]]::new(32)
        [System.Security.Cryptography.RandomNumberGenerator]::Fill($keyBytes)
        $vars['NEXTVIEW_STATE_KEY'] = [Convert]::ToBase64String($keyBytes)
        Write-Note 'generated a new NEXTVIEW_STATE_KEY'
    } else {
        Write-Note 'kept the existing NEXTVIEW_STATE_KEY'
    }
    $vars['NEXTVIEW_BUCKET'] = $Bucket
    $vars['OPENBLAS_NUM_THREADS'] = '1'
    $vars['NEXTVIEW_VERSION'] = $ImageTag
    $envFile = New-TempJson -Object @{ Variables = $vars }; $tempFiles.Add($envFile)

    # Some accounts cap function memory below the requested size; the API error states the cap.
    function Step-DownMemory([string]$Err) {
        if ($Err -match "'MemorySize' value failed to satisfy constraint: Member must have value less than or equal to (\d+)" -and
            [int]$Matches[1] -lt $script:MemoryMb) {
            Write-Warning "account memory cap is $($Matches[1]) MB; deploying with $($Matches[1]) MB instead of $script:MemoryMb MB"
            $script:MemoryMb = [int]$Matches[1]
            return $true
        }
        return $false
    }

    if ($fn) {
        Invoke-Aws lambda update-function-code --function-name $Name --image-uri $ImageUri | Out-Null
        Invoke-Aws lambda wait function-updated-v2 --function-name $Name | Out-Null
        while ($true) {
            $r = Invoke-AwsRaw lambda update-function-configuration --function-name $Name --role $RoleArn `
                --memory-size $MemoryMb --timeout $TimeoutS --ephemeral-storage Size=512 `
                --environment (ConvertTo-FileUri $envFile)
            if ($r.ExitCode -eq 0) { break }
            if (Step-DownMemory $r.Err) { continue }
            throw "update-function-configuration failed: $($r.Err)"
        }
        Invoke-Aws lambda wait function-updated-v2 --function-name $Name | Out-Null
        Write-Note "updated to $ImageDigest"
    } else {
        # A just-created role can take a few seconds before Lambda is allowed to assume it.
        $deadline = (Get-Date).AddSeconds(90)
        while ($true) {
            $r = Invoke-AwsRaw lambda create-function --function-name $Name --package-type Image `
                --code "ImageUri=$ImageUri" --role $RoleArn --architectures x86_64 `
                --memory-size $MemoryMb --timeout $TimeoutS --ephemeral-storage Size=512 `
                --environment (ConvertTo-FileUri $envFile) `
                --description 'NextView next-best-view calibration agent'
            if ($r.ExitCode -eq 0) { break }
            if (Step-DownMemory $r.Err) { continue }
            if ($r.Err -match 'cannot be assumed|role defined for the function' -and (Get-Date) -lt $deadline) {
                Write-Note 'waiting for the new role to propagate...'
                Start-Sleep -Seconds 5
                continue
            }
            throw "create-function failed: $($r.Err)"
        }
        Invoke-Aws lambda wait function-active-v2 --function-name $Name | Out-Null
        Write-Note 'created'
    }
    $fn = Invoke-Aws lambda get-function-configuration --function-name $Name
    $FunctionArn = $fn.FunctionArn
    Write-Note "memory $($fn.MemorySize) MB, timeout $($fn.Timeout) s, state $($fn.State)"

    # 8. HTTP API
    Write-Step "HTTP API $Name"
    $api = (Invoke-Aws apigatewayv2 get-apis).Items | Where-Object { $_.Name -eq $Name } | Select-Object -First 1
    if (-not $api) {
        $api = Invoke-Aws apigatewayv2 create-api --name $Name --protocol-type HTTP `
            --description 'NextView public API'
        Write-Note 'created'
    } else {
        Write-Note 'exists'
    }
    $ApiId = $api.ApiId
    $ApiUrl = $api.ApiEndpoint

    $integ = (Invoke-Aws apigatewayv2 get-integrations --api-id $ApiId).Items |
        Where-Object { $_.IntegrationUri -eq $FunctionArn } | Select-Object -First 1
    if (-not $integ) {
        $integ = Invoke-Aws apigatewayv2 create-integration --api-id $ApiId --integration-type AWS_PROXY `
            --integration-uri $FunctionArn --payload-format-version 2.0 --timeout-in-millis 29000
    } else {
        $integ = Invoke-Aws apigatewayv2 update-integration --api-id $ApiId --integration-id $integ.IntegrationId `
            --payload-format-version 2.0 --timeout-in-millis 29000
    }
    $target = "integrations/$($integ.IntegrationId)"

    $route = (Invoke-Aws apigatewayv2 get-routes --api-id $ApiId).Items |
        Where-Object { $_.RouteKey -eq '$default' } | Select-Object -First 1
    if (-not $route) {
        Invoke-Aws apigatewayv2 create-route --api-id $ApiId --route-key '$default' --target $target | Out-Null
    } elseif ($route.Target -ne $target) {
        Invoke-Aws apigatewayv2 update-route --api-id $ApiId --route-id $route.RouteId --target $target | Out-Null
    }

    $routeSettings = "ThrottlingBurstLimit=$ThrottleBurst,ThrottlingRateLimit=$ThrottleRate"
    $stage = (Invoke-Aws apigatewayv2 get-stages --api-id $ApiId).Items |
        Where-Object { $_.StageName -eq '$default' } | Select-Object -First 1
    if (-not $stage) {
        Invoke-Aws apigatewayv2 create-stage --api-id $ApiId --stage-name '$default' --auto-deploy `
            --default-route-settings $routeSettings | Out-Null
    } else {
        Invoke-Aws apigatewayv2 update-stage --api-id $ApiId --stage-name '$default' --auto-deploy `
            --default-route-settings $routeSettings | Out-Null
    }
    Write-Note "route `$default -> Lambda (payload 2.0, 29 s); stage `$default throttled rate $ThrottleRate/s burst $ThrottleBurst"

    $sourceArn = "arn:aws:execute-api:${Region}:${Account}:${ApiId}/*/*"
    $statementId = 'apigw-invoke'
    $hasPermission = $false
    $pol = Get-AwsOrNull lambda get-policy --function-name $Name
    if ($pol) {
        $doc = $pol.Policy | ConvertFrom-Json
        foreach ($st in @($doc.Statement)) {
            if ($st.Sid -eq $statementId) {
                if (($st.Condition.ArnLike.'AWS:SourceArn') -eq $sourceArn) { $hasPermission = $true }
                else { Invoke-Aws lambda remove-permission --function-name $Name --statement-id $statementId | Out-Null }
            }
        }
    }
    if (-not $hasPermission) {
        Invoke-Aws lambda add-permission --function-name $Name --statement-id $statementId `
            --action lambda:InvokeFunction --principal apigateway.amazonaws.com --source-arn $sourceArn | Out-Null
        Write-Note 'invoke permission for API Gateway added'
    }

    # 9. Concurrency
    Write-Step 'Concurrency'
    $acct = Invoke-Aws lambda get-account-settings
    $cap = [int]$acct.AccountLimit.ConcurrentExecutions
    $reserved = (Invoke-Aws lambda get-function-concurrency --function-name $Name)
    $reservedNow = if ($reserved -and $null -ne $reserved.ReservedConcurrentExecutions) { [int]$reserved.ReservedConcurrentExecutions } else { $null }
    if ($reservedNow -eq 0) {
        Write-Warning 'reserved concurrency is 0 (kill switch engaged); leaving it as is'
    } elseif ($cap -ge 110) {
        Invoke-Aws lambda put-function-concurrency --function-name $Name --reserved-concurrent-executions 5 | Out-Null
        Write-Note "account cap $cap; reserved concurrency set to 5"
    } else {
        Write-Note "account cap $cap is the hard ceiling; reserved concurrency not set"
    }

    # 10. Record and smoke test
    $deployed = [ordered]@{
        api_url = $ApiUrl
        region = $Region
        function = $Name
        image_tag = $ImageTag
        image_digest = $ImageDigest
        base_image_digest = $baseDigest
        memory_mb = [int]$fn.MemorySize
        timeout_s = [int]$fn.Timeout
        deployed_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    }
    $deployedPath = Join-Path $PSScriptRoot 'deployed.json'
    [System.IO.File]::WriteAllText($deployedPath, ($deployed | ConvertTo-Json) + "`n", [System.Text.UTF8Encoding]::new($false))
    Write-Step "Deployed: $ApiUrl"
    Write-Note "wrote $deployedPath"

    if (-not $SkipSmoke) {
        $smokeArgs = @{ Url = $ApiUrl }
        if ($SmokeHealthOnly) { $smokeArgs.HealthOnly = $true }
        & (Join-Path $PSScriptRoot 'smoke_test.ps1') @smokeArgs
    }
} finally {
    foreach ($f in $tempFiles) { Remove-Item -LiteralPath $f -ErrorAction SilentlyContinue }
}
