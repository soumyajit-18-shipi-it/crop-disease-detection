param(
    [string]$Path = ".deployment-secrets.json"
)

$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $Path) {
    Write-Output "Production deployment secrets are already initialized."
    exit 0
}

function New-UrlSafeSecret([int]$ByteCount) {
    $bytes = New-Object byte[] $ByteCount
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
}

$payload = [ordered]@{
    supabase_db_password = New-UrlSafeSecret 48
    auth_secret = New-UrlSafeSecret 64
    supabase_project_ref = ""
}
$payload | ConvertTo-Json | Set-Content -LiteralPath $Path -Encoding UTF8
Write-Output "Production deployment secrets initialized in an ignored local file."
