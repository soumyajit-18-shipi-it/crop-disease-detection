param(
    [Parameter(Mandatory = $true)]
    [string]$OrganizationId,
    [string]$SecretsPath = ".deployment-secrets.json"
)

$ErrorActionPreference = "Stop"
$secrets = Get-Content -LiteralPath $SecretsPath -Raw | ConvertFrom-Json
if ($secrets.supabase_project_ref) {
    Write-Output "Supabase project already recorded: $($secrets.supabase_project_ref)"
    exit 0
}

$raw = & npx.cmd --yes supabase@latest projects create crop-disease `
    --org-id $OrganizationId `
    --db-password $secrets.supabase_db_password `
    --region ap-southeast-1 `
    --size nano `
    --output-format json `
    --yes
if ($LASTEXITCODE -ne 0) {
    throw "Supabase project creation failed. Review the sanitized CLI status separately."
}

$project = ($raw -join "`n") | ConvertFrom-Json
$projectRef = if ($project.ref) { $project.ref } elseif ($project.id) { $project.id } else { "" }
if (-not $projectRef) {
    throw "Supabase did not return a project reference."
}
$secrets.supabase_project_ref = $projectRef
$secrets | ConvertTo-Json | Set-Content -LiteralPath $SecretsPath -Encoding UTF8
Write-Output "Supabase project created: name=crop-disease ref=$projectRef region=ap-southeast-1 size=nano"
