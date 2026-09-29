# Run the offline eval — reads OpenAI key from the backend .env
# Usage:  .\eval_harness\run_offline.ps1

$backendEnv = Join-Path $PSScriptRoot "..\AI-Voice-Agent-Backend-main\.env"
if (Test-Path $backendEnv) {
    Get-Content $backendEnv | ForEach-Object {
        if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)=(.*)$') {
            $k = $Matches[1]; $v = $Matches[2].Trim()
            if (-not [System.Environment]::GetEnvironmentVariable($k)) {
                [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
            }
        }
    }
    Write-Host "[env] loaded from $backendEnv"
}

$script = Join-Path $PSScriptRoot "eval_offline.py"
python $script
