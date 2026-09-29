<#
.SYNOPSIS
  Setup and start the full Highland Greenz eval stack.
  Run from:  D:\real-estate-UI\AI-voice agent

.USAGE
  .\eval_harness\setup_and_start.ps1 -PgPassword "your-pg-password"

.PARAMETERS
  -PgPassword   PostgreSQL superuser password
  -PgUser       PostgreSQL superuser (default: postgres)
  -SkipMigrate  Skip alembic upgrade (if already migrated)
  -SkipSeed     Skip seed script (if already seeded)
#>
param(
    [Parameter(Mandatory=$false)][string]$PgPassword = "",
    [string]$PgUser = "postgres",
    [switch]$SkipMigrate,
    [switch]$SkipSeed
)

$ErrorActionPreference = "Stop"
$Root    = "D:\real-estate-UI\AI-voice agent"
$Backend = "$Root\AI-Voice-Agent-Backend-main"
$Core    = "$Backend\core"
$UI      = "$Root\AI-Voice-Agent-UI-main"
$PgBin   = "C:\Program Files\PostgreSQL\18\bin"
$Redis   = "C:\Users\$env:USERNAME\AppData\Local\Temp\redis"

# ── 1. Start Redis ──────────────────────────────────────────────────────────
Write-Host "`n[redis] checking…"
$redisPing = & "$Redis\redis-cli.exe" ping 2>&1
if ($redisPing -ne "PONG") {
    Write-Host "[redis] starting…"
    Start-Process "$Redis\redis-server.exe" -ArgumentList "--port 6379" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    $redisPing = & "$Redis\redis-cli.exe" ping 2>&1
    if ($redisPing -ne "PONG") { Write-Host "[redis] FAILED to start"; exit 1 }
}
Write-Host "[redis] OK"

# ── 2. Create DB user + database ───────────────────────────────────────────
if ($PgPassword) {
    Write-Host "`n[postgres] creating user aivoiceagent…"
    $env:PGPASSWORD = $PgPassword
    $cmds = @(
        "CREATE USER aivoiceagent WITH PASSWORD 'aivoiceagent' CREATEDB;",
        "CREATE DATABASE aivoiceagent OWNER aivoiceagent;"
    )
    foreach ($sql in $cmds) {
        & "$PgBin\psql.exe" -U $PgUser -h 127.0.0.1 -p 5432 -c $sql 2>&1 | Out-Null
    }
    # Enable pgvector extension
    & "$PgBin\psql.exe" -U $PgUser -h 127.0.0.1 -p 5432 -d aivoiceagent `
        -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1 | Out-Null
    Write-Host "[postgres] user + database + pgvector ready"
} else {
    Write-Host "[postgres] skipping DB creation (no -PgPassword given)"
}

# ── 3. Load backend env ─────────────────────────────────────────────────────
Write-Host "`n[env] loading from $Backend\.env"
Get-Content "$Backend\.env" | ForEach-Object {
    if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim(), "Process")
    }
}

# ── 4. Run migrations ───────────────────────────────────────────────────────
if (-not $SkipMigrate) {
    Write-Host "`n[alembic] running migrations…"
    Push-Location $Core
    python -m alembic upgrade head
    Pop-Location
}

# ── 5. Seed ─────────────────────────────────────────────────────────────────
if (-not $SkipSeed) {
    Write-Host "`n[seed] seeding tenant + admin user…"
    Push-Location $Core
    python scripts\seed.py --tenant "DSR" --email "admin@advora.ai" --password "Admin@123"
    python scripts\seed_dsr.py 2>&1 | Select-Object -Last 5
    Pop-Location
}

# ── 6. Upload knowledge base ────────────────────────────────────────────────
Write-Host "`n[knowledge] uploading knowledge base…"
Push-Location "$Root\eval_harness"
python build_knowledge.py
Pop-Location

# ── 7. Start backend (background) ──────────────────────────────────────────
Write-Host "`n[backend] starting uvicorn on http://localhost:8000"
Push-Location $Core
$backendJob = Start-Job -ScriptBlock {
    param($coreDir)
    Set-Location $coreDir
    $env:DATABASE_URL = "postgresql+asyncpg://aivoiceagent:aivoiceagent@localhost:5432/aivoiceagent"
    $env:REDIS_URL = "redis://localhost:6379/0"
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1
} -ArgumentList $Core
Pop-Location

Write-Host "[backend] waiting for startup…"
$retries = 0
do {
    Start-Sleep -Seconds 3
    $retries++
    try {
        $h = Invoke-WebRequest "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
        Write-Host "[backend] UP: $($h.Content)"
        break
    } catch { Write-Host "  waiting… ($retries/15)" }
} while ($retries -lt 15)

# ── 8. Start UI dev server (background) ────────────────────────────────────
Write-Host "`n[ui] starting Vite on http://localhost:3000"
$uiJob = Start-Job -ScriptBlock {
    param($uiDir)
    Set-Location $uiDir
    npm run dev -- --port 3000 2>&1
} -ArgumentList $UI

Write-Host "[ui] waiting for startup…"
Start-Sleep -Seconds 8

Write-Host "`n✅ Stack ready:"
Write-Host "   Backend : http://localhost:8000"
Write-Host "   UI      : http://localhost:3000"
Write-Host "`nRun the eval:"
Write-Host "   python eval_harness\eval_runner.py"
Write-Host ""
Write-Host "(press Ctrl+C to stop, then: Stop-Job `$backendJob, `$uiJob)"
