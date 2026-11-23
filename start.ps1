$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$backend = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'
$python = Join-Path $backend 'venv\Scripts\python.exe'
$backendPort = 8000
$frontendPort = 5173

if (-not (Test-Path $python)) {
    throw "Backend environment not found. Create it and install backend requirements first. See README.md."
}
if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
    throw "Frontend dependencies not found. Run 'npm install' in the frontend folder first."
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw 'Node.js/npm is not installed or npm.cmd is not on PATH.'
}

function Test-LocalPort {
    param([int]$Port)

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connection = $client.ConnectAsync('127.0.0.1', $Port)
        return $connection.Wait(500) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Start-CommandWindow {
    param([string]$Command)

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
    Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit', '-EncodedCommand', $encoded)
}

$backendHealthy = $false
if (Test-LocalPort $backendPort) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$backendPort/health" -TimeoutSec 3
        $backendHealthy = $health.status -eq 'ok'
    } catch {
        throw "Port $backendPort is already in use, but the DeepFakeShield backend health check failed. Close the other service and run start.bat again."
    }
    if (-not $backendHealthy) {
        throw "A service is using port $backendPort but is not the expected backend. Close it and run start.bat again."
    }
    Write-Host 'Backend is already healthy.'
} else {
    $backendPath = $backend.Replace("'", "''")
    $pythonPath = $python.Replace("'", "''")
    $backendCommand = "Set-Location -LiteralPath '$backendPath'; & '$pythonPath' -m uvicorn main:app --reload --host 127.0.0.1 --port $backendPort"
    Start-CommandWindow $backendCommand

    Write-Host 'Starting backend and loading detectors. This can take a few minutes on first launch.'
    $lastHealthError = ''
    for ($attempt = 0; $attempt -lt 150; $attempt++) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$backendPort/health" -TimeoutSec 3
            if ($health.status -eq 'ok') {
                $backendHealthy = $true
                break
            }
            $lastHealthError = 'Backend returned an unexpected health response.'
        } catch {
            $lastHealthError = $_.Exception.Message
        }
        Start-Sleep -Seconds 2
    }
    if (-not $backendHealthy) {
        throw "Backend did not become ready within five minutes. Check the backend terminal. Last health-check error: $lastHealthError"
    }
    Write-Host 'Backend is ready.'
}

if (Test-LocalPort $frontendPort) {
    Write-Host "Frontend is already running at http://127.0.0.1:$frontendPort"
} else {
    $frontendPath = $frontend.Replace("'", "''")
    $frontendCommand = "Set-Location -LiteralPath '$frontendPath'; npm.cmd run dev"
    Start-CommandWindow $frontendCommand

    $frontendReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Test-LocalPort $frontendPort) {
            $frontendReady = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $frontendReady) {
        throw 'Frontend did not start. Check the frontend terminal for errors.'
    }
    Write-Host 'Frontend is ready.'
}

Start-Process "http://127.0.0.1:$frontendPort"
Write-Host 'DeepFakeShield is running. Keep both server terminal windows open.'
