Write-Host "Checking Tailscale..." -ForegroundColor Cyan

$tailscalePath = "C:\Program Files\Tailscale\tailscale.exe"
$tailscaleGUI = "C:\Program Files\Tailscale\tailscale-ipn.exe"
$tailscaleIP = $null

function Get-TSStatus {
    try {
        return & $tailscalePath status --json 2>$null | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

if (Test-Path $tailscalePath) {
    $status = Get-TSStatus

    if (-not $status -or $status.BackendState -ne "Running") {
        Write-Host "Tailscale backend not ready, launching the Tailscale app..." -ForegroundColor Yellow

        if (Test-Path $tailscaleGUI) {
            if (-not (Get-Process tailscale-ipn -ErrorAction SilentlyContinue)) {
                Start-Process $tailscaleGUI
            }
        }

        $waited = 0
        while ($waited -lt 30) {
            Start-Sleep -Seconds 3
            $waited += 3
            $status = Get-TSStatus

            if ($status -and $status.BackendState -eq "Running") {
                break
            }

            Write-Host "  waiting for Tailscale... ($waited s)" -ForegroundColor DarkGray
        }
    }
    else {
        Write-Host "Tailscale is already connected." -ForegroundColor Green
    }

    if ($status -and $status.Self.TailscaleIPs) {
        $tailscaleIP = $status.Self.TailscaleIPs[0]
        Write-Host "Tailscale IP for this PC: $tailscaleIP" -ForegroundColor Green
    }
    else {
        Write-Host "Tailscale is not ready. Mac access may not work." -ForegroundColor Yellow
        Write-Host "Continuing because local access will still work." -ForegroundColor Yellow
    }
}
else {
    Write-Host "Tailscale executable not found, skipping check." -ForegroundColor Yellow
}

$tailscaleMonitor = $null

if (Test-Path $tailscalePath) {
    $tailscaleMonitor = Start-Job -Name "SupportAgentTailscaleMonitor" -ArgumentList @(
        $tailscalePath,
        $tailscaleGUI
    ) -ScriptBlock {
        param($cliPath, $guiPath)

        while ($true) {
            Start-Sleep -Seconds 300

            try {
                $currentStatus = & $cliPath status --json 2>$null | ConvertFrom-Json
            }
            catch {
                $currentStatus = $null
            }

            if ($currentStatus -and $currentStatus.BackendState -eq "Running") {
                continue
            }

            try {
                Start-Service -Name "Tailscale" -ErrorAction SilentlyContinue
            }
            catch {
                # The GUI and CLI attempts below may still restore the connection.
            }

            if (Test-Path $guiPath) {
                if (-not (Get-Process tailscale-ipn -ErrorAction SilentlyContinue)) {
                    Start-Process $guiPath
                }
            }

            & $cliPath up *>$null
        }
    }

    Write-Host "Tailscale will be checked every 5 minutes." -ForegroundColor Green
}

Write-Host "Freeing port 8080 if still held..." -ForegroundColor Cyan
wsl -d Ubuntu -- pkill -f llama-server 2>$null
Start-Sleep -Seconds 2

Write-Host "Updating WSL network proxy..." -ForegroundColor Cyan

$wslAddresses = (wsl -d Ubuntu -- hostname -I).Trim() -split "\s+"
$wslIP = $wslAddresses[0]

if (-not $wslIP) {
    Write-Host "Could not determine the WSL IP address." -ForegroundColor Red
    exit 1
}

netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0 *>$null
netsh interface portproxy delete v4tov4 listenport=3000 listenaddress=0.0.0.0 *>$null
netsh interface portproxy delete v4tov4 listenport=3001 listenaddress=0.0.0.0 *>$null

netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIP
netsh interface portproxy add v4tov4 listenport=3001 listenaddress=0.0.0.0 connectport=3001 connectaddress=$wslIP

Write-Host "Portproxy updated to WSL IP: $wslIP" -ForegroundColor Green

Write-Host "Starting project containers..." -ForegroundColor Cyan
wsl -d Ubuntu -- bash -lc "cd /home/saleh/Projects/multi-agent-teq-support && docker compose up -d"

Write-Host "Open WebUI: http://localhost:3001" -ForegroundColor Green
if ($tailscaleIP) {
    Write-Host "Open WebUI over Tailscale: http://${tailscaleIP}:3001" -ForegroundColor Green
}

Write-Host "Starting llama-server. Leave this window open." -ForegroundColor Cyan
try {
    wsl -d Ubuntu -- bash -lc "~/run-qwen38.sh"
}
finally {
    if ($tailscaleMonitor) {
        Stop-Job -Job $tailscaleMonitor -ErrorAction SilentlyContinue
        Remove-Job -Job $tailscaleMonitor -Force -ErrorAction SilentlyContinue
    }
}
