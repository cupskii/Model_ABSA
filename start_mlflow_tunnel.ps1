<#
.SYNOPSIS
    Expose MLflow lokal (docker-compose, port 5000) ke internet lewat Cloudflare
    Quick Tunnel, supaya bisa dipakai sebagai MLFLOW_TRACKING_URI dari Google Colab
    (atau environment remote lain yang tidak bisa akses localhost laptop ini).

.DESCRIPTION
    - Cek container `mlflow` docker-compose sudah jalan (opsional: -StartMlflow
      buat nyalain otomatis lewat `docker compose up -d`).
    - Jalankan `cloudflared tunnel --url <MlflowUrl>`, tunggu sampai URL publik
      *.trycloudflare.com muncul di log, lalu tampilkan (dan salin ke clipboard).
    - URL ini BERSIFAT SEMENTARA — mati kalau script dihentikan (Ctrl+C) atau
      laptop restart. Buat ulang lalu update MLFLOW_TRACKING_URI di notebook
      Colab / .env / Modal Secret setiap kali dibuat ulang.

.PARAMETER MlflowUrl
    Alamat MLflow lokal yang mau di-tunnel. Default http://localhost:5000
    (sesuai docker-compose.yaml dan configs/*.yaml saat ini).

.PARAMETER StartMlflow
    Kalau di-set, jalankan `docker compose up -d mlflow postgres` dulu sebelum
    membuka tunnel.

.EXAMPLE
    .\start_mlflow_tunnel.ps1

.EXAMPLE
    .\start_mlflow_tunnel.ps1 -StartMlflow

.EXAMPLE
    .\start_mlflow_tunnel.ps1 -MlflowUrl http://localhost:5001
#>

param(
    [string]$MlflowUrl = "http://localhost:5000",
    [switch]$StartMlflow
)

$ErrorActionPreference = "Stop"

if ($StartMlflow) {
    Write-Host "Menjalankan 'docker compose up -d mlflow postgres' ..."
    docker compose up -d mlflow postgres
    if (-not $?) {
        Write-Error "docker compose gagal jalan. Cek instalasi Docker Desktop dan docker-compose.yaml."
        exit 1
    }
}

try {
    $probe = Invoke-WebRequest -Uri $MlflowUrl -TimeoutSec 5 -UseBasicParsing
    Write-Host "MLflow terjangkau di $MlflowUrl (HTTP $($probe.StatusCode))."
} catch {
    Write-Warning "MLflow belum terjangkau di $MlflowUrl. Pastikan sudah jalan (docker ps, atau jalankan script ini dengan -StartMlflow)."
}

$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflared) {
    Write-Error "cloudflared tidak ditemukan di PATH. Install dari https://github.com/cloudflare/cloudflared/releases lalu coba lagi."
    exit 1
}

$logOut = Join-Path $env:TEMP "cloudflared_mlflow_out_$PID.log"
$logErr = Join-Path $env:TEMP "cloudflared_mlflow_err_$PID.log"

Write-Host ""
Write-Host "Membuka Cloudflare Quick Tunnel ke $MlflowUrl ..."

$proc = Start-Process -FilePath $cloudflared.Source `
    -ArgumentList @("tunnel", "--url", $MlflowUrl) `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $logOut -RedirectStandardError $logErr

$publicUrl = $null
$deadline = (Get-Date).AddSeconds(30)
$pattern = 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com'

while (-not $publicUrl -and (Get-Date) -lt $deadline -and -not $proc.HasExited) {
    Start-Sleep -Milliseconds 500
    foreach ($f in @($logErr, $logOut)) {
        if (-not $publicUrl -and (Test-Path $f)) {
            $match = Select-String -Path $f -Pattern $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($match) { $publicUrl = $match.Matches[0].Value }
        }
    }
}

if (-not $publicUrl) {
    Write-Error "Gagal mendapatkan URL tunnel dalam 30 detik. Isi log:"
    Get-Content $logErr, $logOut -ErrorAction SilentlyContinue
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    exit 1
}

Write-Host ""
Write-Host "==============================================================="
Write-Host " MLflow tunnel URL : $publicUrl"
Write-Host "==============================================================="
Write-Host " Paste ke variabel MLFLOW_TRACKING_URI di notebook Colab"
Write-Host " (cell '5. Muat konfigurasi eksperimen & mulai MLflow run')."
Write-Host " URL ini sementara -- mati kalau script ini dihentikan."
Write-Host "==============================================================="

try {
    Set-Clipboard -Value $publicUrl
    Write-Host " (URL sudah disalin ke clipboard)"
} catch {
    # Set-Clipboard bisa gagal di sesi non-interaktif -- abaikan saja.
}

Write-Host ""
Write-Host "Tunnel jalan di background process PID $($proc.Id). Tekan Ctrl+C untuk menghentikan."

try {
    Wait-Process -Id $proc.Id
} finally {
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $logOut, $logErr -ErrorAction SilentlyContinue
}
