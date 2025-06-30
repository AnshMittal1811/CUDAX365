$dxc = "dxc.exe"
if (-not (Get-Command $dxc -ErrorAction SilentlyContinue)) {
  Write-Host "dxc.exe not found"
  exit 1
}

& $dxc -T lib_6_3 -Fo raygen.dxil raygen.hlsl
