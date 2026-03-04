# Batch kill Python processes (e.g. runaway multi_bgemm subprocesses).
# Run in PowerShell: .\kill_python_processes.ps1
# Or: powershell -ExecutionPolicy Bypass -File kill_python_processes.ps1

$procs = Get-Process -Name python*, py -ErrorAction SilentlyContinue
if ($procs) {
    $count = ($procs | Measure-Object).Count
    Write-Host "Found $count Python process(es). Killing..."
    $procs | Stop-Process -Force
    Write-Host "Done."
} else {
    Write-Host "No Python processes found."
}
