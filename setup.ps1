# PowerShell script for setting up the Smart Goggles prototype on Windows
Write-Host "Setting up Smart Goggles Prototype..." -ForegroundColor Green

# Create virtual environment if it doesn't exist
if (-not (Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install required packages
Write-Host "Installing required packages (this may take a while)..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create .env file if it doesn't exist
if (-not (Test-Path -Path ".env")) {
    Write-Host "Creating .env file from example..." -ForegroundColor Yellow
    Copy-Item -Path ".env.example" -Destination ".env"
    Write-Host "Please edit the .env file to set up your API keys and configuration" -ForegroundColor Cyan
}

# Create known_faces directory if it doesn't exist
if (-not (Test-Path -Path "known_faces")) {
    Write-Host "Creating known_faces directory..." -ForegroundColor Yellow
    New-Item -Path "known_faces" -ItemType Directory | Out-Null
    Write-Host "Add face images to the known_faces directory (name each file with the person's name)" -ForegroundColor Cyan
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To run the Smart Goggles prototype: python main.py" -ForegroundColor Green
