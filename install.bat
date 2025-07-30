@echo off
REM Quick Installation Script for Indian License Plate Recognition System (Windows)

echo 🚗 Indian License Plate Recognition System - Quick Setup
echo =========================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.7+ first.
    echo 📥 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found

REM Ask about virtual environment
set /p create_venv="🤔 Create virtual environment? (y/n): "
if /i "%create_venv%"=="y" (
    echo 📦 Creating virtual environment...
    python -m venv venv

    REM Activate virtual environment
    call venv\Scripts\activate.bat

    echo ✅ Virtual environment created and activated
)

REM Install requirements
echo 📦 Installing Python packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Error installing packages. Please check your internet connection.
    pause
    exit /b 1
)

echo ✅ All packages installed successfully

REM Run setup script
echo ⚙️ Running setup script...
python setup.py

if %errorlevel% neq 0 (
    echo ❌ Setup script failed
    pause
    exit /b 1
)

echo ✅ Setup completed successfully
echo.
echo 🎉 Installation completed successfully!
echo.
echo 🚀 Quick Start:
echo    python demo.py                              # Run demo
echo    python indian_lpr_system.py -i image.jpg    # Process image
echo    python batch_process.py -i images/          # Batch process
echo.
echo 📚 Check README.md for detailed usage instructions
echo.
pause
