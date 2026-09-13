@echo off
REM FoveaLiDAR - Setup
REM This script performs initial setup (clean venv and dependencies)

echo.
echo ========================================
echo FoveaLiDAR - Phase 0 Setup
echo ========================================
echo.

REM Check if Python 3.11 is installed
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 is not installed or not in PATH
    echo Please install Python 3.11 from https://www.python.org/
    echo Or use: winget install -e --id Python.Python.3.11
    pause
    exit /b 1
)

echo Checking Python version...
py -3.11 --version

REM Remove existing virtual environment if requested
if exist "venv\" (
    echo.
    echo Virtual environment already exists.
    set /p CLEAN="Remove and recreate? (y/n): "
    if /i "!CLEAN!"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q venv
    )
)

REM Create virtual environment
if not exist "venv\" (
    echo.
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo Upgrading pip...
py -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Trying with verbose output for debugging...
    pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run Application:  double-click run_app.bat
echo   2. Run Tests:        double-click run_tests.bat
echo.
echo Or from command line:
echo   - venv\Scripts\activate.bat
echo   - python main.py
echo   - pytest tests\ -v
echo.

pause
exit /b 0