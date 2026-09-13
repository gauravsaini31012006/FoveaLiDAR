@echo off
REM FoveaLiDAR - Run Tests
REM This script sets up the environment and runs the test suite

echo.
echo ========================================
echo FoveaLiDAR - Phase 0 Test Runner
echo ========================================
echo.

REM Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    py -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/upgrade dependencies
echo.
echo Installing dependencies from requirements.txt...
pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Run tests
echo.
echo ========================================
echo Running Tests - Phase 0 Foundation
echo ========================================
echo.

pytest tests\test_phase0_foundation.py -v --tb=short
if errorlevel 1 (
    echo.
    echo ERROR: Some tests failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo All Tests PASSED!
echo ========================================
echo.

pause
exit /b 0