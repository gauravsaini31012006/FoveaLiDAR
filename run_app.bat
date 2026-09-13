@echo off
REM FoveaLiDAR - Main Control Center
setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

:menu
cls
echo.
echo ============================================================================
echo              FoveaLiDAR - Main Control Center (Phases 0-10)
echo ============================================================================
echo.
echo [TESTS]
echo 1. Run all tests (Phases 0-10: 0,1,2,3,4,5,6,7,8,9,10)
echo 2. Phase 0 only (6 tests)
echo 3. Phase 1 only (13 tests)
echo 4. Phase 1+2 (10 tests)
echo 5. Phase 3 only (22 tests)
echo 6. Phase 4 only (6 tests - Mode A Uniform 2.5D Map)
echo 7. Phase 5 only (5 tests - Mode B Distance-Adaptive Map)
echo 8. Phase 6 only (22 tests - Mode C Dynamic Foveal Controller)
echo 9. Phase 7+8 only (2 tests - Semantic AI + Detection/Tracking)
echo 10. Phase 9+10 only (Compute Scheduler + Benchmark/Demo)
echo.
echo [VISUALIZATION]
echo 11. GUI: Full FoveaLiDAR Dashboard (Phases 0-10)
echo 12. Launch 3D Foveation + Semantic AI View (Phases 3-8)
echo 13. Launch 2.5D Map Modes A, B, C + Semantic AI + Tracking (Phases 4-10)
echo.
echo [DATA PROCESSING]
echo 14. Process KITTI data with Foveation (233 frames)
echo.
echo [INFO]
echo 15. Project status
echo 16. Exit
echo.
set /p choice="Select option (1-16): "

if "%choice%"=="1" goto test_all
if "%choice%"=="2" goto test_phase0
if "%choice%"=="3" goto test_phase1
if "%choice%"=="4" goto test_phase12
if "%choice%"=="5" goto test_phase3
if "%choice%"=="6" goto test_phase4
if "%choice%"=="7" goto test_phase5
if "%choice%"=="8" goto test_phase6
if "%choice%"=="9" goto test_phase78
if "%choice%"=="10" goto test_phase910
if "%choice%"=="11" goto launch_gui
if "%choice%"=="12" goto launch_phase3
if "%choice%"=="13" goto launch_phase4
if "%choice%"=="14" goto process_data
if "%choice%"=="15" goto show_info
if "%choice%"=="16" goto exit_app
goto menu

:test_all
cls
echo.
echo Running ALL TESTS (Phases 0 through 10)...
echo.
call .\venv\Scripts\python -m pytest tests/ -v --tb=short
echo.
pause
goto menu

:test_phase0
cls
echo.
echo PHASE 0 - FOUNDATION (6 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase0_foundation.py -v --tb=short
echo.
pause
goto menu

:test_phase1
cls
echo.
echo PHASE 1 - KITTI ENGINE (13 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase1_kitti.py -v --tb=short
echo.
pause
goto menu

:test_phase12
cls
echo.
echo PHASE 1+2 - RAW KITTI + PREPROCESSING (10 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase1_2_raw_and_preprocessing.py -v --tb=short
echo.
pause
goto menu

:test_phase3
cls
echo.
echo PHASE 3 - FOVEATION + ROI (22 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase3_foveation.py tests/test_phase3_autonomous_3d.py -v --tb=short
echo.
pause
goto menu

:test_phase4
cls
echo.
echo PHASE 4 - MODE A UNIFORM 2.5D MAP + GEOMETRY (6 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase4_uniform_map.py -v --tb=short
echo.
pause
goto menu

:test_phase5
cls
echo.
echo PHASE 5 - MODE B DISTANCE-ADAPTIVE 2.5D MAP (5 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase5_adaptive_map.py -v --tb=short
echo.
pause
goto menu

:test_phase6
cls
echo.
echo PHASE 6 - MODE C DYNAMIC FOVEAL CONTROLLER (22 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase6_foveal_controller.py -v --tb=short
echo.
pause
goto menu

:test_phase78
cls
echo.
echo PHASE 7+8 - SEMANTIC AI + DETECTION/TRACKING (2 tests)
echo.
call .\venv\Scripts\python -m pytest tests/test_phase7_8_perception.py -v --tb=short
echo.
pause
goto menu

:test_phase910
cls
echo.
echo PHASE 9+10 - COMPUTE SCHEDULER + BENCHMARK/DEMO
echo.
call .\venv\Scripts\python -m pytest tests/test_phase9_10_compute.py -v --tb=short
echo.
pause
goto menu

:launch_gui
cls
echo.
echo Launching FoveaLiDAR GUI (Phases 0-10: Compute-Aware + Benchmark + Demo)...
echo.
call .\venv\Scripts\python main.py
goto menu

:launch_phase3
cls
echo.
echo Launching Phases 3-8 Premium Foveation Visualizer...
echo (Real-time 3D + Camera View + Semantic AI + Tracking)
echo.
call .\venv\Scripts\python main.py
echo.
echo Select "3D Foveation + Semantic AI (Phases 3-8)" tab in the application
echo.
pause
goto menu

:launch_phase4
cls
echo.
echo Launching Phases 4-10: 2.5D Map (Modes A, B, C + Semantic AI + Tracking)...
echo (5 cm Near / 50 cm Far Adaptive and Uniform 5 cm Baseline)
echo.
call .\venv\Scripts\python main.py
echo.
echo Select "2.5D Map: Mode A, B ^& C + Semantic AI + Tracking (Phases 4-10)" tab
echo.
pause
goto menu

:process_data
cls
echo.
echo Processing your KITTI data with Foveation...
echo.
call .\venv\Scripts\python _process_with_foveation.py
echo.
pause
goto menu

:show_info
cls
echo.
call .\venv\Scripts\python _show_info.py
echo.
pause
goto menu

:exit_app
exit /b 0
