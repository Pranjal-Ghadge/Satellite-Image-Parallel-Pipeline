@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~1"
set "ACTION=%~2"
set "ENABLE_CUDA=%~3"

if "%PROJECT_DIR%"=="" set "PROJECT_DIR=%CD%"
if "%ACTION%"=="" set "ACTION=configure"
if "%ENABLE_CUDA%"=="" set "ENABLE_CUDA=OFF"

set "VS_PATH="

:: Use quotes properly for vswhere command
for /f "usebackq tokens=*" %%i in (`^""%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath^"`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo ERROR: Visual Studio installation not found.
    exit /b 1
)

if "%ENABLE_CUDA%"=="ON" (
    call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
) else (
    call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
)

cd /d "%PROJECT_DIR%"

if "%ACTION%"=="configure" (
    if exist build-ninja rmdir /s /q build-ninja
    mkdir build-ninja
    cd build-ninja
    if "%ENABLE_CUDA%"=="ON" (
        cmake -G "Ninja" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DENABLE_CUDA=ON -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" ..
    ) else (
        cmake -G "Ninja" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DENABLE_CUDA=OFF -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" ..
    )
) else if "%ACTION%"=="build" (
    cd build-ninja
    cmake --build .
)