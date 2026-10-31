# Satellite-Image-Parallel-Pipeline
High-Performance Parallel Pipeline for Real-Time Satellite Image Filtering and Edge Detection Using Hybrid OpenMP CPU and CUDA GPU Architectures.

## Option A: MinGW CPU/OpenMP build

Run these commands from PowerShell in the project root. This configuration does not use CUDA or Visual Studio's `cl.exe`.

```powershell
# Remove the old build and any cached generator/compiler selection.
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
# Configure with MinGW and explicitly disable CUDA.
cmake -S . -B build -G "MinGW Makefiles" -DUSE_CUDA=OFF

# Compile.
cmake --build build --parallel
```

Run the executable from the project root so the default input/output paths resolve correctly:

```powershell
.\build\parallel_pipeline.exe
```

Requirements: MinGW GCC/G++, CMake, OpenMP, and OpenCV installed for MinGW. Do not place this build in a Visual Studio Developer Command Prompt with a stale Ninja generator/cache.

## Option B: Visual Studio 2022 CUDA build

1. Start the standard Windows PowerShell or Command Prompt.
2. Launch a Visual Studio 2022 x64 Developer Command Prompt. For the Community edition:

```bat
"%comspec%" /k ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64"
```

If VS 2022 Professional or Enterprise is installed, replace `Community` with the matching edition. Alternatively, search the Start menu for `x64 Native Tools Command Prompt for VS 2022`.

3. In that Developer Prompt, change to the project root and run:

```bat
cd /d "C:\Users\PRANJAL GHADGE\Satellite-Image-Parallel-Pipeline"
if exist build-gpu rmdir /s /q build-gpu

cmake -S . -B build-gpu -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON
cmake --build build-gpu --config Release --parallel
build-gpu\Release\parallel_pipeline.exe
```

Requirements: Visual Studio 2022 with the Desktop development with C++ workload, the NVIDIA CUDA Toolkit, and an OpenCV build compatible with MSVC/x64. Run CMake from the Developer Prompt so `cl.exe` is available to both MSVC and `nvcc`.

## Rebuild the CUDA executable for the Web UI

PowerShell:

```powershell
& .\scripts\msvc-ninja.cmd `
    "C:\Users\PRANJAL GHADGE\Satellite-Image-Parallel-Pipeline" `
    configure-build ON
```

To configure CUDA only:

```powershell
& .\scripts\msvc-ninja.cmd `
    "C:\Users\PRANJAL GHADGE\Satellite-Image-Parallel-Pipeline" `
    configure ON
```

After a successful CUDA build, start the dashboard:

```powershell
python run_ui.py
```
