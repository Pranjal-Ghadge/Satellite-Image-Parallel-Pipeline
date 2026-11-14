# AAKASHNETRA : The Eye in the Sky
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
The native project also requires GDAL 3.0 or newer, including its C++ development files and CMake package configuration. For the documented Windows/MSVC setup, use the x64 GDAL 3.12.1 development package and matching runtime.

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

## Validate an aligned Sentinel-1 GeoTIFF pair

The native loader reads one band from each GeoTIFF into Float64 physical values and retains
each source's CRS, affine geotransform, data type, unit, scale, offset, and NoData value.
GDAL scale and offset are applied to valid samples. Each raster also exposes a valid-pixel
mask; NoData, GDAL mask-band exclusions, and non-finite input or converted samples are set
to zero and marked invalid so later classification can exclude them. Complex-valued SAR
bands are rejected with an actionable error; provide real-valued backscatter data.

The pair must have the same dimensions, equivalent CRS, pixel size, grid orientation, and
pixel-aligned origin. It reports mismatches instead of reprojecting or resizing either
image. Band numbers are one-based; band 1 is the default.

```powershell
.\build-msvc\parallel_pipeline.exe --validate-pair .\data\pre_flood.tif .\data\post_flood.tif
.\build-msvc\parallel_pipeline.exe --validate-pair .\data\pre_flood.tif .\data\post_flood.tif --band 2
```

Once a matching x64 GDAL development package is installed, open an x64 MSVC developer
command prompt, change to the project root, and set `GDAL_DIR` to the directory containing
that installation's `GDALConfig.cmake`. For example:

```bat
set "GDAL_DIR=C:\Program Files\GDAL\lib\cmake\gdal"
cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DOpenCV_DIR=C:\Users\HP\Downloads\opencv\build "-DGDAL_DIR=%GDAL_DIR%" -DUSE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-msvc --config Release --parallel
ctest --test-dir build-msvc -C Release --output-on-failure
```

`GDAL_DIR` is the package-config directory, not just the folder containing `gdal.h`.
If the installed SDK places `GDALConfig.cmake` elsewhere, set `GDAL_DIR` to that directory
instead. GDAL 3.5+ uses its installed CMake package configuration; CMake falls back to its
`FindGDAL` module for older GDAL 3.x installations.
