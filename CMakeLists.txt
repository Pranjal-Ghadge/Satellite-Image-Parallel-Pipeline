cmake_minimum_required(VERSION 3.18)
project(SatelliteImageParallelPipeline LANGUAGES CXX)

option(USE_CUDA "Enable CUDA GPU acceleration" OFF)
option(ENABLE_CUDA "Compatibility alias for USE_CUDA" OFF)

if(ENABLE_CUDA)
    set(USE_CUDA ON CACHE BOOL "Enable CUDA GPU acceleration" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT OpenCV_DIR AND EXISTS "C:/Users/HP/Downloads/opencv/build")
    set(OpenCV_DIR "C:/Users/HP/Downloads/opencv/build")
endif()

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

find_package(GDAL CONFIG QUIET)
if(GDAL_FOUND)
    if(GDAL_VERSION VERSION_LESS "3.0")
        message(FATAL_ERROR "GDAL 3.0 or newer is required; found GDAL ${GDAL_VERSION}.")
    endif()
else()
    find_package(GDAL 3.0 REQUIRED MODULE)
endif()

set(SOURCES
    src/main.cpp
    src/utils.cpp
    src/cpu_sequential.cpp
    src/cpu_openmp.cpp
    src/satellite_dataset.cpp
)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND SOURCES src/gpu_cuda.cu)
endif()

add_executable(parallel_pipeline ${SOURCES})

target_include_directories(parallel_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(parallel_pipeline PRIVATE
    $<$<BOOL:${USE_CUDA}>:USE_CUDA>
)

target_link_libraries(parallel_pipeline PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    GDAL::GDAL
)

include(CTest)
if(BUILD_TESTING)
    add_executable(satellite_dataset_tests
        tests/satellite_dataset_test.cpp
        src/satellite_dataset.cpp
    )
    target_include_directories(satellite_dataset_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(satellite_dataset_tests PRIVATE
        ${OpenCV_LIBS}
        GDAL::GDAL
    )
    add_test(NAME satellite_dataset_tests COMMAND satellite_dataset_tests)
endif()

if(USE_CUDA)
    target_link_libraries(parallel_pipeline PRIVATE CUDA::cudart)
    set_target_properties(parallel_pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()