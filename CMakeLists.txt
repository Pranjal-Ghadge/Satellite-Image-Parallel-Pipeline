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

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(SOURCES
    src/main.cpp
    src/utils.cpp
    src/cpu_sequential.cpp
    src/cpu_openmp.cpp
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
)

if(USE_CUDA)
    target_link_libraries(parallel_pipeline PRIVATE CUDA::cudart)
    set_target_properties(parallel_pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()