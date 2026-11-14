
cmake_minimum_required(VERSION 3.12)

project(SequentialSatellitePipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Users/HP/Downloads/opencv/build")

find_package(OpenCV REQUIRED)

add_executable(sequential_pipeline main.cpp)

target_include_directories(sequential_pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(sequential_pipeline PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(sequential_pipeline PRIVATE
        $<$<CONFIG:Release>:/O2>
    )
else()
    target_compile_options(sequential_pipeline PRIVATE
        $<$<CONFIG:Release>:-O3>
    )
endif()
