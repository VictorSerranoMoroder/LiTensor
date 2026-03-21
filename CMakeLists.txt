cmake_minimum_required(VERSION 3.18)
project(VictorNet LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


option(FEATHERLIGHT_ENABLE_TRACY "Enable Tracy profiling" OFF)
option(FEATHERLIGHT_BUILD_TESTS "Build tests" ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)
# Let cuda select the correct drivers
set(CUDA_DRIVER_LIBRARY "/run/opengl-driver/lib/libcuda.so")
# Let CMake pick up headers automatically via nvcc
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})

find_package(CUDAToolkit REQUIRED)

add_library(Featherlight INTERFACE)

# Include directories
target_include_directories(Featherlight
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

if(FEATHERLIGHT_ENABLE_TRACY)
    add_library(tracy
        third_party/tracy/TracyClient.cpp
    )

    target_include_directories(tracy PUBLIC third_party/tracy)
    target_compile_definitions(tracy PUBLIC TRACY_ENABLE)

    # Link as INTERFACE → propagates to users
    target_link_libraries(my_library INTERFACE tracy)

    target_compile_definitions(my_library INTERFACE MYLIB_ENABLE_TRACY)
endif()

if(FEATHERLIGHT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()




