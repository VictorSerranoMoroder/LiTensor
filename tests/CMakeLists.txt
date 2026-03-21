include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)

FetchContent_MakeAvailable(googletest)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(CoreTests
    CoreTests.cpp
)

target_link_libraries(CoreTests
    PRIVATE
        Featherlight
        GTest::gtest_main
        CUDA::cudart
)

# Allow direct third_party includes in tests
target_include_directories(CoreTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

include(GoogleTest)
gtest_discover_tests(CoreTests)