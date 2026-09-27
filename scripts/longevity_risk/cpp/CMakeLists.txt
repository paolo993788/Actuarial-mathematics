# Standalone build of the header-only C++ engines: unit tests.
#
# The Python extension is built by setup.py (pybind11); this project compiles the same headers without Python,
# with strict warnings and optional sanitizers, so that the C++ code is checked on its own.
#
#   cmake -S scripts/longevity_risk/cpp -B build/cpp -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   cmake --build build/cpp
#   ctest --test-dir build/cpp --output-on-failure
#
# Sanitizers: -DLR_SANITIZE=address,undefined (memory errors and undefined behaviour) or -DLR_SANITIZE=thread
# (data races in the parallel loops). They cannot be combined in one build.

cmake_minimum_required(VERSION 3.16)
project(longevity_risk_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LR_WERROR "Treat compiler warnings as errors" ON)
set(LR_SANITIZE "" CACHE STRING "Comma-separated sanitizers passed to -fsanitize (for example address,undefined)")

find_package(Threads REQUIRED)

add_library(lr_engine INTERFACE)
target_include_directories(lr_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lr_engine INTERFACE Threads::Threads)

if(MSVC)
  target_compile_options(lr_engine INTERFACE /W4 /permissive- $<$<BOOL:${LR_WERROR}>:/WX>)
else()
  target_compile_options(lr_engine INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
                         -Wold-style-cast -Wnull-dereference -Wdouble-promotion
                         $<$<BOOL:${LR_WERROR}>:-Werror>)
  # Bounds-checked standard containers in debug-like builds.
  target_compile_definitions(lr_engine INTERFACE $<$<NOT:$<CONFIG:Release>>:_GLIBCXX_ASSERTIONS>)
endif()

if(LR_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "LR_SANITIZE is supported with GCC and Clang only")
  endif()
  target_compile_options(lr_engine INTERFACE -fsanitize=${LR_SANITIZE} -fno-omit-frame-pointer
                         -fno-sanitize-recover=all)
  target_link_options(lr_engine INTERFACE -fsanitize=${LR_SANITIZE})
endif()

enable_testing()

add_executable(core_tests tests/test_core.cpp)
target_link_libraries(core_tests PRIVATE lr_engine)
add_test(NAME core_tests COMMAND core_tests)

