cmake_minimum_required(VERSION 3.20)
project(NEURAL_CPP VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for LSP and tooling support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warning flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core library target
add_library(neural_cpp STATIC src/nn/tensor.cpp)
target_include_directories(neural_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Unit testing
enable_testing()
add_executable(neural_tests tests/tests.cpp)
target_link_libraries(neural_tests PRIVATE neural_cpp)
add_test(NAME neural_tests COMMAND neural_tests)
