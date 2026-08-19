# NEURAL_CPP

A modern C++ neural network and tensor library layout.

## Project Structure

```
NEURAL_CPP/
├── .vscode/          # VS Code editor configuration
├── build/            # CMake build outputs directory
├── include/          # Public header files
│   └── nn/
│       └── tensor.h  # Tensor class header implementation
├── src/              # Source files
├── tests/            # Test suite
│   └── tests.cpp     # Unit tests implementation
├── .clang-format     # Code formatting configuration
├── .gitignore        # Git ignore rules
├── CMakeLists.txt    # Top-level CMake build script
└── README.md         # Documentation
```

## Getting Started

### Prerequisites

- C++20 compliant compiler (GCC 10+, Clang 11+, or MSVC 2019+)
- CMake 3.20 or higher

### Building the Project

```bash
# Navigate into the project directory
cd NEURAL_CPP

# Configure CMake build files
cmake -B build -S .

# Build targets
cmake --build build
```

### Running Tests

```bash
# Run test executable directly
./build/neural_tests

# Or using ctest
ctest --test-dir build --output-on-failure
```
