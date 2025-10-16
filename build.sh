#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build .

# Report success
echo "Build complete! The compiler executable is located at: $(pwd)/simple_compiler"
