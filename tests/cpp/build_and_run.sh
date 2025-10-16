#!/bin/bash

# Build and run C++ tests for Jam compiler

echo "Building C++ Tests..."
echo "========================"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build .

if [ $? -eq 0 ]; then
    echo ""
    echo "Running C++ Tests..."
    echo "======================="
    ./jam_tests
else
    echo "Build failed!"
    exit 1
fi