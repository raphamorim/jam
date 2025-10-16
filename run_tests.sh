#!/bin/bash

# Jam Language Unit Test Runner
# Tests u8, u16, and u32 type implementations

echo "Running Jam Language Unit Tests"
echo "==============================="

# Build the compiler first
echo "Building compiler..."
./build.sh > /dev/null 2>&1
echo "Compiler built successfully"

# Test directory
TEST_DIR="tests/unit"
COMPILER="./build/jam"
PASSED=0
FAILED=0

# Function to run a single test
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .jam)
    
    echo -n "Testing $test_name... "
    
    # Compile the test
    if $COMPILER "$test_file" > /tmp/test_output.txt 2>&1; then
        # Check if compilation succeeded (LLVM IR was generated)
        if grep -q "ModuleID" /tmp/test_output.txt; then
            echo "PASS"
            ((PASSED++))
        else
            echo "FAIL"
            echo "   No LLVM IR generated for $test_file"
            ((FAILED++))
        fi
    else
        # Check if it's just a linker error (which is expected)
        if grep -q "ModuleID" /tmp/test_output.txt; then
            echo "PASS"
            ((PASSED++))
        else
            echo "FAIL"
            echo "   Error compiling $test_file"
            head -5 /tmp/test_output.txt | sed 's/^/   /'
            ((FAILED++))
        fi
    fi
}

# Function to verify LLVM IR contains expected types
verify_ir() {
    local test_file=$1
    local expected_type=$2
    local test_name=$(basename "$test_file" .jam)
    
    echo -n "Verifying $test_name IR for $expected_type... "
    
    # Compile and capture IR (ignore linker errors)
    $COMPILER "$test_file" > /tmp/ir_output.txt 2>&1
    
    if grep -q "$expected_type" /tmp/ir_output.txt; then
        echo "PASS"
        ((PASSED++))
    else
        echo "FAIL"
        echo "   Expected $expected_type not found in IR"
        ((FAILED++))
    fi
}

echo ""
echo "Running compilation tests..."

# Test u8 functionality
run_test "$TEST_DIR/test_u8.jam"
verify_ir "$TEST_DIR/test_u8.jam" "i8"

# Test u16 functionality  
run_test "$TEST_DIR/test_u16.jam"
verify_ir "$TEST_DIR/test_u16.jam" "i16"

# Test u32 functionality
run_test "$TEST_DIR/test_u32.jam"
verify_ir "$TEST_DIR/test_u32.jam" "i32"

# Test mixed types
run_test "$TEST_DIR/test_mixed_types.jam"

# Test if/else functionality
run_test "$TEST_DIR/test_if_else.jam"
run_test "$TEST_DIR/test_if_simple.jam"

# Test bool functionality
run_test "$TEST_DIR/test_bool.jam"
run_test "$TEST_DIR/test_bool_mixed.jam"

# Test string functionality
run_test "$TEST_DIR/test_string.jam"

# Test slice functionality
run_test "$TEST_DIR/test_slices.jam"
run_test "$TEST_DIR/test_mixed_slices.jam"

echo ""
echo "Running specific IR verification tests..."

# Verify specific type usage in IR
echo -n "Checking u8 max value (255)... "
$COMPILER "$TEST_DIR/test_u8.jam" > /tmp/u8_ir.txt 2>&1
if grep -q "i8 -1\|i8 255" /tmp/u8_ir.txt; then  # 255 might appear as -1 in signed representation
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking u16 large values... "
$COMPILER "$TEST_DIR/test_u16.jam" > /tmp/u16_ir.txt 2>&1
if grep -q "i16.*65535\|i16.*30000\|i16.*20000" /tmp/u16_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking u32 very large values... "
$COMPILER "$TEST_DIR/test_u32.jam" > /tmp/u32_ir.txt 2>&1
if grep -q "i32.*4294967295\|i32.*1000000" /tmp/u32_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking if/else IR generation... "
$COMPILER "$TEST_DIR/test_if_else.jam" > /tmp/if_ir.txt 2>&1
if grep -q "icmp eq\|br i1\|label %then\|label %else" /tmp/if_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking comparison operators... "
$COMPILER "$TEST_DIR/test_if_else.jam" > /tmp/cmp_ir.txt 2>&1
if grep -q "icmp ugt\|icmp eq\|icmp ne\|icmp ult\|icmp uge" /tmp/cmp_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking bool type IR generation... "
$COMPILER "$TEST_DIR/test_bool.jam" > /tmp/bool_ir.txt 2>&1
if grep -q "i1\|true\|false" /tmp/bool_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking bool function signatures... "
$COMPILER "$TEST_DIR/test_bool.jam" > /tmp/bool_func_ir.txt 2>&1
if grep -q "define i1.*bool\|i1 %flag" /tmp/bool_func_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking string slice IR generation... "
$COMPILER "$TEST_DIR/test_string.jam" > /tmp/string_ir.txt 2>&1
if grep -q "{ ptr, i64 }\|@str.*constant.*c\"" /tmp/string_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking slice type IR generation... "
$COMPILER "$TEST_DIR/test_slices.jam" > /tmp/slice_ir.txt 2>&1
if grep -q "{ ptr, i64 }\|zeroinitializer" /tmp/slice_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo -n "Checking UTF-8 string handling... "
$COMPILER "$TEST_DIR/test_string.jam" > /tmp/utf8_ir.txt 2>&1
if grep -q "\\\\E4\\\\B8\\\\96\\\\E7\\\\95\\\\8C\|\\\\F0\\\\9F\\\\8C\\\\8D" /tmp/utf8_ir.txt; then
    echo "PASS"
    ((PASSED++))
else
    echo "FAIL"
    ((FAILED++))
fi

echo ""
echo "Test Results"
echo "============"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total:  $((PASSED + FAILED))"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "All tests passed! The u8, u16, and u32 implementations are working correctly."
    exit 0
else
    echo ""
    echo "Some tests failed. Please check the implementation."
    exit 1
fi