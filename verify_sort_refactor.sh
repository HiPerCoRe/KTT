#!/bin/bash
# Script to verify that the refactored example behaves identically to the original
# by comparing the kernel configurations searched during tuning
#
# Usage: ./verify_sort_refactor.sh <ExampleName>
# Example: ./verify_sort_refactor.sh Sort
#          ./verify_sort_refactor.sh Transpose

set -e

# Get example name from argument
EXAMPLE_NAME="${1}"
if [ -z "$EXAMPLE_NAME" ]; then
    echo "Error: Example name required"
    echo "Usage: $0 <ExampleName>"
    echo "Example: $0 Sort"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/Build"
BIN_DIR="$BUILD_DIR/x86_64_Release"
EXAMPLE_DIR="$SCRIPT_DIR/Examples/$EXAMPLE_NAME"
EXAMPLE_CPP="$EXAMPLE_DIR/${EXAMPLE_NAME}.cpp"

if [ ! -f "$EXAMPLE_CPP" ]; then
    echo "Error: Example file not found: $EXAMPLE_CPP"
    exit 1
fi

echo "=== $EXAMPLE_NAME Refactor Verification Script ==="
echo ""

# Step 1: Build the original version (from master branch)
echo "Step 1: Building original $EXAMPLE_NAME example (from master branch)..."
cd "$SCRIPT_DIR"

# Checkout master to get original version
git checkout master -- "$EXAMPLE_CPP"

# Build with premake
echo "Generating build files with premake..."
premake5 gmake

echo "Building ${EXAMPLE_NAME}OpenCl..."
cd "$BUILD_DIR"
make > /dev/null 2>&1

# Run original version
echo "Running original $EXAMPLE_NAME example..."
cd "$BIN_DIR"
# ./Build/x86_64_Release/${EXAMPLE_NAME}OpenCl 0 0 "$EXAMPLE_DIR/${EXAMPLE_NAME}.cl" "$EXAMPLE_DIR/${EXAMPLE_NAME}Reference.cl" || true
./${EXAMPLE_NAME}OpenCl || true

# Save original Output.json configuration section
OUTPUT_JSON="$SCRIPT_DIR/Output.json"
NAME_OUTPUT_JSON="$SCRIPT_DIR/${EXAMPLE_NAME}Output.json"
cp "$OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Original_Output.json" 2>/dev/null || {
    cp "$NAME_OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Original_Output.json" 2>/dev/null || {
        echo "Error: Original $EXAMPLE_NAME did not produce Output.json"
        exit 1
    }
}

# Step 2: Restore refactored version and rebuild
echo ""
echo "Step 2: Building refactored $EXAMPLE_NAME example..."

# Restore the refactored version
git checkout examples-refactor -- "$EXAMPLE_CPP"

echo "Rebuilding ${EXAMPLE_NAME}OpenCl..."
cd "$BUILD_DIR"
make > /dev/null 2>&1

# Run refactored version
echo "Running refactored $EXAMPLE_NAME example..."
cd "$BIN_DIR"
./${EXAMPLE_NAME}OpenCl || true

# Save refactored Output.json configuration section
cp "$OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json" 2>/dev/null || {
    cp "$NAME_OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json" 2>/dev/null || {
        echo "Error: Refactored $EXAMPLE_NAME did not produce Output.json"
        exit 1
    }
}

cd "$SCRIPT_DIR"

# Step 3: Compare configurations
echo ""
echo "Step 3: Comparing kernel configurations..."

ORIGINAL_OUTPUT="$SCRIPT_DIR/${EXAMPLE_NAME}Original_Output.json"
REFACTORED_OUTPUT="$SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json"

if [ -f "$ORIGINAL_OUTPUT" ] && [ -f "$REFACTORED_OUTPUT" ]; then
    # Extract configurations from JSON using grep/sed (no jq dependency)
    echo "Comparing JSON outputs..."

    # Extract Configuration arrays from both files
    python3 compare_sort_configs.py "$ORIGINAL_OUTPUT" "$REFACTORED_OUTPUT"
else
    echo "JSON not found"
    exit 1
fi

rm $SCRIPT_DIR/${EXAMPLE_NAME}Original_Output.json
rm $SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json
rm $SCRIPT_DIR/${EXAMPLE_NAME}Output.json
