#!/bin/bash
# Script to verify that the refactored example behaves identically to the original
# by comparing the kernel configurations searched during tuning
# Uses ReferenceVersions folder instead of git checkout to avoid changing repo state
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
REF_VERSIONS_DIR="$SCRIPT_DIR/Examples/ReferenceVersions/$EXAMPLE_NAME"

echo "=== $EXAMPLE_NAME Refactor Verification Script ==="
echo ""

# Check that both regular and reference versions exist
if [ ! -d "$EXAMPLE_DIR" ]; then
    echo "Error: Example directory not found: $EXAMPLE_DIR"
    exit 1
fi

if [ ! -d "$REF_VERSIONS_DIR" ]; then
    echo "Error: Reference version directory not found: $REF_VERSIONS_DIR"
    echo "The ReferenceVersions folder must contain the original version of the example."
    exit 1
fi

# Step 1: Build with reference versions enabled
echo "Step 1: Generating build files with premake (reference-versions enabled)..."
cd "$SCRIPT_DIR"

# Clean previous build to ensure fresh generation
rm -rf "$BUILD_DIR"

premake5 gmake --reference-versions

echo "Step 2: Building ${EXAMPLE_NAME}OpenCl and ${EXAMPLE_NAME}ReferenceOpenCl..."
cd "$BUILD_DIR"
make > /dev/null 2>&1

# Step 3: Run reference version first
echo ""
echo "Step 3: Running reference $EXAMPLE_NAME example..."
cd "$BIN_DIR"
./${EXAMPLE_NAME}ReferenceOpenCl || true

# Save reference version Output.json
OUTPUT_JSON="$SCRIPT_DIR/Output.json"
NAME_OUTPUT_JSON="$SCRIPT_DIR/${EXAMPLE_NAME}Output.json"

if [ -f "$OUTPUT_JSON" ]; then
    cp "$OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Reference_Output.json"
elif [ -f "$NAME_OUTPUT_JSON" ]; then
    cp "$NAME_OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Reference_Output.json"
else
    echo "Error: Reference $EXAMPLE_NAME did not produce Output.json"
    exit 1
fi

# Step 4: Run refactored version
echo ""
echo "Step 4: Running refactored $EXAMPLE_NAME example..."
cd "$BIN_DIR"
./${EXAMPLE_NAME}OpenCl || true

# Save refactored version Output.json
if [ -f "$OUTPUT_JSON" ]; then
    cp "$OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json"
elif [ -f "$NAME_OUTPUT_JSON" ]; then
    cp "$NAME_OUTPUT_JSON" "$SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json"
else
    echo "Error: Refactored $EXAMPLE_NAME did not produce Output.json"
    exit 1
fi

cd "$SCRIPT_DIR"

# Step 5: Compare configurations
echo ""
echo "Step 5: Comparing kernel configurations..."

REFERENCE_OUTPUT="$SCRIPT_DIR/${EXAMPLE_NAME}Reference_Output.json"
REFACTORED_OUTPUT="$SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json"

if [ -f "$REFERENCE_OUTPUT" ] && [ -f "$REFACTORED_OUTPUT" ]; then
    echo "Comparing JSON outputs..."
    python3 compare_sort_configs.py "$REFERENCE_OUTPUT" "$REFACTORED_OUTPUT"
else
    echo "JSON not found"
    exit 1
fi

# Cleanup
echo ""
echo "Cleaning up..."
rm -f "$SCRIPT_DIR/${EXAMPLE_NAME}Reference_Output.json"
rm -f "$SCRIPT_DIR/${EXAMPLE_NAME}Refactored_Output.json"
rm -f "$SCRIPT_DIR/${EXAMPLE_NAME}Output.json"
rm -f "$SCRIPT_DIR/Output.json"

echo ""
echo "Verification complete!"
