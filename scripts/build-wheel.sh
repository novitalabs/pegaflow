#!/bin/bash
# Automated build script for pegaflow Python package with embedded binary
# Usage: ./scripts/build-wheel.sh [--release]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/python"

# Parse arguments
RELEASE_FLAG=""
PROFILE="debug"
if [[ "$1" == "--release" ]]; then
    RELEASE_FLAG="--release"
    PROFILE="release"
    shift
fi

echo "==> Building pegaflow-server-py binary ($PROFILE mode)..."
cd "$PROJECT_ROOT"
cargo build $RELEASE_FLAG -p pegaflow-py --bin pegaflow-server-py

echo "==> Copying binary to Python package..."
cp "$PROJECT_ROOT/target/$PROFILE/pegaflow-server-py" "$PYTHON_DIR/pegaflow/pegaflow-server-py"
chmod +x "$PYTHON_DIR/pegaflow/pegaflow-server-py"

echo "==> Building Python wheel with maturin..."
cd "$PYTHON_DIR"
maturin build $RELEASE_FLAG "$@"

echo ""
echo "==> Done! Wheel built at:"
ls -lh "$PROJECT_ROOT/target/wheels/"pegaflow-*.whl | tail -1
echo ""
echo "To install: pip install $(ls -t $PROJECT_ROOT/target/wheels/pegaflow-*.whl | head -1)"
