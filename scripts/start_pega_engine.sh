#!/bin/bash
# Start PegaEngine Server
#
# This script starts the PegaEngine server as an independent process.
# The server will handle KV cache operations for vLLM via CUDA IPC.
#
# Usage:
#   ./scripts/start_pega_engine.sh [OPTIONS]
#
# Options:
#   --device <n>   CUDA device index (default: 0)
#   --socket <path> ZMQ socket path (default: ipc:///tmp/pega_engine.sock)
#   --daemon       Run as background daemon
#

set -e

# Default values
DEVICE=0
SOCKET="ipc:///tmp/pega_engine.sock"
DAEMON=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --socket)
            SOCKET="$2"
            shift 2
            ;;
        --daemon)
            DAEMON=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --device <n>      CUDA device index (default: 0)"
            echo "  --socket <path>   ZMQ socket path (default: ipc:///tmp/pega_engine.sock)"
            echo "  --daemon          Run as background daemon"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Run engine server
echo "Starting PegaEngine Server..."
echo "  Device: $DEVICE"
echo "  Socket: $SOCKET"
echo "  Daemon: $DAEMON"
echo ""

if [ $DAEMON -eq 1 ]; then
    # Run in background
    nohup python -m pegaflow.engine_server --device "$DEVICE" --socket "$SOCKET" \
        > /tmp/pega_engine.log 2>&1 &
    PID=$!
    echo "PegaEngine server started in background (PID: $PID)"
    echo "Log file: /tmp/pega_engine.log"
    echo "To stop: kill $PID"
    echo $PID > /tmp/pega_engine.pid
else
    # Run in foreground
    python -m pegaflow.engine_server --device "$DEVICE" --socket "$SOCKET"
fi
