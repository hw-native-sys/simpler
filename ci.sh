#!/bin/bash
set -e  # Exit on error

OS=$(uname -s)
echo "Running tests on $OS..."

# Run pytest
if [ -d "tests" ]; then
    pytest tests -v
fi

# Run examples based on OS
if [ "$OS" = "Darwin" ]; then
    # Mac: only run simulation
    echo "Mac detected, running simulation only..."
    python examples/scripts/run_example.py \
        -k examples/host_build_graph_example/kernels \
        -g examples/host_build_graph_example/golden.py \
        -p a2a3sim
else
    # Linux: run all platforms
    echo "Linux detected, running all platforms..."
    python examples/scripts/run_example.py \
        -k examples/host_build_graph_example/kernels \
        -g examples/host_build_graph_example/golden.py \
        -p a2a3
    python examples/scripts/run_example.py \
        -k examples/host_build_graph_example/kernels \
        -g examples/host_build_graph_example/golden.py \
        -p a2a3sim
fi

echo "All tests passed!"
