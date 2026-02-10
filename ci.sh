#!/bin/bash
set -e  # Exit immediately if any command fails (non-zero exit code)

# Parse arguments
PLATFORM=""
DEVICE_ID="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

OS=$(uname -s)
echo "Running tests on $OS..."

# Run pytest
if [[ -d "tests" && "$OS" == "Linux" && "$PLATFORM" != "a2a3sim" ]]; then
    echo "Running pytest tests..."
    pytest tests -v
fi

# Discover and run all examples (recursively search for valid example directories)
EXAMPLES_DIR="examples"
while IFS= read -r -d '' example_dir; do
    # Skip the scripts directory
    if [[ "$example_dir" == *"/scripts" ]]; then
        continue
    fi

    # Check if this is a valid example (has kernels/kernel_config.py and golden.py)
    kernel_config="${example_dir}/kernels/kernel_config.py"
    golden="${example_dir}/golden.py"

    if [[ -f "$kernel_config" && -f "$golden" ]]; then
        # Get relative path from examples directory for display
        example_name="${example_dir#$EXAMPLES_DIR/}"
        echo "========================================"
        echo "Running example: $example_name"
        echo "========================================"

        # If platform is specified, use it
        if [[ -n "$PLATFORM" ]]; then
            if [[ "$PLATFORM" == "a2a3" ]]; then
                python examples/scripts/run_example.py \
                    -k "${example_dir}/kernels" \
                    -g "$golden" \
                    -p "$PLATFORM" -d "$DEVICE_ID"
            else
                python examples/scripts/run_example.py \
                    -k "${example_dir}/kernels" \
                    -g "$golden" \
                    -p "$PLATFORM"
            fi
        # Otherwise, use OS-based defaults
        elif [ "$OS" = "Darwin" ]; then
            # Mac: only simulation
            python examples/scripts/run_example.py \
                -k "${example_dir}/kernels" \
                -g "$golden" \
                -p a2a3sim
        else
            # Linux: both hardware and simulation
            python examples/scripts/run_example.py \
                -k "${example_dir}/kernels" \
                -g "$golden" \
                -p a2a3 -d "$DEVICE_ID"
            python examples/scripts/run_example.py \
                -k "${example_dir}/kernels" \
                -g "$golden" \
                -p a2a3sim
        fi
    fi
done < <(find "$EXAMPLES_DIR" -mindepth 1 -type d -print0 | sort -z)

echo "All tests passed!"
