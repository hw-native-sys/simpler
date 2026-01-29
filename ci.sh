#!/bin/bash

# run all tests
pytest tests
# run all examples
python examples/host_build_graph_example/main.py
python examples/host_build_graph_sim_example/main.py
