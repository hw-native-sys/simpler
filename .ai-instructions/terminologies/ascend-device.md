# Terminology (Based on Ascend NPU Architecture)

## Project Name

- **simpler** = Simple/Simpler Runtime: The repository name for the PTO Runtime project

## Hardware Units

- **AIC** = **AICore-CUBE**: Matrix computation unit for tensor operations (matmul, convolution)
- **AIV** = **AICore-VECTOR**: Vector computation unit for element-wise operations (add, mul, activation)
- **AICPU**: Control processor for task scheduling and data movement (not a worker type - acts as scheduler)
