# Terminology (Based on Ascend NPU Architecture)

## Hardware Units

- **AIC** = **AICore-CUBE**: Matrix computation unit for tensor operations (matmul, convolution)
- **AIV** = **AICore-VECTOR**: Vector computation unit for element-wise operations (add, mul, activation)
- **AICPU**: Control processor for task scheduling and data movement (not a worker type - acts as scheduler)
