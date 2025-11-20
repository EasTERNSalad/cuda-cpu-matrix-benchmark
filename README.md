# CUDA CPU Matrix Multiplication Benchmark

## Summary
This repository benchmarks CPU vs GPU matrix multiplication for square matrices (256, 512, 1024). It measures CPU time using `clock()` and GPU kernel time using `cudaEventRecord()`.

## Requirements
- NVIDIA GPU + CUDA toolkit (tested with CUDA >= 11)
- nvcc compiler
- Python 3 with matplotlib

## Build
```bash
make
