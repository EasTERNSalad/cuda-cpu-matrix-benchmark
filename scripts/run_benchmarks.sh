#!/bin/bash

# Output CSV files
NAIVE_CSV="results_gpu_naive.csv"
TILED_CSV="results_gpu_tiled.csv"

# Matrix sizes to test
sizes=(128 256 512 1024 1536 2048 4096)

# Write headers
echo "Size,GPU_s" > $NAIVE_CSV
echo "Size,GPU_s" > $TILED_CSV

echo "Running GPU NAIVE benchmarks..."
for n in "${sizes[@]}"; do
    echo "N=$n (gpu_naive)"
    out=$(./matrix_benchmark $n gpu_naive | grep "CSV")
    # CSV,N,NA,cpu,gpu
    gpu_time=$(echo $out | cut -d',' -f4)   # GPU seconds is 4th field
    echo "$n,$gpu_time" >> $NAIVE_CSV
done

echo "Running GPU TILED benchmarks..."
for n in "${sizes[@]}"; do
    echo "N=$n (gpu_tiled)"
    out=$(./matrix_benchmark $n gpu_tiled | grep "CSV")
    gpu_time=$(echo $out | cut -d',' -f4)
    echo "$n,$gpu_time" >> $TILED_CSV
done

echo "Done!"
echo "Saved:"
echo " - $NAIVE_CSV"
echo " - $TILED_CSV"
