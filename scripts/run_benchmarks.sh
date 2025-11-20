#!/bin/bash
set -e
OUT=results.csv
echo "Size,CPU_s,GPU_s,Speedup" > $OUT
BIN=./matrix_benchmark

SIZES=(256 512 1024)
for N in "${SIZES[@]}"; do
  echo "Running N=$N"
  # run both (cpu + naive gpu)
  ./matrix_benchmark $N both | tee tmp.out
  # extract CSV line
  CSVLINE=$(grep '^CSV' tmp.out | sed 's/^CSV,//')
  echo $CSVLINE >> $OUT
done

rm -f tmp.out
echo "Results saved to $OUT"
