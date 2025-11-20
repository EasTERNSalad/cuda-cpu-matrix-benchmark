# Makefile
NVCC = nvcc
TARGET = matrix_benchmark
SRC = src/matrix_benchmark.cu
CFLAGS = -O3 -g -std=c++14
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
