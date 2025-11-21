// matrix_benchmark.cu
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>         // <<< added for ofstream
#include <string>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                             \
    cudaError_t err = (call);                             \
    if (err != cudaSuccess) {                             \
        fprintf(stderr, "CUDA Error %s:%d: %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                               \
    }                                                     \
} while(0)

// Naive CPU matrix multiply (row-major)
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N*N; ++i) C[i] = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A[i*N + k];
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }
}

// CUDA naive kernel: one thread computes one C element
__global__ void matmul_kernel_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optional: tiled/shared memory kernel (faster on large matrices)
template <int TILE_SIZE>
__global__ void matmul_kernel_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float val = 0.0f;
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        int aRow = row;
        int aCol = m * TILE_SIZE + threadIdx.x;
        int bRow = m * TILE_SIZE + threadIdx.y;
        int bCol = col;

        sA[threadIdx.y][threadIdx.x] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();
        for (int e = 0; e < TILE_SIZE; ++e) {
            val += sA[threadIdx.y][e] * sB[e][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = val;
}

int main(int argc, char** argv) {
    // Usage: matrix_benchmark [size] [mode]
    // mode: cpu | gpu_naive | gpu_tiled
    int N = 256;
    std::string mode = "both";

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) mode = argv[2];

    printf("Matrix multiply N=%d mode=%s\n", N, mode.c_str());

    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA(N*N), hB(N*N), hC(N*N);

    // initialize with simple values
    for (int i = 0; i < N*N; ++i) {
        hA[i] = static_cast<float>((i % 100) + 1) * 0.001f; // small values
        hB[i] = static_cast<float>(((i+3) % 100) + 1) * 0.001f;
    }

    // CPU run
    double cpu_seconds = -1.0;
    if (mode == "cpu" || mode == "both") {
        // warm-up
        matmul_cpu(hA.data(), hB.data(), hC.data(), N);
        std::clock_t start = std::clock();
        matmul_cpu(hA.data(), hB.data(), hC.data(), N);
        std::clock_t end = std::clock();
        cpu_seconds = double(end - start) / CLOCKS_PER_SEC;
        printf("CPU time (clock()) = %.6f s\n", cpu_seconds);
    }

    // GPU run
    if (mode == "gpu_naive" || mode == "gpu_tiled" || mode == "both") {
        float *dA=nullptr, *dB=nullptr, *dC=nullptr;
        CHECK_CUDA(cudaMalloc(&dA, bytes));
        CHECK_CUDA(cudaMalloc(&dB, bytes));
        CHECK_CUDA(cudaMalloc(&dC, bytes));

        CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        // kernel config
        const int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        // Events for timing kernel only
        cudaEvent_t startEvent, stopEvent;
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));
        CHECK_CUDA(cudaEventRecord(startEvent, 0));

        if (mode == "gpu_naive" || (mode=="both" && argc<4)) {
            // naive kernel
            matmul_kernel_naive<<<grid, block>>>(dA, dB, dC, N);
            CHECK_CUDA(cudaGetLastError());
        } else if (mode == "gpu_tiled" || (mode=="both" && argc>=4 && std::string(argv[3])=="tiled")) {
            // choose tiled kernel with TILE=16 or 32 based on N
            matmul_kernel_tiled<16><<<grid, block>>>(dA, dB, dC, N);
            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaEventRecord(stopEvent, 0));
        CHECK_CUDA(cudaEventSynchronize(stopEvent));
        float gpu_milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_milliseconds, startEvent, stopEvent));

        // Copy back (not included in kernel timing)
        CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

        printf("GPU kernel time (cudaEvent) = %.3f ms (%.6f s)\n", gpu_milliseconds, gpu_milliseconds / 1000.0f);

        // Cleanup
        CHECK_CUDA(cudaEventDestroy(startEvent));
        CHECK_CUDA(cudaEventDestroy(stopEvent));
        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));

        // print CSV row: Size,CPU_time_s,GPU_time_s,Speedup
        double gpu_seconds = double(gpu_milliseconds) / 1000.0;
        double speedup = (cpu_seconds > 0.0) ? (cpu_seconds / gpu_seconds) : 0.0;
        if (cpu_seconds > 0.0) {
            printf("CSV,%d,%.6f,%.6f,%.3f\n", N, cpu_seconds, gpu_seconds, speedup);
        } else {
            printf("CSV,%d,NA,%.6f,NA\n", N, gpu_seconds);
        }

        // Append to results.csv (create header if not present)
        bool need_header = false;
        {
            std::ifstream infile("results.csv");
            if (!infile.good()) need_header = true;
        }

        std::ofstream outfile;
        outfile.open("results.csv", std::ios::app);

        if (!outfile) {
            std::cerr << "Error: Could not open results.csv for writing!\n";
        } else {
            if (need_header) {
                outfile << "Size,CPU_s,GPU_s,Speedup\n";
            }

            if (cpu_seconds > 0.0) {
                outfile << N << ","
                        << cpu_seconds << ","
                        << gpu_seconds << ","
                        << speedup << "\n";
            } else {
                outfile << N << ",NA," << gpu_seconds << ",NA\n";
            }
        }

        outfile.close();
    } // end GPU block

    return 0;
}
