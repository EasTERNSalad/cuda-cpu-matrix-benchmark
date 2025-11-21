import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
naive = pd.read_csv("results_gpu_naive.csv")
tiled = pd.read_csv("results_gpu_tiled.csv")

plt.figure(figsize=(10, 6))

# Plot GPU naive
plt.plot(naive["Size"], naive["GPU_s"], marker='o', label="GPU Naive")

# Plot GPU tiled
plt.plot(tiled["Size"], tiled["GPU_s"], marker='o', label="GPU Tiled")

plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Execution Time (seconds)")
plt.title("GPU Performance: Naive vs Tiled Kernel")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("plots/gpu_performance.png")

print("Saved plot: plots/gpu_performance.png")
