# plot_performance.py
import matplotlib.pyplot as plt
import csv

sizes = []
cpu_times = []
gpu_times = []
speedups = []

with open('results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sizes.append(int(row['Size']))
        cpu_times.append(float(row['CPU_s']) if row['CPU_s'] != 'NA' else None)
        gpu_times.append(float(row['GPU_s']))
        speedups.append(float(row['Speedup']) if row['Speedup'] != 'NA' else None)

plt.figure()
plt.plot(sizes, cpu_times, marker='o', label='CPU time (s)')
plt.plot(sizes, gpu_times, marker='o', label='GPU kernel time (s)')
plt.xlabel('Matrix size (N x N)')
plt.ylabel('Execution time (s)')
plt.title('CPU vs GPU Performance')
plt.xscale('log', base=2)
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.savefig('plots/performance.png', dpi=300)
print("Saved performance.png")
