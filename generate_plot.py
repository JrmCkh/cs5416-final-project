#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python generate_plot.py <mem_profile_file> <output_png>")
    sys.exit(1)

MEM_PROFILE = sys.argv[1]
MEM_PLOT = sys.argv[2]

times = []
memory = []

# Load memory profile data
with open(MEM_PROFILE) as f:
    first_ts = None
    for line in f:
        if line.startswith("MEM"):
            _, mem, ts = line.strip().split()
            mem = float(mem)
            ts = float(ts)
            if first_ts is None:
                first_ts = ts
            times.append(ts - first_ts)
            memory.append(mem)

# Plot memory over time
plt.plot(times, memory, label="Memory Usage (MB)")

# Annotate peak memory
peak_idx = memory.index(max(memory))
plt.scatter([times[peak_idx]], [memory[peak_idx]], color='red', label="Peak Memory")
plt.text(times[peak_idx], memory[peak_idx], f'{memory[peak_idx]:.1f} MB', color='red')

plt.xlabel("Time (s)")
plt.ylabel("Memory (MB)")
plt.title(f"Memory Usage Over Time")
plt.legend()
plt.grid(True)
plt.savefig(MEM_PLOT)
print(f"Memory plot saved to {MEM_PLOT}")
