#!/usr/bin/env python3
"""
Script to collect benchmark results from test_scan.py across different batch sizes.
Runs the main method with different batch sizes (powers of two from 1 to 256)
and collects the results into a single pandas dataframe.
"""

import pandas as pd
from test_scan import main

# Powers of two from 1 to 256
batch_sizes = [2**i for i in range(10)]  # 1, 2, 4, 8, 16, 32, 64, 128, 256

all_results = []

print("=" * 80)
print("COLLECTING BENCHMARK RESULTS")
print("=" * 80)
print(f"Batch sizes to test: {batch_sizes}")
print("=" * 80)

for batch_size in batch_sizes:
    print(f"\n{'=' * 80}")
    print(f"Running benchmark for batch_size={batch_size}")
    print(f"{'=' * 80}")

    # Run the main function from test_scan.py with fixed parameters
    results = main(batch_size=batch_size, repeats=200, warmup=10, skip_test=True)

    if not results:
        print(f"Warning: No results returned for batch_size={batch_size}")
        continue

    # Add batch size to each result and collect
    for kernel_name, avg_time in results.items():
        all_results.append({
            'batch_size': batch_size,
            'kernel': kernel_name,
            'avg_time_ms': avg_time
        })

# Create DataFrame
df = pd.DataFrame(all_results)

# Print results
print("\n" + "=" * 80)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 80)

if df.empty:
    print("No results collected!")
else:
    # Pivot to have kernels as columns and batch sizes as rows
    df_pivot = df.pivot(index='batch_size', columns='kernel', values='avg_time_ms')

    print("\nAverage execution time (ms) by batch size and kernel:")
    # print(df_pivot.to_string())
    print(df_pivot.to_csv())
