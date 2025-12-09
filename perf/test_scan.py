#!/usr/bin/env python3
"""
Test script for selective_state_update function.
Compares the original selective_state_update with selective_state_update2.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# Import the functions to test
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update


def create_test_inputs(
    batch_size, nheads, dim, dstate, ngroups, dtype=torch.float16, device="cuda"
):
    """
    Create test inputs based on the statistics from mydebug_selective_scan.txt

    Shape statistics show:
    - ssm_state_cache: (384, 64, 64, 128) - constant
    - x_decode: (1-2, 64, 64) - batch varies
    - dt_hp: (1-2, 64, 64) - batch varies
    - A_full: (64, 64, 128) - constant
    - B_decode: (1-2, 8, 128) - batch varies
    - C_decode: (1-2, 8, 128) - batch varies
    - D_full: (64, 64) - constant
    - dt_bias_hp: (64, 64) - constant
    - slot_idx_decode: (1-2,) - batch varies
    """
    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # if we use the cache, then the state indices are taken from a specific slot
    # so the state in the kernel will have batch as the first dimension, but it will
    # only come from a particular slot; the full tensor first dim is larger
    ssm_state_cache_size = max( 384, batch_size )
    # ssm_state_cache_size = batch_size

    ssm_state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=dtype, device=device
    )

    # Decode inputs - batch_size varies between 1-2 in the logs
    x_decode = torch.randn(batch_size, nheads, dim, dtype=dtype, device=device)
    dt = torch.randn(batch_size, nheads, dim, dtype=dtype, device=device)

    # A matrix - (nheads, dim, dstate)
    A_full = -torch.rand(nheads, dim, dstate, dtype=dtype, device=device)

    # B and C - (batch_size, ngroups, dstate)
    B_decode = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)
    C_decode = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)

    # D - (nheads, dim)
    D_full = torch.randn(nheads, dim, dtype=dtype, device=device)

    # dt_bias - (nheads, dim)
    dt_bias_hp = torch.randn(nheads, dim, dtype=dtype, device=device)

    # Slot indices for state batching - (batch_size,)
    # slot_idx_decode = torch.randint(0, batch_size, (batch_size,), dtype=torch.int32, device=device)
    slot_idx_decode = torch.randperm(ssm_state_cache_size, dtype=torch.int32, device=device)[:batch_size]

    return {
        "ssm_state_cache": ssm_state_cache,
        "x_decode": x_decode,
        "dt_hp": dt,
        "A_full": A_full,
        "B_decode": B_decode,
        "C_decode": C_decode,
        "D_full": D_full,
        "dt_bias_hp": dt_bias_hp,
        "slot_idx_decode": slot_idx_decode,
    }


def test_correctness(
    batch_size, nheads, dim, dstate, ngroups, dtype=torch.float16, atol=1e-3, rtol=1e-2
):
    """
    Test correctness by computing reference once and comparing multiple implementations against it.
    """
    print(f"\n{'=' * 80}")
    print(f"Testing correctness with batch_size={batch_size}, dtype={dtype}")
    print(f"{'=' * 80}")

    device = "cuda"

    inputs = create_test_inputs(
        batch_size=batch_size,
        nheads=nheads,
        dim=dim,
        dstate=dstate,
        ngroups=ngroups,
        dtype=dtype,
        device=device,
    )

    # Define test kernels to compare against reference
    kernels = [
        torch.ops.trtllm.selective_state_update,
        torch.ops.trtllm.selective_state_update_opt,
        torch.ops.trtllm.selective_state_update_simple,
        torch.ops.trtllm.selective_state_update_simple3,
        torch.ops.trtllm.selective_state_update_producer_consumer,
    ]

    # Run reference implementation once
    print("\nComputing reference output (selective_state_update)...")
    state_ref = inputs["ssm_state_cache"].clone()
    try:
        y_ref = selective_state_update(
            state_ref,
            inputs["x_decode"],
            inputs["dt_hp"],
            inputs["A_full"],
            inputs["B_decode"],
            inputs["C_decode"],
            D=inputs["D_full"],
            z=None,
            dt_bias=inputs["dt_bias_hp"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx_decode"],
            pad_slot_id=-1,
        )
        print(f"Reference output shape: {y_ref.shape}")
        print(
            f"Reference output stats: min={y_ref.min():.6f}, max={y_ref.max():.6f}, mean={y_ref.mean():.6f}"
        )
    except Exception as e:
        print(f"Reference implementation failed: {e}")
        return False

    # Test each kernel implementation against the reference
    all_passed = True
    for kernel_func in kernels:
        # kernel_name = kernel_func.__name__
        kernel_name = f"{kernel_func.__module__}.{kernel_func.__name__}"
        print(f"\n{'-' * 80}")
        print(f"Testing kernel: {kernel_name}")
        print(f"{'-' * 80}")

        # Create fresh state copy for this kernel
        state_test = inputs["ssm_state_cache"].clone()

        try:
            y_test = kernel_func(
                state_test,
                inputs["x_decode"],
                inputs["dt_hp"],
                inputs["A_full"],
                inputs["B_decode"],
                inputs["C_decode"],
                D=inputs["D_full"],
                z=None,
                dt_bias=inputs["dt_bias_hp"],
                dt_softplus=True,
                state_batch_indices=inputs["slot_idx_decode"],
                pad_slot_id=-1,
            )
            print(f"Output shape: {y_test.shape}")
            print(
                f"Output stats: min={y_test.min():.6f}, max={y_test.max():.6f}, mean={y_test.mean():.6f}"
            )
        except Exception as e:
            print(f"✗ Kernel failed: {e}")
            all_passed = False
            continue

        # Compare outputs
        print("\nComparing with reference...")
        output_diff = torch.abs(y_ref - y_test)
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()

        print(f"Max absolute difference: {max_diff:.6e}")
        print(f"Mean absolute difference: {mean_diff:.6e}")

        # Check if outputs match within tolerance
        outputs_match = torch.allclose(y_ref, y_test, atol=atol, rtol=rtol)

        if outputs_match:
            print(f"✓ Outputs match within tolerance (atol={atol}, rtol={rtol})")
        else:
            print(f"✗ Outputs do NOT match within tolerance (atol={atol}, rtol={rtol})")
            all_passed = False

            # Detailed comparison using numpy testing
            y_ref_np = y_ref.detach().cpu().numpy()
            y_test_np = y_test.detach().cpu().numpy()

            print("\nDetailed mismatch analysis:")
            mismatch_mask = ~np.isclose(y_ref_np, y_test_np, atol=atol, rtol=rtol)
            num_mismatches = np.sum(mismatch_mask)
            total_elements = y_ref_np.size

            print(
                f"Number of mismatched elements: {num_mismatches} / {total_elements} ({100 * num_mismatches / total_elements:.2f}%)"
            )

            if num_mismatches > 0:
                mismatch_indices = np.argwhere(mismatch_mask)
                print(f"First few mismatch locations (up to 10):")
                for i, idx in enumerate(mismatch_indices[:10]):
                    idx_tuple = tuple(idx)
                    ref_val = y_ref_np[idx_tuple]
                    test_val = y_test_np[idx_tuple]
                    diff = abs(ref_val - test_val)
                    rel_diff = diff / (abs(ref_val) + 1e-8)
                    print(
                        f"  Index {idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, diff={diff:.6e}, rel_diff={rel_diff:.6e}"
                    )

    return all_passed


def benchmark_performance(
    kernel,
    batch_size,
    nheads,
    dim,
    ngroups=8,
    dstate=128,
    num_iterations=100,
    warmup=10,
):

    device = "cuda"
    dtype = torch.float16

    inputs = create_test_inputs(
        batch_size=batch_size,
        nheads=nheads,
        dim=dim,
        dstate=dstate,
        ngroups=ngroups,
        dtype=dtype,
        device=device,
    )

    # Benchmark implementation
    # kernel_name = kernel.__name__
    kernel_name = f"{kernel.__module__}.{kernel.__name__}"
    print(f"\nBenchmarking {kernel_name}...")

    # Warmup phase
    state = inputs["ssm_state_cache"].clone()
    for i in range(warmup):
        y = kernel(
            state,
            inputs["x_decode"],
            inputs["dt_hp"],
            inputs["A_full"],
            inputs["B_decode"],
            inputs["C_decode"],
            D=inputs["D_full"],
            z=None,
            dt_bias=inputs["dt_bias_hp"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx_decode"],
            pad_slot_id=-1,
        )

    # Benchmark phase
    torch.cuda.synchronize()
    state = inputs["ssm_state_cache"].clone()
    start = time.perf_counter()

    for i in range(num_iterations):
        y = kernel(
            state,
            inputs["x_decode"],
            inputs["dt_hp"],
            inputs["A_full"],
            inputs["B_decode"],
            inputs["C_decode"],
            D=inputs["D_full"],
            z=None,
            dt_bias=inputs["dt_bias_hp"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx_decode"],
            pad_slot_id=-1,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = (end - start) * 1000  # Convert to ms
    mean_time = total_time / num_iterations

    # print(f"  Total time: {total_time:.3f} ms")
    print(f"  Mean time per iteration: {mean_time:.3f} ms")

    return mean_time


def main(batch_size, repeats, warmup, skip_test):
    """Main test function."""

    print("Selective State Update Test Suite")
    print(f"{'=' * 80}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        return {}

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Test correctness with different batch sizes
    print("\n" + "=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)

    nheads = 64
    dim = 64
    dstate = 128
    ngroups = 8

    # nheads = 1
    # dim = 32
    # dstate = 128
    # ngroups = 1

    dtype = torch.float16

    test_passed = True
    if not skip_test:
        for test_batch_size in [1, 16, 32, 64, 256, 512]:
            passed = test_correctness(test_batch_size, nheads, dim, dstate, ngroups, dtype=dtype)
            if not passed:
                print(f"✗ Test {batch_size} failed")
                exit(1)
            test_passed = test_passed and passed

        # Final summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        if test_passed:
            print("✓ All correctness tests PASSED")
        else:
            print("✗ Some correctness tests FAILED")
            return {}
    else:
        print("\n" + "=" * 80)
        print("SKIPPING CORRECTNESS TESTS")
        print("=" * 80)

    # Run performance benchmark
    # batch_size = 2
    # batch_size = 10

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print(f"Batch size: {batch_size}")
    print(f"Warmup iterations: {warmup}")
    print(f"Benchmark iterations: {repeats}")
    print("=" * 80)

    results = {}

    kernels = [
        selective_state_update,
        # torch.ops.trtllm.selective_state_update,
        torch.ops.trtllm.selective_state_update_opt,
        torch.ops.trtllm.selective_state_update_simple,
        torch.ops.trtllm.selective_state_update_simple3,
    ]

    for kernel in kernels:
        kernel_name = f"{kernel.__module__}.{kernel.__name__}"
        avg_time = benchmark_performance(kernel, batch_size, nheads, dim, ngroups, dstate, repeats, warmup)
        results[kernel_name] = avg_time

    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective State Update Test Suite")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=100, help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size during benchmark")
    parser.add_argument("--skip-test", action="store_true", default=False, help="Skip correctness tests (default: False)")
    args = parser.parse_args()

    results = main(args.batch_size, args.repeat, args.warmup, args.skip_test)

    if results:
        # print("\nBenchmark Results:")
        # for kernel_name, avg_time in results.items():
        #     print(f"  {kernel_name}: {avg_time:.3f} ms")
        exit(0)
    else:
        exit(1)
