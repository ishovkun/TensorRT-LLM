#!/usr/bin/env python3
"""
Test script for selective_state_update function.
Compares the original selective_state_update with selective_state_update2.
"""

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
    Test correctness by comparing selective_state_update with reference implementation.
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

    # Make copies for each function (since state is modified in-place)
    state_ref = inputs["ssm_state_cache"].clone()
    state_test = inputs["ssm_state_cache"].clone()

    # Run reference implementation
    print("\nRunning reference implementation (selective_state_update_ref)...")
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

    # Run optimized implementation
    print("\nRunning optimized implementation (selective_state_update)...")
    try:
        y_test = torch.ops.trtllm.selective_state_update(
            state_test, inputs["x_decode"], inputs["dt_hp"],
            inputs["A_full"], inputs["B_decode"], inputs["C_decode"], D=inputs["D_full"],
            z=None,
            dt_bias=inputs["dt_bias_hp"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx_decode"],
            pad_slot_id=-1,
        )
        print(f"Optimized output shape: {y_test.shape}")
        print(
            f"Optimized output stats: min={y_test.min():.6f}, max={y_test.max():.6f}, mean={y_test.mean():.6f}"
        )
    except Exception as e:
        print(f"Optimized implementation failed: {e}")
        return False

    # Compare outputs
    print("\nComparing outputs...")
    output_diff = torch.abs(y_ref - y_test)
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    # np.testing.assert_allclose(y_test.detach().cpu().numpy(), y_ref.detach().cpu().numpy(), atol=atol)
    # Check if outputs match within tolerance
    outputs_match = torch.allclose(y_ref, y_test, atol=atol, rtol=rtol)

    if outputs_match:
        print(f"✓ Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ Outputs do NOT match within tolerance (atol={atol}, rtol={rtol})")
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

        # print(f"  Max difference: {max_diff}")
        # print(f"  Relative error: {max_diff / (torch.abs(y_ref).max().item() + 1e-8):.6e}")
        # print("y_test")
        # print(y_test)
        # print("y_ref")
        # print(y_ref)

    return outputs_match


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
    kernel_name = kernel.__name__
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


def main():
    """Main test function."""
    print("Selective State Update Test Suite")
    print(f"{'=' * 80}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        return

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
    for batch_size in [1, 10, 100]:
        passed = test_correctness(batch_size, nheads, dim, dstate, ngroups, dtype=dtype)
        if not passed:
            print(f"✗ Test {batch_size} failed")
            exit(1)
        test_passed = test_passed and passed

    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    if test_passed:
        print("✓ All correctness tests PASSED")
    else:
        print("✗ Some correctness tests FAILED")
        return False

    # Run performance benchmark
    repeats = 100
    warmup = 10
    batch_size = 2

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print(f"Batch size: {batch_size}")
    print(f"Warmup iterations: {warmup}")
    print(f"Benchmark iterations: {repeats}")
    print("=" * 80)

    make my container do sleep inf;

    benchmark_performance(selective_state_update, batch_size, nheads, dim, ngroups, dstate, repeats, warmup)
    benchmark_performance(torch.ops.trtllm.selective_state_update, batch_size, nheads, dim, ngroups, dstate, repeats, warmup)

    print("=" * 80)

    return True


if __name__ == "__main__":
    main()
