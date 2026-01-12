#!/usr/bin/env python3
"""
Test script for selective_state_update function.
Compares the original selective_state_update with selective_state_update2.
"""

import argparse
import pickle
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
    # ssm_state_cache_size = max(384, int(2*batch_size))
    ssm_state_cache_size = max(384, batch_size)
    # ssm_state_cache_size = batch_size

    ssm_state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=dtype, device=device
    )

    # Decode inputs - batch_size varies between 1-2 in the logs
    x_decode = torch.randn(batch_size, nheads, dim, dtype=dtype, device=device)

    # dt: shape (batch, nheads, dim) with strides (nheads, 1, 0) - one value per head
    dt_base = torch.randn(batch_size, nheads, dtype=dtype, device=device)
    dt = dt_base.as_strided((batch_size, nheads, dim), (nheads, 1, 0))

    # A matrix - (nheads, dim, dstate) with strides (1, 0, 0) - one value per head
    A_base = -torch.rand(nheads, dtype=torch.float32, device=device)
    A_full = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    # B and C - (batch_size, ngroups, dstate)
    B_decode = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)
    C_decode = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)

    # D - (nheads, dim) with strides (1, 0) - one value per head
    D_full = torch.randn(nheads, dtype=dtype, device=device).as_strided((nheads, dim), (1, 0))

    # dt_bias - (nheads, dim) with strides (1, 0) - one value per head
    dt_bias_base = torch.randn(nheads, dtype=dtype, device=device)
    dt_bias_hp = dt_bias_base.as_strided((nheads, dim), (1, 0))

    # Slot indices for state batching - (batch_size,)
    slot_idx_decode = torch.randperm(ssm_state_cache_size, dtype=torch.int32, device=device)[
        :batch_size
    ]

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


def test_correctness(inputs, atol=1e-3, rtol=1e-2):
    """
    Test correctness by computing reference once and comparing multiple implementations against it.
    """

    # Define test kernels to compare against reference
    kernels = [
        torch.ops.trtllm.selective_state_update_simple,
        torch.ops.trtllm.selective_state_update_producer_consumer,
        torch.ops.trtllm.selective_state_update_producer_consumer_writeback,
        torch.ops.trtllm.selective_state_update_producer_consumer_horizontal,
        torch.ops.trtllm.selective_state_update_producer_consumer_horizontal_warps,
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
        print(f"Reference output shape: {y_ref.shape} dtype = {y_ref.dtype}")
        print(
            f"Reference output stats: min={y_ref.min():.6f}, max={y_ref.max():.6f}, mean={y_ref.mean():.6f}"
        )
    except Exception as e:
        print(f"Reference implementation failed: {e}")
        return False

    # Test each kernel implementation against the reference
    all_passed = True
    for kernel_func in kernels:
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
            print(f"Output shape: {y_test.shape} dtype = {y_test.dtype}")
            print(
                f"Output stats: min={y_test.min():.6f}, max={y_test.max():.6f}, mean={y_test.mean():.6f}"
            )
        except Exception as e:
            print(f"✗ Kernel failed: {e}")
            all_passed = False
            continue

        # Compare outputs
        print("\nComparing outputs with reference...")
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
            y_ref_np = y_ref.detach().cpu().float().numpy()
            y_test_np = y_test.detach().cpu().float().numpy()
            print(f"dtypes: ref {y_ref_np.dtype}, test {y_test_np.dtype}")

            print("\nDetailed mismatch analysis:")
            mismatch_mask = ~np.isclose(y_ref_np, y_test_np, atol=atol, rtol=rtol)
            num_mismatches = np.sum(mismatch_mask)
            total_elements = y_ref_np.size

            print(
                f"Number of mismatched output (y) elements: {num_mismatches} / {total_elements} ({100 * num_mismatches / total_elements:.2f}%)"
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

        # Compare states (updated in-place)
        print("\nComparing states with reference...")

        # Extract the relevant state slices using slot indices
        state_ref_batch = state_ref[inputs["slot_idx_decode"]]
        state_test_batch = state_test[inputs["slot_idx_decode"]]

        state_diff = torch.abs(state_ref_batch - state_test_batch)
        max_state_diff = state_diff.max().item()
        mean_state_diff = state_diff.mean().item()

        print(f"Max absolute state difference: {max_state_diff:.6e}")
        print(f"Mean absolute state difference: {mean_state_diff:.6e}")

        # Check if states match within tolerance
        states_match = torch.allclose(state_ref_batch, state_test_batch, atol=atol, rtol=rtol)

        if states_match:
            print(f"✓ States match within tolerance (atol={atol}, rtol={rtol})")
        else:
            print(f"✗ States do NOT match within tolerance (atol={atol}, rtol={rtol})")
            all_passed = False

            # Detailed comparison using numpy testing
            state_ref_np = state_ref_batch.detach().cpu().float().numpy()
            state_test_np = state_test_batch.detach().cpu().float().numpy()

            print("\nDetailed state mismatch analysis:")
            state_mismatch_mask = ~np.isclose(state_ref_np, state_test_np, atol=atol, rtol=rtol)
            num_state_mismatches = np.sum(state_mismatch_mask)
            total_state_elements = state_ref_np.size

            print(
                f"Number of mismatched state elements: {num_state_mismatches} / {total_state_elements} ({100 * num_state_mismatches / total_state_elements:.2f}%)"
            )

            if num_state_mismatches > 0:
                state_mismatch_indices = np.argwhere(state_mismatch_mask)
                print(f"First few state mismatch locations (up to 10):")
                for i, idx in enumerate(state_mismatch_indices[:10]):
                    idx_tuple = tuple(idx)
                    ref_val = state_ref_np[idx_tuple]
                    test_val = state_test_np[idx_tuple]
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
    # dtype = torch.float16
    dtype = torch.bfloat16

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


def load_custom_inputs_from_debug_file(filename, device='cuda'):
    """Load custom inputs from debug file and reconstruct tensors with exact strides.

    Args:
        filename: Path to the debug file (e.g., 'debug.pt')
        device: Device to move tensors to ('cuda' or 'cpu')

    Returns:
        Dictionary with inputs formatted for test_correctness
    """
    def reconstruct_tensor(metadata, device='cuda'):
        """Reconstruct a tensor from saved metadata with exact strides."""
        if not isinstance(metadata, dict) or 'data' not in metadata:
            return metadata

        # Extract metadata
        storage_bytes = metadata['data']
        dtype_raw = metadata['dtype']

        # Convert dtype to torch.dtype if it's a string
        if isinstance(dtype_raw, str):
            # Handle strings like "torch.float32" or "float32"
            dtype_str = dtype_raw.replace('torch.', '')
            dtype = getattr(torch, dtype_str)
        else:
            dtype = dtype_raw

        shape = tuple(metadata['shape'])
        stride = tuple(metadata['stride'])
        storage_offset = metadata['storage_offset']

        # Create UntypedStorage from raw bytes and move to target device
        untyped_storage = torch.UntypedStorage.from_buffer(storage_bytes, dtype=torch.uint8)
        if device != 'cpu':
            # Extract device index if device is a string like 'cuda' or 'cuda:0'
            if isinstance(device, str):
                if ':' in device:
                    device_idx = int(device.split(':')[1])
                else:
                    device_idx = 0
            else:
                device_idx = device
            untyped_storage = untyped_storage.cuda(device_idx)

        # Create tensor directly on target device with exact layout
        tensor = torch.tensor([], dtype=dtype, device=device).set_(
            untyped_storage,
            storage_offset,
            shape,
            stride
        )

        return tensor

    # Load debug data
    debug_data = torch.load(filename)
    print(f"Loaded {filename} successfully")

    # Reconstruct tensors from saved metadata to preserve exact strides
    custom_inputs = {}
    for key, value in debug_data.items():
        custom_inputs[key] = reconstruct_tensor(value, device)

    # Map keys to match test_correctness expected format
    inputs_for_test = {
        'ssm_state_cache': custom_inputs['ssm_state_cache'],
        'x_decode': custom_inputs['x_decode'],
        'dt_hp': custom_inputs['dt_hp'],
        'A_full': custom_inputs['A_full'],
        'B_decode': custom_inputs['B_decode'],
        'C_decode': custom_inputs['C_decode'],
        'D_full': custom_inputs['D_full'],
        'dt_bias_hp': custom_inputs['dt_bias_hp'],
        'slot_idx_decode': custom_inputs['state_batch_indices'],
    }

    # Check for NaN values in tensors
    for key, value in inputs_for_test.items():
        if isinstance(value, torch.Tensor):
            if key != 'ssm_state_cache':
                if torch.isnan(value).any():
                    print(f"Error: Tensor '{key}' contains NaN values")
                    exit(1)
            else:
                if torch.isnan(value[inputs_for_test['slot_idx_decode']]).any():
                    print(f"Error: Tensor '{key}' contains NaN values in selected indices")
                    exit(1)

    # Print reconstructed tensor information
    print(f"\nCustom input shapes and strides after reconstruction:")
    for key, value in inputs_for_test.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, stride={value.stride()}, dtype={value.dtype}, device={value.device}")

    return inputs_for_test


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

    # dtype = torch.float16
    dtype = torch.bfloat16

    test_passed = True
    if not skip_test:
        for test_batch_size in [1, 16, 32, 64, 256, 512]:
            inputs = create_test_inputs(
                batch_size=batch_size,
                nheads=nheads,
                dim=dim,
                dstate=dstate,
                ngroups=ngroups,
                dtype=dtype,
                device="cuda",
            )

            print(f"\n{'=' * 80}")
            input_dtype = inputs["x_decode"].dtype
            print(f"Testing correctness with batch_size={batch_size}, input_dtype={input_dtype}")
            print(f"{'=' * 80}")
            passed = test_correctness(inputs)
            if not passed:
                print(f"✗ Test {batch_size} failed")
                exit(1)
            test_passed = test_passed and passed

        # print("testing with custom input")
        # # Load debug.pt and test with custom inputs
        # custom_test_file = 'debug.pt'
        # try:
        #     inputs_for_test = load_custom_inputs_from_debug_file(custom_test_file, device='cuda')
        #     print("inputs loaded from debug.pt successfully")

        #     # Test with custom inputs
        #     passed = test_correctness(inputs_for_test)
        #     if not passed:
        #         print(f"✗ Custom input test failed")
        #         exit(1)
        #     test_passed = test_passed and passed
        #     if not passed:
        #         print(f"✗ Test custom failed")
        #         exit(1)

        # except FileNotFoundError:
        #     print("debug.pt not found, skipping custom input test")
        #     exit(1)


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

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print(f"Batch size: {batch_size}")
    print(f"Warmup iterations: {warmup}")
    print(f"Benchmark iterations: {repeats}")
    print("=" * 80)

    results = {}

    kernels = [
        selective_state_update,
        torch.ops.trtllm.selective_state_update_simple,
        torch.ops.trtllm.selective_state_update_producer_consumer,
        torch.ops.trtllm.selective_state_update_producer_consumer_writeback,
        torch.ops.trtllm.selective_state_update_producer_consumer_horizontal,
        torch.ops.trtllm.selective_state_update_producer_consumer_horizontal_warps,
    ]

    for kernel in kernels:
        kernel_name = f"{kernel.__module__}.{kernel.__name__}"
        avg_time = benchmark_performance(
            kernel, batch_size, nheads, dim, ngroups, dstate, repeats, warmup
        )
        results[kernel_name] = avg_time

    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective State Update Test Suite")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--repeat", type=int, default=100, help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size during benchmark")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=False,
        help="Skip correctness tests (default: False)",
    )
    args = parser.parse_args()

    results = main(args.batch_size, args.repeat, args.warmup, args.skip_test)

    if results:
        # print("\nBenchmark Results:")
        # for kernel_name, avg_time in results.items():
        #     print(f"  {kernel_name}: {avg_time:.3f} ms")
        exit(0)
    else:
        exit(1)
