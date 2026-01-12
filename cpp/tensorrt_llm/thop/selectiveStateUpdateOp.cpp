#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

namespace th = torch;

namespace torch_ext
{

using namespace tensorrt_llm::kernels;

void validate_shape(
    th::Tensor const& tensor, std::string const& tensor_name, std::vector<int64_t> const& expected_shape)
{
    // Check shape
    bool shape_matches = tensor.dim() == static_cast<int64_t>(expected_shape.size());
    if (shape_matches)
    {
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            if (tensor.size(i) != static_cast<int64_t>(expected_shape[i]))
            {
                shape_matches = false;
                break;
            }
        }
    }

    if (!shape_matches)
    {
        std::cerr << tensor_name << ".shape = [";
        for (int64_t i = 0; i < tensor.dim(); ++i)
        {
            std::cerr << tensor.size(i);
            if (i < tensor.dim() - 1)
                std::cerr << ", ";
        }
        std::cerr << "], expected shape = [";
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            std::cerr << expected_shape[i];
            if (i < expected_shape.size() - 1)
                std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        TORCH_CHECK(false && "Tensor shape mismatch");
    }
}

// Validate tensor shape and check if contiguous
void validate_contiguous_tensor(
    th::Tensor const& tensor, std::string const& tensor_name, std::vector<int64_t> const& expected_shape)
{
    validate_shape(tensor, tensor_name, expected_shape);

    // Check contiguity
    if (!tensor.is_contiguous())
    {
        std::cerr << tensor_name << " is not contiguous!" << std::endl;
        std::cerr << tensor_name << ".dim() = " << tensor.dim() << std::endl;
        std::cerr << tensor_name << ".shape = [";
        for (int64_t i = 0; i < tensor.dim(); ++i)
        {
            std::cerr << tensor.size(i);
            if (i < tensor.dim() - 1)
                std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        std::cerr << tensor_name << ".strides = [";
        for (int64_t i = 0; i < tensor.dim(); ++i)
        {
            std::cerr << tensor.stride(i);
            if (i < tensor.dim() - 1)
                std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        std::cerr << "expected shape = [";
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            std::cerr << expected_shape[i];
            if (i < expected_shape.size() - 1)
                std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;

        // Calculate expected contiguous strides
        std::cerr << "expected strides (contiguous) = [";
        if (!expected_shape.empty())
        {
            // Calculate total product
            int64_t product = 1;
            for (size_t i = 0; i < expected_shape.size(); ++i)
            {
                product *= expected_shape[i];
            }
            // Print strides by dividing from left to right
            for (size_t i = 0; i < expected_shape.size(); ++i)
            {
                product /= expected_shape[i];
                std::cerr << product;
                if (i < expected_shape.size() - 1)
                    std::cerr << ", ";
            }
        }
        std::cerr << "]" << std::endl;
    }

    TORCH_CHECK(tensor.is_contiguous());
}

// Validate tensor shape and strides
void validate_tensor(th::Tensor const& tensor, std::string const& tensor_name,
    std::vector<int64_t> const& expected_shape, std::vector<int64_t> const& expected_strides)
{
    // Check shape
    bool shape_matches = tensor.dim() == static_cast<int64_t>(expected_shape.size());
    if (shape_matches)
    {
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            if (tensor.size(i) != static_cast<int64_t>(expected_shape[i]))
            {
                shape_matches = false;
                break;
            }
        }
    }

    if (!shape_matches)
    {
        std::cerr << tensor_name << ".shape = [";
        for (int64_t i = 0; i < tensor.dim(); ++i)
        {
            std::cerr << tensor.size(i);
            if (i < tensor.dim() - 1)
                std::cerr << ", ";
        }
        std::cerr << "], expected shape = [";
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            std::cerr << expected_shape[i];
            if (i < expected_shape.size() - 1)
                std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        TORCH_CHECK(false && "Tensor shape mismatch");
    }

    // Check strides
    bool strides_match = tensor.dim() == static_cast<int64_t>(expected_strides.size());
    if (strides_match)
    {
        for (size_t i = 0; i < expected_strides.size(); ++i)
        {
            if (tensor.stride(i) != static_cast<int64_t>(expected_strides[i]))
            {
                strides_match = false;
                break;
            }
        }
    }

    if (!strides_match)
    {
        std::cerr << tensor_name << ".strides = [";
        for (int64_t i = 0; i < tensor.dim(); ++i)
        {
            std::cerr << tensor.stride(i);
            if (i < tensor.dim() - 1)
                std::cerr << ", ";
        }
        std::cerr << "], expected strides = [";
        for (size_t i = 0; i < expected_strides.size(); ++i)
        {
            std::cerr << expected_strides[i];
            if (i < expected_strides.size() - 1)
                std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        TORCH_CHECK(false && "Tensor strides mismatch");
    }
}

template <size_t N>
void copy_strides(th::Tensor const& tensor, std::array<int64_t, N>& strides_array)
{
    TORCH_CHECK(strides_array.size() == static_cast<size_t>(tensor.dim()), "Stride array size (", strides_array.size(),
        ") must match tensor dimensions (", tensor.dim(), ")");

    for (size_t i = 0; i < N; ++i)
    {
        strides_array[i] = tensor.stride(i);
    }
}

auto run_selective_state_update(                   //
    th::Tensor const& state,                       // (state_cache_size, nheads, dim, dstate)
    th::Tensor const& x,                           // (batch, nheads, dim)
    th::Tensor const& dt,                          // (batch, nheads, dim)
    th::Tensor const& A,                           // (nheads, dim, dstate)
    th::Tensor const& B,                           // (batch, ngroups, dstate)
    th::Tensor const& C,                           // (batch, ngroups, dstate)
    th::Tensor const& D,                           // (nheads, dim)
    std::optional<th::Tensor> z,                   // (batch, nheads, dim)
    std::optional<th::Tensor> dt_bias,             // (nheads, dim)
    bool dt_softplus,                              //
    std::optional<th::Tensor> state_batch_indices, // (batch,)
    int64_t pad_slot_id,                           // padded entries
    SelectiveStateUpdateKernelType kernel_type     // kernel type
    ) -> th::Tensor                                // out: (batch, dim) or (batch, nheads, dim)
{
    /* if cache_indices is passed, lets the kernel identify padded
    // entries that will not be processed,
    // for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
    // in this case, the kernel will not process entries at
    // indices 0 and 3 */

    auto const batch = x.size(0);
    auto const state_cache_size = state.size(0);
    auto const nheads = state.size(1);
    auto const dim = state.size(2);
    auto const dstate = state.size(3);
    auto const ngroups = B.size(1);

    TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");

    using namespace tensorrt_llm::kernels;
    SelectiveStateUpdateParams p;

    validate_shape(x, "x", {batch, nheads, dim});
    p.x_stride_batch = x.stride(0);
    TORCH_CHECK(x.stride(1) == dim);
    TORCH_CHECK(x.stride(2) == 1);

    validate_shape(dt, "dt", {batch, nheads, dim});
    p.dt_stride_batch = dt.stride(0);
    TORCH_CHECK(dt.stride(1) == 1);
    TORCH_CHECK(dt.stride(2) == 0);

    validate_contiguous_tensor(state, "state", {state_cache_size, nheads, dim, dstate});

    validate_shape(B, "B", {batch, B.size(1), dstate});
    p.B_stride_batch = B.stride(0);
    TORCH_CHECK(B.stride(1) == dstate);
    TORCH_CHECK(B.stride(2) == 1);

    validate_shape(C, "C", {batch, C.size(1), dstate});
    p.C_stride_batch = C.stride(0);
    TORCH_CHECK(C.stride(1) == dstate);
    TORCH_CHECK(C.stride(2) == 1);

    validate_tensor(D, "D", {nheads, dim}, {1, 0});
    validate_tensor(A, "A", {nheads, dim, dstate}, {1, 0, 0});

    if (z)
    {
        auto& z_tensor = z.value();
        TORCH_CHECK(z_tensor.is_cuda(), "z must be a CUDA tensor");
        TORCH_CHECK(z_tensor.dim() == 3, "z must have 3 dimensions");
        TORCH_CHECK(z_tensor.size(0) == batch, "z.size(0) must equal batch");
        TORCH_CHECK(z_tensor.size(1) == nheads, "z.size(1) must equal nheads");
        TORCH_CHECK(z_tensor.size(2) == dim, "z.size(2) must equal dim");
        TORCH_CHECK(z_tensor.stride(2) == 1, "z must be contiguous in the last dimension");
        TORCH_CHECK(
            z_tensor.stride(1) == dim, "z.stride(1) must equal dim, got ", z_tensor.stride(1), " expected ", dim);
    }

    if (dt_bias)
    {
        validate_tensor(*dt_bias, "dt_bias", {nheads, dim}, {1, 0});
    }

    p.batch = batch;
    p.nheads = nheads;
    p.dim = dim;
    p.dstate = dstate;
    p.ngroups = ngroups;
    p.state_cache_size = state_cache_size;
    p.pad_slot_id = pad_slot_id;

    p.state = state.data_ptr();
    p.x = x.data_ptr();
    p.dt = dt.data_ptr();
    if (dt_bias)
    {
        p.dt_bias = dt_bias->data_ptr();
    }
    if (z)
    {
        p.z = z->data_ptr();
        p.z_stride_batch = z->stride(0);
    }

    p.A = A.data_ptr();
    p.B = B.data_ptr();
    p.C = C.data_ptr();
    p.D = D.data_ptr();
    p.dt_softplus = dt_softplus;

    auto output = torch::empty_like(x);
    p.out_stride_batch = output.stride(0);
    p.output = output.data_ptr();
    TORCH_CHECK(output.is_contiguous());

    if (state_batch_indices)
    {
        TORCH_CHECK(state_batch_indices->size(0) == batch, "state_batch_indices.shape must be (", batch, ")");
        p.state_batch_indices = state_batch_indices->data_ptr();
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto input_dtype = x.scalar_type();
    auto weight_dtype = dt.scalar_type();

    TORCH_CHECK(state.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(D.scalar_type() == weight_dtype);
    if (dt_bias)
    {
        TORCH_CHECK(dt_bias->scalar_type() == weight_dtype);
    }

    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && "A must be float32");
    TORCH_CHECK(state.scalar_type() == input_dtype && "For now, state must have the same dtype as x");
    TORCH_CHECK(B.scalar_type() == input_dtype && C.scalar_type() == input_dtype);

    switch (input_dtype)
    {
    // case torch::kFloat32: invokeSelectiveStateUpdate<float, float>(p, stream, kernel_type); break;
    case torch::kBFloat16:
        invokeSelectiveStateUpdate<__nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16>(p, stream, kernel_type);
        break;

    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype: " + std::string(torch::toString(input_dtype))
            + ". Only supports float16, float32, and bfloat16");
    }

    return output;
}

auto run_selective_state_update_simple(th::Tensor const& state, th::Tensor const& x, th::Tensor const& dt,
    th::Tensor const& A, th::Tensor const& B, th::Tensor const& C, th::Tensor const& D, std::optional<th::Tensor> z,
    std::optional<th::Tensor> dt_bias, bool dt_softplus, std::optional<th::Tensor> state_batch_indices,
    int64_t pad_slot_id) -> th::Tensor
{
    return run_selective_state_update(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus, state_batch_indices,
        pad_slot_id, SelectiveStateUpdateKernelType::simple);
}

auto run_selective_state_update_producer_consumer_vertical(th::Tensor const& state, th::Tensor const& x,
    th::Tensor const& dt, th::Tensor const& A, th::Tensor const& B, th::Tensor const& C, th::Tensor const& D,
    std::optional<th::Tensor> z, std::optional<th::Tensor> dt_bias, bool dt_softplus,
    std::optional<th::Tensor> state_batch_indices, int64_t pad_slot_id) -> th::Tensor
{
    return run_selective_state_update(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus, state_batch_indices,
        pad_slot_id, SelectiveStateUpdateKernelType::producer_consumer_vertical);
}

auto run_selective_state_update_producer_consumer_horizontal(th::Tensor const& state, th::Tensor const& x,
    th::Tensor const& dt, th::Tensor const& A, th::Tensor const& B, th::Tensor const& C, th::Tensor const& D,
    std::optional<th::Tensor> z, std::optional<th::Tensor> dt_bias, bool dt_softplus,
    std::optional<th::Tensor> state_batch_indices, int64_t pad_slot_id) -> th::Tensor
{
    return run_selective_state_update(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus, state_batch_indices,
        pad_slot_id, SelectiveStateUpdateKernelType::producer_consumer_horizontal);
}

} // end namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "selective_state_update_simple("
        "Tensor state, Tensor x, Tensor dt, "
        "Tensor A, Tensor B, Tensor C, Tensor D, "
        "Tensor? z, "
        "Tensor? dt_bias,"
        "bool dt_softplus,"
        "Tensor? state_batch_indices,"
        "int pad_slot_id"
        ") -> Tensor");
    m.def(
        "selective_state_update_producer_consumer_vertical("
        "Tensor state, Tensor x, Tensor dt, "
        "Tensor A, Tensor B, Tensor C, Tensor D, "
        "Tensor? z, "
        "Tensor? dt_bias,"
        "bool dt_softplus,"
        "Tensor? state_batch_indices,"
        "int pad_slot_id"
        ") -> Tensor");
    m.def(
        "selective_state_update_producer_consumer_horizontal("
        "Tensor state, Tensor x, Tensor dt, "
        "Tensor A, Tensor B, Tensor C, Tensor D, "
        "Tensor? z, "
        "Tensor? dt_bias,"
        "bool dt_softplus,"
        "Tensor? state_batch_indices,"
        "int pad_slot_id"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("selective_state_update_simple", &torch_ext::run_selective_state_update_simple);
    m.impl("selective_state_update_producer_consumer_vertical",
        &torch_ext::run_selective_state_update_producer_consumer_vertical);
    m.impl("selective_state_update_producer_consumer_horizontal",
        &torch_ext::run_selective_state_update_producer_consumer_horizontal);
}
