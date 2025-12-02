#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

namespace th = torch;

namespace torch_ext
{

using namespace tensorrt_llm::kernels;

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
    auto const nheads = state.size(1);
    auto const dim = state.size(2);
    auto const dstate = state.size(3);

    TORCH_CHECK(x.size(0) == batch && x.size(1) == nheads && x.size(2) == dim, "x.shape must be (", batch, ", ", nheads,
        ", ", dim, ")");
    TORCH_CHECK(dt.sizes() == x.sizes(), "dt.shape must match x.shape");
    TORCH_CHECK(A.size(0) == nheads && A.size(1) == dim && A.size(2) == dstate, "A.shape must be (", nheads, ", ", dim,
        ", ", dstate, ")");

    auto const ngroups = B.size(1);
    TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
    TORCH_CHECK(B.size(0) == batch && B.size(1) == ngroups && B.size(2) == dstate, "B.shape must be (", batch, ", ",
        ngroups, ", ", dstate, ")");
    TORCH_CHECK(C.sizes() == B.sizes(), "C.shape must match B.shape");

    TORCH_CHECK(D.size(0) == nheads && D.size(1) == dim, "D.shape must be (", nheads, ", ", dim, ")");
    // TORCH_CHECK(z.sizes() == x.sizes(), "z.shape must match x.shape");
    if (dt_bias)
    {
        TORCH_CHECK(
            dt_bias->size(0) == nheads && dt_bias->size(1) == dim, "dt_bias.shape must be (", nheads, ", ", dim, ")");
    }

    using namespace tensorrt_llm::kernels;
    SelectiveStateUpdateParams p;
    p.batch = batch;
    p.nheads = nheads;
    p.dim = dim;
    p.dstate = dstate;
    p.ngroups = ngroups;

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
    }
    p.A = A.data_ptr();
    p.B = B.data_ptr();
    p.C = C.data_ptr();
    p.D = D.data_ptr();
    auto output = torch::empty_like(x);
    p.output = output.data_ptr();

    if (state_batch_indices)
    {
        TORCH_CHECK(state_batch_indices->size(0) == batch, "state_batch_indices.shape must be (", batch, ")");
        p.state_batch_indices = state_batch_indices->data_ptr();
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto dtype = x.scalar_type();
    switch (dtype)
    {
    case torch::kFloat32: invokeSelectiveStateUpdate<float, float>(p, stream, kernel_type); break;
    case torch::kFloat16: invokeSelectiveStateUpdate<half, half>(p, stream, kernel_type); break;
    default:
        // Handle other data types
        throw std::invalid_argument(
            "Invalid dtype: " + std::string(torch::toString(dtype)) + ". Only supports float16, float32, and bfloat16");
    }

    return output;
}

auto run_selective_state_update_naive(th::Tensor const& state, th::Tensor const& x, th::Tensor const& dt,
    th::Tensor const& A, th::Tensor const& B, th::Tensor const& C, th::Tensor const& D, std::optional<th::Tensor> z,
    std::optional<th::Tensor> dt_bias, bool dt_softplus, std::optional<th::Tensor> state_batch_indices,
    int64_t pad_slot_id) -> th::Tensor
{
   return run_selective_state_update(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus, state_batch_indices, pad_slot_id,
       SelectiveStateUpdateKernelType::naive);
}

auto run_selective_state_update_opt(th::Tensor const& state, th::Tensor const& x, th::Tensor const& dt,
    th::Tensor const& A, th::Tensor const& B, th::Tensor const& C, th::Tensor const& D, std::optional<th::Tensor> z,
    std::optional<th::Tensor> dt_bias, bool dt_softplus, std::optional<th::Tensor> state_batch_indices,
    int64_t pad_slot_id) -> th::Tensor
{
   return run_selective_state_update(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus, state_batch_indices, pad_slot_id,
       SelectiveStateUpdateKernelType::optimized);
}

} // end namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "selective_state_update("
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
    m.impl("selective_state_update", &torch_ext::run_selective_state_update_naive);
    m.impl("selective_state_update_opt", &torch_ext::run_selective_state_update_opt);
}
