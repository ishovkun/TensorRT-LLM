#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

namespace th = torch;

namespace torch_ext
{

auto run_selective_state_update(th::Tensor const& state, // (batch, dim, dstate) of (batch, nheads, dim, dstate)
    th::Tensor const& x,                                 // (batch, dim) or (batch, nheads, dim)
    th::Tensor const& dt,                                // (batch, dim) or (batch, nheads, dim)
    th::Tensor const& A,                                 // (dim, dstate) or (nheads, dim, dstate)
    th::Tensor const& B,                                 // (batch, dstate) or (batch, ngroups, dstate)
    th::Tensor const& C,                                 // (batch, dstate) or (batch, ngroups, dstate)
    th::Tensor const& D,                                 // (dim,) or (nheads, dim)
    // th::Tensor const& z,                                 // (batch, dim) or (batch, nheads, dim)
    th::Tensor const& dt_bias, // (dim,) or (nheads, dim)
    bool dt_softplus,
    // th::Tensor const& state_batch_indices, // (batch,)
    std::optional<th::Tensor> state_batch_indices, // (batch,)
    int64_t pad_slot_id                    // if cache_indices is passed, lets the kernel identify padded
                                           // entries that will not be processed,
    // for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
    // in this case, the kernel will not process entries at
    // indices 0 and 3
    ) -> th::Tensor // out: (batch, dim) or batch, nheads, dim
{
    // bool has_heads = state.dim() > 3;

    th::Tensor _state = state;
    th::Tensor _x = x;
    th::Tensor _dt = dt;
    th::Tensor _A = A;
    th::Tensor _B = B;
    th::Tensor _C = C;
    th::Tensor _D = D;
    // th::Tensor _z = z;
    th::Tensor _dt_bias = dt_bias;

    if (state.dim() == 3)
        _state = _state.unsqueeze(1);
    if (x.dim() == 2)
        _x = x.unsqueeze(1);
    if (dt.dim() == 2)
        _dt = dt.unsqueeze(1);
    if (A.dim() == 2)
        _A = A.unsqueeze(0);
    if (B.dim() == 2)
        _B = B.unsqueeze(1);

    if (C.dim() == 2)
        _C = C.unsqueeze(1);
    if (D.dim() == 1)
        _D = D.unsqueeze(0);

    // if (z.dim() == 2)
    //     _z = z.unsqueeze(1);
    if (dt_bias.dim() == 1)
        _dt_bias = dt_bias.unsqueeze(0);

    // auto const state_shape = _state.sizes();
    auto const batch = _x.size(0);
    auto const nheads = _state.size(1);
    auto const dim = _state.size(2);
    auto const dstate = _state.size(3);

    TORCH_CHECK(_x.size(0) == batch && _x.size(1) == nheads && _x.size(2) == dim, "x.shape must be (", batch, ", ",
        nheads, ", ", dim, ")");
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
    TORCH_CHECK(dt_bias.size(0) == nheads && dt_bias.size(1) == dim, "dt_bias.shape must be (", nheads, ", ", dim, ")");

    using namespace tensorrt_llm::kernels;
    SelectiveStateUpdateParams p;
    p.batch = batch;
    p.nheads = nheads;
    p.dim = dim;
    p.dstate = dstate;
    p.ngroups = ngroups;

    p.state = _state.data_ptr();
    p.x = _x.data_ptr();
    p.dt = _dt.data_ptr();
    p.dt_bias = _dt_bias.data_ptr();
    p.A = _A.data_ptr();
    p.B = _B.data_ptr();
    p.C = _C.data_ptr();
    p.D = _D.data_ptr();
    auto output = torch::empty_like(x);
    p.output = output.data_ptr();
    if (state_batch_indices) {
        TORCH_CHECK(state_batch_indices->size(0) == batch,
                    "state_batch_indices.shape must be (", batch, ")");
        p.state_batch_indices = state_batch_indices->data_ptr();
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto dtype = x.scalar_type();
    switch (dtype)
    {
    case torch::kFloat32: invokeSelectiveStateUpdate<float, float>(p, stream); break;
    case torch::kFloat16: invokeSelectiveStateUpdate<half, half>(p, stream); break;
    default:
        // Handle other data types
        // throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
        throw std::invalid_argument(
            "Invalid dtype: " + std::string(torch::toString(dtype)) + ". Only supports float16, float32, and bfloat16");
    }

    return output;
}

} // end namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "selective_state_update(Tensor state, Tensor x, Tensor dt, "
        "Tensor A, Tensor B, Tensor C, Tensor D, "
        // "Tensor z, "
        "Tensor dt_bias,"
        "bool dt_softplus,"
        "Tensor? state_batch_indices,"
        "int pad_slot_id"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("selective_state_update", &torch_ext::run_selective_state_update);
}
