#pragma once
#include <stdint.h>
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels
{

struct SelectiveStateUpdateParams
{
    uint32_t batch, nheads, dim, dstate, ngroups;
    int32_t pad_slot_id{-1};

    bool dt_softplus, tie_hdim, has_state_batch_indices;

    void* __restrict__ state;
    void* __restrict__ x;
    void* __restrict__ dt;
    void* __restrict__ dt_bias;
    void* __restrict__ A;
    void* __restrict__ B;
    void* __restrict__ C;
    void* __restrict__ D;
    void* __restrict__ z{nullptr};
    void* __restrict__ output;
    void* __restrict__ state_batch_indices{nullptr};
};

template <typename input_t, typename weight_t>
void invokeSelectiveStateUpdate(SelectiveStateUpdateParams& params, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
