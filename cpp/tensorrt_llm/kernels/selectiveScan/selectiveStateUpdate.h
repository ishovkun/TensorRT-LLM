#pragma once
#include "tensorrt_llm/common/cudaUtils.h"
#include <stdint.h>

namespace tensorrt_llm::kernels
{

enum class SelectiveStateUpdateKernelType
{
    simple,
    producer_consumer,
};

struct SelectiveStateUpdateParams
{
    uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{}, state_cache_size{};
    int32_t pad_slot_id{-1};

    bool dt_softplus{false};

    void* __restrict__ state{nullptr}; // input_t: (state_cache_size, nheads, dim, dstate)
    void* __restrict__ x{nullptr}; // input_t: (batch, nheads, dim)
    void* __restrict__ dt{nullptr}; // weight_t: (batch, nheads, dim)
    void* __restrict__ dt_bias{nullptr}; // weight_t (nheads, dim)
    void* __restrict__ A{nullptr}; // weight_t: (nheads, dim, dstate)
    void* __restrict__ B{nullptr}; // input_t: (batch, ngroups, dstate)
    void* __restrict__ C{nullptr}; // input_t: (batch, ngroups, dstate)
    void* __restrict__ D{nullptr}; // weight_t: (nheads, dim)
    void* __restrict__ z{nullptr}; // input_t: (batch, nheads, dim)
    void* __restrict__ output{nullptr}; // input_t: (batch, nheads, dim)
    void* __restrict__ state_batch_indices{nullptr}; // state_batch_indices: (batch,)
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t>
void invokeSelectiveStateUpdate(SelectiveStateUpdateParams& params, cudaStream_t stream,
    SelectiveStateUpdateKernelType = SelectiveStateUpdateKernelType::simple);

} // namespace tensorrt_llm::kernels
