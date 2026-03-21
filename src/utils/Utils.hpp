#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

/// @brief Tensor core utilities
///
/// @details
///
namespace core::utils
{
    template<class TData>
    __host__ std::vector<std::uint32_t>
    compute_tensor_strides(const std::vector<TData>& shape)
    {
        std::vector<std::uint32_t> strides(shape.size());

        std::uint32_t stride = 1;

        for(int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    __host__ __device__ inline std::uint32_t
    compute_tensor_offset(
        const std::uint32_t* indices,
        const std::uint32_t* strides,
        std::uint32_t dims)
    {
        std::uint32_t offset = 0;

        for(std::uint32_t i = 0; i < dims; ++i)
        {
            offset += indices[i] * strides[i];
        }

        return offset;
    }

    __host__ inline std::uint32_t
    compute_tensor_offset(
        const std::vector<std::uint32_t>& indices,
        const std::vector<std::uint32_t>& strides)
    {
        return compute_tensor_offset(
            indices.data(),
            strides.data(),
            indices.size());
    }
}