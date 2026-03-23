#pragma once

#include "core/utils/Utils.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace core
{
    /// @brief Dynamic TensorView for tensor access
    ///
    /// @details
    /// CUDA device code can use parts of the C++ standard library, but only if they:
    ///    -do not allocate memory
    ///    -do not depend on host-only APIs
    ///    -are compiled for device
    ///
    /// GPU kernels operate best on plain data structures containing
    /// raw pointers and trivially copyable types.
    ///
    /// So any std container will not be able to function properly inside a device kernel.
    /// Also many cpp OOP features like inheritance and virtual functions are not allowed in CUDA device kernel code
    ///  (this is not true anymore but virtual dispatch and vtable lookups is quite slow on GPUs).
    /// Thus, classes that are expected to be used in device must adhere to this rules.
    ///
    /// TensorView is a lightweight, non-owning structure that provides
    /// access to tensor data allocated in both device and host memory organized in N Ranks or dimensions.
    /// It contains:
    /// - a pointer to the tensor data
    /// - stride information
    /// This structure is designed to be trivially copyable so that it can be passed directly to CUDA kernels.
    ///
    /// @note
    /// TensorView does not manage memory and should typically be created from a host-side tensor object
    /// responsible for allocation and synchronization with the GPU.
    template <class TData, std::uint32_t Rank>
    struct TensorView
    {
        TData* __restrict__ data;
        std::uint32_t* strides;

        __device__ __forceinline__
        TData& operator()(const std::uint32_t* indices)
        {
            return data[compute_tensor_offset(indices)];
        }

        __device__ __forceinline__
        std::uint32_t compute_tensor_offset(const std::uint32_t* indices) const
        {
            std::uint32_t offset = 0;

            #pragma unroll
            for(std::uint32_t i = 0; i < Rank; ++i)
            {
                offset += indices[i] * strides[i];
            }

            return offset;
        }
    };


    template<class TData>
    struct TensorView<TData, 1>
    {
        TData* __restrict__ data;
        std::uint32_t* strides;

        __device__ __forceinline__
        TData& operator()(uint32_t i)
        {
            return data[i*strides[0]];
        }
    };

    template<class TData>
    struct TensorView<TData, 2>
    {
        TData* __restrict__ data;
        std::uint32_t* strides;

        __device__ __forceinline__
        TData& operator()(uint32_t i, uint32_t j)
        {
            return data[
                i*strides[0] +
                j*strides[1]
            ];
        }
    };

    template<class TData>
    struct TensorView<TData, 4>
    {
        TData* __restrict__ data;
        std::uint32_t* strides;

        __device__ __forceinline__
        TData& operator()(
            uint32_t n,
            uint32_t c,
            uint32_t h,
            uint32_t w)
        {
            return data[
                n*strides[0] +
                c*strides[1] +
                h*strides[2] +
                w*strides[3]
            ];
        }
    };
}