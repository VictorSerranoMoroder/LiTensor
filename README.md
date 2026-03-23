# LiTensor

LiTensor is a lightweight C++ tensor abstraction designed for high-performance computing and GPU (CUDA) interoperability.
This library provides a minimal foundation for working with N-dimensional data across host and device memory.

## Features

- N-dimensional tensor abstraction
- Explicit memory layout control
- Zero-copy views over existing memory
- GPU-ready design with device memory support
- Lightweight and composable utilities

---

## Core Concepts
Understanding these concepts is essential to correctly use LiTensor.

### Shape
Defines the size of each dimension:

```cpp
shape = {D0, D1, ..., DN-1}
size = D0 * D1 * ... * DN-1
```

### Indices
Access elements using N-dimensional indices:

```cpp
tensor({i, j, k});
```
Each index must be within bounds.

### Stride
Stride defines how memory is traversed for each dimension.

Example for shape {W, H, C} in WHC layout:
```cpp
stride[0] = 1            // W
stride[1] = W            // H
stride[2] = W * H        // C
```

### Offset
Maps N-dimensional indices to linear memory:
```cpp
offset = sum(indices[i] * stride[i])
```

### Memory layout
Layout determines how dimensions are ordered in memory:

| Layout | Order                                |
| ------ | -------------------------------------|
| WHC    | Width - Height - Channel             |
| HWC    | Height - Width - Channel             |
| CHW    | Channel → Height - Width             |
| NHWC   | N-element - Height - Width - Channel |
| NCHW   | N-element - Channel - Height - Width |


Layout mismatches are a common source of bugs and tensors are designed to be Rank-fixed, tensors of different Ranks are incompatible.

---

## Core Components

### `Tensor<TData, Rank>`
A **RAII** **host-only** N-dimensional dense tensor with contiguous storage. Only trivially copyable types (`TData`) are supported for efficiency and safety.


**Responsibilities**:
- Stores shape and stride
- Provides multi-dimensional indexing via 'operator()'
- Owns and manages the underlying memory

Example:
```cpp
 core::Tensor<float> tensorA{
        {static_cast<std::uint32_t>(width),
         static_cast<std::uint32_t>(height),
         static_cast<std::uint32_t>(channels)},
        utils::normalize_to_float(rawA, size).data()
    };
```
