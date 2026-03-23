#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <numeric>

#include <vector>
#include <stdexcept>

#include "featherlight/core/Definitions.hpp"
#include "utils/Utils.hpp"
#include "utils/DataNormalization.hpp"

namespace core
{
    /// @brief Fixed-rank multidimensional tensor RAII container.
    ///
    /// @tparam TData Type of the elements stored in the tensor. Must be trivially copyable.
    /// @tparam Rank  Number of dimensions (rank) of the tensor.
    ///
    /// @details
    /// The `Tensor` class represents an N-dimensional array with contiguous storage.
    /// It computes strides automatically to allow multi-dimensional indexing using a flattened storage vector.
    /// Strides are computed in a row-major layout by default.
    ///
    /// @note Only trivially copyable types are supported for storage.
    ///
    template <class TData, LAYOUT_TYPE Rank, std::enable_if_t<std::is_trivially_copyable_v<TData>, bool> = true>
    class Tensor
    {
        public:
        /// @brief Constructs a tensor with the given shape and default-initialized elements.
        ///
        /// @param[in] shape Vector of dimension sizes. Must match the tensor's rank.
        /// @throws std::invalid_argument if shape is empty or does not match Rank.
        Tensor(const std::vector<std::uint32_t>& shape)
            : data_{}
            , shape_{shape}
            , strides_{}
        {
            if (shape.empty())
            {
                throw std::invalid_argument("Shape cannot be empty");
            }

            // Compute total size
            const auto size = std::accumulate(shape.begin(), shape.end(), uint32_t{1}, std::multiplies<uint32_t>());
            // Allocate contiguous storage
            data_.resize(size, TData{});

            // Compute strides for each dimension
            strides_ = utils::compute_tensor_strides(shape);

            if (strides_.size() != static_cast<std::uint32_t>(Rank))
            {
                throw std::invalid_argument("Tensor rank does not match with given dimensions");
            }
        }

        /// @brief Constructs a tensor with the given shape and prefilled data. (Copy)
        ///
        /// @param[in] shape Vector of dimension sizes. Must match the tensor's rank.
        /// @param[in] data  Pointer to contiguous data of size matching the product of shape elements.
        /// @throws std::invalid_argument if shape is empty, rank mismatch, or data pointer is null.
        Tensor(const std::vector<std::uint32_t>& shape, const std::vector<TData>& data)
            : data_{data}
            , shape_{shape}
            , strides_{}
        {
            if (shape.empty())
            {
                throw std::invalid_argument("Shape cannot be empty");
            }

            const auto expected_size =
            std::accumulate(shape.begin(), shape.end(),
                            std::uint32_t{1}, std::multiplies<std::uint32_t>());

            if (data_.size() != expected_size)
            {
                throw std::invalid_argument("Data vector does not correspond with given shape");
            }

            // Compute strides for each dimension
            strides_ = utils::compute_tensor_strides(shape_);

            if (strides_.size() != static_cast<std::uint32_t>(Rank))
            {
                throw std::invalid_argument("Tensor rank does not match with given dimensions");
            }
        }

        /// @brief Constructs a tensor with the given shape and given data (Move).
        ///
        /// @param[in] shape Vector of dimension sizes. Must match the tensor's rank.
        /// @param[in] data  Pointer to contiguous data of size matching the product of shape elements.
        /// @throws std::invalid_argument if shape is empty, rank mismatch, or data pointer is null.
        Tensor(const std::vector<std::uint32_t>& shape, std::vector<TData>&& data)
            : data_{std::move(data)}
            , shape_{shape}
            , strides_{}
        {
            if (shape.empty())
            {
                throw std::invalid_argument("Shape cannot be empty");
            }

            const auto expected_size =
            std::accumulate(shape.begin(), shape.end(),
                            std::uint32_t{1}, std::multiplies<std::uint32_t>());

            if (data_.size() != expected_size)
            {
                throw std::invalid_argument("Data vector does not correspond with given shape");
            }

            // Compute strides for each dimension
            strides_ = utils::compute_tensor_strides(shape_);

            if (strides_.size() != static_cast<std::uint32_t>(Rank))
            {
                throw std::invalid_argument("Tensor rank does not match with given dimensions");
            }
        }

        // Copyable
        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;

        // Movable
        Tensor(Tensor&&) = default;
        Tensor& operator=(Tensor&&) = default;

        /// @brief Set a new shape for the tensor.
        ///
        /// @param[in] shape New vector of dimension sizes. Must match the tensor's rank.
        /// @throws std::invalid_argument if shape size != Rank
        void set_shape(std::vector<std::uint32_t>& shape)
        {
            if (shape.size() != static_cast<std::uint32_t>(Rank))
            {
                throw std::invalid_argument("Tensor rank does not match with given dimensions");
            }
            shape_.swap(shape);
        }

        /// @brief Returns the tensor shape.
        ///
        /// @return Vector of dimension sizes.
        const std::vector<std::uint32_t> get_shape() const
        {
            return shape_;
        }

        /// @brief Access the internal 1D storage vector.
        ///
        /// @warning No data check is done whatsoever.
        ///
        /// @return Vector with data.
        std::vector<TData>& get_vector()
        {
            return data_;
        }

        /// @brief Access the internal 1D storage vector. (Constant)
        ///
        /// @return Constant vector with data.
        const std::vector<TData>& get_vector() const
        {
            return data_;
        }

        /// @brief Access the internal 1D storage vector.
        ///
        /// @warning No data check is done whatsoever.
        ///
        /// @return Flattened raw data ptr.
        TData* data()
        {
            return data_.data();
        }

        /// @brief Access the internal 1D raw data. (Constant)
        ///
        /// @return Constant raw data.
        const TData* data() const
        {
            return data_.data();
        }

        /// @brief Returns the total number of elements in the tensor.
        ///
        /// @return Data vector size
        std::uint32_t size() const
        {
            return data_.size();
        }

        /// @brief Direct access operator
        ///
        /// @param[in] index Vector of indices per dimension
        /// @throws std::out_of_range if any index is out of bounds
        /// @return Non constant reference to the element
        TData& operator[](const std::size_t index)
        {
            return data_.at(index);
        }

        /// @brief Direct access operator
        ///
        /// @param[in] index Vector of indices per dimension
        /// @throws std::out_of_range if any index is out of bounds
        /// @return Constant reference to the element
        const TData& operator[](const std::size_t index) const
        {
            return data_.at(index);
        }

        /// @brief Access an element using multi-dimensional indices.
        ///
        /// @param[in] indices Vector of indices per dimension
        /// @throws std::invalid_argument if index size != Rank
        /// @throws std::out_of_range if any index is out of bounds
        /// @return Non constant reference to the element
        TData& operator()(const std::vector<std::uint32_t>& indices)
        {
            return data_.at(compute_offset(indices));
        }

        /// @brief Access an element using multi-dimensional indices. (Constant)
        ///
        /// @param[in] indices Vector of indices per dimension
        /// @throws std::invalid_argument if index size != Rank
        /// @throws std::out_of_range if any index is out of bounds
        /// @return Constant reference to the element
        const TData& operator()(const std::vector<std::uint32_t>& indices) const
        {
            return data_.at(compute_offset(indices));
        }

        private:

        /// @brief Compute the offset in the flat vector from multi-dimensional indices.
        ///
        /// @param[in] indices Multi-dimensional indices
        /// @throws std::invalid_argument if index dimensionality does not match tensor rank
        /// @throws std::out_of_range if any index is out of bounds
        /// @return Linear offset into data_
        std::uint32_t compute_offset(const std::vector<std::uint32_t>& indices) const
        {
            if (indices.size() != shape_.size())
            {
                throw std::invalid_argument("Index dimensionality does not match tensor");
            }
            std::uint32_t offset {0};
            for (std::uint32_t i {0}; i < shape_.size(); ++i)
            {
                if (indices[i] >= shape_[i])
                {
                    throw std::out_of_range("Index out of bounds");
                }
                offset += indices[i] * strides_[i];
            }
            return offset;
        }

        ///@brief Flat contiguous storage of tensor elements
        std::vector<TData> data_;
        ///@brief Number of elements in each dimension
        std::vector<std::uint32_t> shape_;
        ///@brief Strides per dimension (distance in 1D vector between adjacent elements in that dimension)
        std::vector<std::uint32_t> strides_;
    };
}