#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace utils
{
    std::vector<float> normalize_to_float(std::uint8_t* input, int size)
    {
        std::vector<float> output {};
        output.reserve(size);

        constexpr double scale = 1.0 / 255.0;

        for (std::size_t i = 0; i < size; ++i)
        {
            output[i] = static_cast<float>(input[i]) * scale;
        }

        return output;
    }

    std::vector<std::uint8_t> denormalize_to_uint8(const float* input, std::size_t size)
    {
        std::vector<std::uint8_t> output(size);

        for (std::size_t i = 0; i < size; ++i)
        {
            float v = std::clamp(input[i], 0.0f, 1.0f);
            output[i] = static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        }

        return output;
    }
}