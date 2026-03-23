#pragma once

#include <cstdint>

namespace core
{
    enum class LAYOUT_TYPE : std::uint32_t
    {
        SINGLE_CHANNEL = 1
        , DUAL_CHANNEL = 2
        , WHC_LAYOUT = 3
        , NHWC_LAYOUT = 4
        , NCHW_LAYOUT = 4
    };
}