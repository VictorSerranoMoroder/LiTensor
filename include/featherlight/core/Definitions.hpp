#pragma once

#include <cstdint>

namespace core
{
    enum class LAYOUT_TYPE : std::uint32_t
    {
        NCHW_LAYOUT = 4
        , NHWC_LAYOUT = 4
        , WHC_LAYOUT = 3
    };
}