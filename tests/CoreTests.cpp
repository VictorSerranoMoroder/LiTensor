#include "gtest/gtest.h"
#include <gtest/gtest.h>

extern "C" {

    #define STB_IMAGE_IMPLEMENTATION
    #include <stb/stb_image.h>
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include <stb/stb_image_write.h>
}

#include <filesystem>

#include <featherlight/core/Tensor.hpp>
#include <featherlight/core/Definitions.hpp>

TEST(CoreTests, ImageAllocation)
{
    int width{}, height{}, channels{};

    const auto original_path = "/home/victor/workarea/Featherlight/tests/data/test_image.jpg";
    const auto result_path = "/home/victor/workarea/Featherlight/tests/data/result_image.jpg";

    // --- Load original image ---
    stbi_info(original_path, &width, &height, &channels);

    std::uint8_t* rawA =
        stbi_load(original_path, &width, &height, &channels, STBI_rgb);

    ASSERT_NE(rawA, nullptr);

    const std::uint32_t size = width * height * channels;

    core::Tensor<float,core::LAYOUT_TYPE::WHC_LAYOUT> tensorA{
        {
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            static_cast<std::uint32_t>(channels)
        },
        utils::normalize_to_float(rawA, size).data()
    };

    // --- Write back as PNG (lossless) ---
    auto denorm = utils::denormalize_to_uint8(tensorA.data(), size);

    stbi_write_png(result_path,
                   width,
                   height,
                   channels,
                   denorm.data(),
                   width * channels);

    stbi_image_free(rawA);

    // --- Load generated image ---
    std::uint8_t* rawB =
        stbi_load(result_path, &width, &height, &channels, STBI_rgb);

    ASSERT_NE(rawB, nullptr);

    core::Tensor<float, core::LAYOUT_TYPE::WHC_LAYOUT> tensorB{
        {static_cast<std::uint32_t>(width),
         static_cast<std::uint32_t>(height),
         static_cast<std::uint32_t>(channels)},
        utils::normalize_to_float(rawB, size).data()
    };

    stbi_image_free(rawB);

    // --- Compare tensors ---
    ASSERT_EQ(tensorA.size(), tensorB.size());

    for (std::uint32_t i = 0; i < tensorA.size(); ++i)
    {
        EXPECT_NEAR(tensorA.data()[i],
                    tensorB.data()[i],
                    1e-5f);
    }
}

TEST(CoreTests, TensorRankInvalid)
{
    int width{}, height{}, channels{};

    const auto original_path = "/home/victor/workarea/Featherlight/tests/data/test_image.jpg";
    const auto result_path = "/home/victor/workarea/Featherlight/tests/data/result_image.jpg";

    // --- Load original image ---
    stbi_info(original_path, &width, &height, &channels);

    std::uint8_t* rawA =
        stbi_load(original_path, &width, &height, &channels, STBI_rgb);

    ASSERT_NE(rawA, nullptr);

    const std::uint32_t size = width * height * channels;

    try
    {
        core::Tensor<float,core::LAYOUT_TYPE::NCHW_LAYOUT> tensorA
        {
            {
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                static_cast<std::uint32_t>(channels)
            },
            utils::normalize_to_float(rawA, size).data()
        };
        FAIL() << "Expected exception to be thrown, invalid ranks";
    }
    catch(const std::invalid_argument& e)
    {
        SUCCEED(); // Ranks are invalid, Tensor has risen exception as expected
    }
}