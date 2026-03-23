#include "gtest/gtest.h"
#include <gtest/gtest.h>

extern "C" {

    #define STB_IMAGE_IMPLEMENTATION
    #include <stb/stb_image.h>
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include <stb/stb_image_write.h>
}

#include <filesystem>
#include <utility>

#include <featherlight/core/Tensor.hpp>
#include <featherlight/core/Definitions.hpp>

TEST(CoreTests, BasicAccess)
{
    std::vector<float> values{1,2,3,4,5,6,7,8,9};
    core::Tensor<float, core::LAYOUT_TYPE::DUAL_CHANNEL> tensorA{
        {3,3},
        std::move(values)
    };

    // Values should have been emptied
    ASSERT_TRUE(values.empty());

    // Linear access
    ASSERT_EQ(tensorA[0], 1);
    ASSERT_EQ(tensorA[4], 5);
    ASSERT_EQ(tensorA[8], 9);

    // Multi-dimensional access
    ASSERT_EQ(tensorA({0,0}), 1);
    ASSERT_EQ(tensorA({0,1}), 2);
    ASSERT_EQ(tensorA({1,0}), 4);
    ASSERT_EQ(tensorA({2,2}), 9);

    //Check write
    tensorA({0,0}) = 42;
    ASSERT_EQ(tensorA({0,0}), 42);

    // Check bounds
    ASSERT_THROW(tensorA[100], std::out_of_range);
    ASSERT_THROW(tensorA({5,0}), std::out_of_range);
    ASSERT_THROW(tensorA({0}), std::invalid_argument);
}

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

    // Update to required channels
    channels = STBI_rgb;

    const std::uint32_t size = width * height * channels;
    std::vector<float> normA {utils::normalize_to_float(rawA, size)};
    ASSERT_EQ(normA.size(), size);

    core::Tensor<float,core::LAYOUT_TYPE::WHC_LAYOUT> tensorA{
        {
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            static_cast<std::uint32_t>(channels)
        },
        std::move(normA)
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
        std::move(utils::normalize_to_float(rawB, size))
    };

    stbi_image_free(rawB);

    // --- Compare tensors ---
    ASSERT_EQ(tensorA.size(), tensorB.size());
    for (std::uint32_t i = 0; i < tensorA.size(); ++i)
    {
        EXPECT_NEAR(tensorA[i],
                    tensorB[i],
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
            utils::normalize_to_float(rawA, size)
        };
        FAIL() << "Expected exception to be thrown, invalid ranks";
    }
    catch(const std::invalid_argument& e)
    {
        SUCCEED(); // Ranks are invalid, Tensor has risen exception as expected
    }
}