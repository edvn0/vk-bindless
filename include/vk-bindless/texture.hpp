#pragma once

#include "allocator_interface.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

#include "vk-bindless/forward.hpp"
#include "vk-bindless/holder.hpp"

#include <memory>
#include <optional>
#include <vector>
#include <vulkan/vulkan_core.h>

#define MAKE_BIT_FIELD(E)                                                      \
  constexpr E operator|(const E lhs, const E rhs) {                            \
    const auto underlying_lhs = std::to_underlying(lhs);                       \
    const auto underlying_rhs = std::to_underlying(rhs);                       \
    return static_cast<E>(underlying_lhs | underlying_rhs);                    \
  }                                                                            \
  constexpr E operator&(const E lhs, const E rhs) {                            \
    const auto underlying_lhs = std::to_underlying(lhs);                       \
    const auto underlying_rhs = std::to_underlying(rhs);                       \
    return static_cast<E>(underlying_lhs & underlying_rhs);                    \
  }                                                                            \
  constexpr bool operator!(const E value) {                                    \
    return std::to_underlying(value) == 0;                                     \
  }                                                                            \
  constexpr bool operator==(const E lhs, const E rhs) {                        \
    return std::to_underlying(lhs) == std::to_underlying(rhs);                 \
  }                                                                            \
  constexpr bool operator!=(const E lhs, const E rhs) {                        \
    return std::to_underlying(lhs) != std::to_underlying(rhs);                 \
  }                                                                            \
  constexpr E &operator|=(E &lhs, const E rhs) {                               \
    lhs = lhs | rhs;                                                           \
    return lhs;                                                                \
  }                                                                            \
  constexpr E &operator&=(E &lhs, const E rhs) {                               \
    lhs = lhs & rhs;                                                           \
    return lhs;                                                                \
  }                                                                            \
  constexpr E operator~(const E value) {                                       \
    return static_cast<E>(~std::to_underlying(value));                         \
  }

namespace VkBindless {

struct TextureError {
  enum class Code : std::uint8_t {
    InvalidHandle,
    StaleHandle,
    IndexOutOfBounds
  };

  std::string message;
  Code code;
};

enum class TextureUsageFlags : std::uint8_t {
  TransferSource = 1 << 0,
  TransferDestination = 1 << 1,
  Sampled = 1 << 2,
  Storage = 1 << 3,
  ColourAttachment = 1 << 4,
  DepthStencilAttachment = 1 << 5,
  TransientAttachment = 1 << 6,
  InputAttachment = 1 << 7,
};
MAKE_BIT_FIELD(TextureUsageFlags);

struct VkTextureDescription {
  std::span<const std::uint8_t> data{}; // This can absolutely be empty, but if
                                        // it is not, it must be a valid image
  VkFormat format{VK_FORMAT_UNDEFINED};
  VkExtent3D extent{1, 1, 1};
  TextureUsageFlags usage_flags{TextureUsageFlags::Sampled |
                                TextureUsageFlags::TransferSource |
                                TextureUsageFlags::TransferDestination};
  std::uint32_t layers{1};
  std::optional<std::uint32_t> mip_levels{
      std::nullopt}; // If not set, it will be calculated from the extent
  VkSampleCountFlagBits sample_count{VK_SAMPLE_COUNT_1_BIT};
  VkImageTiling tiling{VK_IMAGE_TILING_OPTIMAL};
  VkImageLayout initial_layout{VK_IMAGE_LAYOUT_UNDEFINED};
  bool is_owning{true};
  bool is_swapchain{false};

  std::string_view debug_name;
};

class VkTexture {
public:
  VkTexture() = default;
  VkTexture(IContext &, const VkTextureDescription &);

  static auto create(IContext &, const VkTextureDescription &)
      -> Holder<TextureHandle>;

  [[nodiscard]] auto get_image_view() const -> const VkImageView & {
    return image_view;
  }
  [[nodiscard]] auto get_storage_image_view() const -> const VkImageView & {
    return storage_image_view;
  }
  [[nodiscard]] auto get_sampler() const -> const VkSampler & {
    return sampler;
  }
  [[nodiscard]] auto is_sampled() const -> bool { return sampled; }
  [[nodiscard]] auto is_storage() const -> bool { return storage; }
  [[nodiscard]] auto get_sample_count() const -> VkSampleCountFlagBits {
    return sample_count;
  }
  [[nodiscard]] auto get_image() const -> VkImage { return image; }
  [[nodiscard]] auto get_mip_layers_image_views() const {
    return std::span(mip_layer_views);
  }
  [[nodiscard]] auto owns_self() const -> bool { return image_owns_itself; }
  [[nodiscard]] auto is_swapchain_image() const { return is_swapchain; }

  auto create_image_view(VkDevice, const VkImageViewCreateInfo &) -> void;

private:
  VkImageView image_view{VK_NULL_HANDLE};
  VkImageView storage_image_view{VK_NULL_HANDLE};
  VkSampler sampler{VK_NULL_HANDLE};
  VkSampleCountFlagBits sample_count{VK_SAMPLE_COUNT_1_BIT};
  bool image_owns_itself{true};
  bool is_swapchain{false};

  std::vector<VkImageView> mip_layer_views;

  AllocationInfo image_allocation{};
  VkImage image{VK_NULL_HANDLE};

  bool sampled{false};
  bool storage{false};
};

class VkTextureSampler {
public:
  static auto create(IContext &, const VkSamplerCreateInfo &)
      -> Holder<SamplerHandle>;
};

} // namespace VkBindless