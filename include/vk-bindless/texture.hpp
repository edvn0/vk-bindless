#pragma once

#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/common.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

#include "vk-bindless/common.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/holder.hpp"

#include <memory>
#include <optional>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace VkBindless {

static constexpr auto max_mip_levels = 15ULL; // A ~33k texture
static constexpr auto cube_array_layers = 6ULL;

struct TextureError
{
  enum class Code : std::uint8_t
  {
    InvalidHandle,
    StaleHandle,
    IndexOutOfBounds
  };

  std::string message;
  Code code;
};

enum class TextureUsageFlags : std::uint8_t
{
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

struct VkTextureDescription
{
  std::span<const std::uint8_t> data{}; // This can absolutely be empty, but if
                                        // it is not, it must be a valid image
  Format format{ Format::Invalid };
  VkExtent3D extent{ 1, 1, 1 };
  TextureUsageFlags usage_flags{ TextureUsageFlags::Sampled |
                                 TextureUsageFlags::TransferSource |
                                 TextureUsageFlags::TransferDestination };
  std::uint32_t layers{ 1 };
  std::optional<std::uint32_t> mip_levels{
    std::nullopt
  }; // If not set, it will be calculated from the extent
  VkSampleCountFlagBits sample_count{ VK_SAMPLE_COUNT_1_BIT };
  VkImageTiling tiling{ VK_IMAGE_TILING_OPTIMAL };
  VkImageLayout initial_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  std::optional<VkImageLayout> final_layout{
    std::nullopt
  }; // If not set, it will be VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL if
     // sampled, VK_IMAGE_LAYOUT_GENERAL if storage, else VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  bool is_owning{ true };
  bool is_swapchain{ false };

  std::optional<VkImage> externally_created_image{ std::nullopt };

  std::string_view debug_name;
};

class VkTexture
{
public:
  VkTexture() = default;
  VkTexture(IContext&, const VkTextureDescription&);

  static auto from_file(IContext&, std::string_view,const VkTextureDescription&) -> Holder<TextureHandle>;
  static auto from_memory(IContext&, std::span<const std::uint8_t>,const VkTextureDescription&) -> Holder<TextureHandle>;
  static auto create(IContext&, const VkTextureDescription&)
    -> Holder<TextureHandle>;

  [[nodiscard]] auto get_image_view() const -> const VkImageView&
  {
    return image_view;
  }
  [[nodiscard]] auto get_storage_image_view() const -> const VkImageView&
  {
    return storage_image_view;
  }
  [[nodiscard]] auto get_sampler() const -> const VkSampler& { return sampler; }
  [[nodiscard]] auto is_sampled() const -> bool { return sampled; }
  [[nodiscard]] auto is_storage() const -> bool { return storage; }
  [[nodiscard]] auto get_sample_count() const -> VkSampleCountFlagBits
  {
    return sample_count;
  }
  [[nodiscard]] auto get_image() const -> VkImage { return image; }
  [[nodiscard]] auto get_mip_layers_image_views() const
  {
    return std::span(mip_layer_views);
  }
  [[nodiscard]] auto owns_self() const -> bool { return image_owns_itself; }
  [[nodiscard]] auto is_swapchain_image() const { return is_swapchain; }
  [[nodiscard]] auto get_image_aspect_flags() const -> VkImageAspectFlags
  {
    return image_aspect_flags;
  }
  [[nodiscard]] auto get_extent() const -> VkExtent3D { return extent; }
  [[nodiscard]] auto get_framebuffer_views() const
    -> std::span<const VkImageView>
  {
    return std::span(cached_framebuffer_views);
  }
  [[nodiscard]] auto get_mip_layer_views() const -> std::span<const VkImageView>
  {
    return std::span(mip_layer_views);
  }

  auto get_or_create_framebuffer_view(IContext&,
                                      std::uint32_t mip,
                                      std::uint32_t layer) -> VkImageView;
  auto create_image_view(VkDevice, const VkImageViewCreateInfo&) -> void;

  [[nodiscard]] auto get_format() const -> Format { return format; }
  [[nodiscard]] auto is_swapchain_texture() const -> bool
  {
    return is_swapchain;
  }

  [[nodiscard]] auto get_layout() const -> VkImageLayout { return current_layout; }
  auto set_layout(const VkImageLayout layout) -> void { current_layout = layout; }

  static auto write_hdr(std::string_view path,
                        std::uint32_t width,
                        std::uint32_t height,
                        std::span<const float> data) -> bool;

private:
  VkImageView image_view{ VK_NULL_HANDLE };
  VkImageView storage_image_view{ VK_NULL_HANDLE };
  VkSampler sampler{ VK_NULL_HANDLE };
  VkImageLayout current_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  VkSampleCountFlagBits sample_count{ VK_SAMPLE_COUNT_1_BIT };
  VkImageAspectFlags image_aspect_flags{ VK_IMAGE_ASPECT_COLOR_BIT };
  VkExtent3D extent{ 1, 1, 1 };
  Format format{ Format::Invalid };
  bool image_owns_itself{ true };
  bool is_swapchain{ false };
  std::uint32_t mip_levels{ 1 };
  std::uint32_t array_layers{ 0 };

  std::vector<VkImageView> mip_layer_views;

  AllocationInfo image_allocation{};
  VkImage image{ VK_NULL_HANDLE };

  std::array<VkImageView, max_mip_levels * cube_array_layers>
    cached_framebuffer_views{};

  bool sampled{ false };
  bool storage{ false };
  bool is_depth{ false };

  auto create_internal_image(IContext&, const VkTextureDescription&) -> void;
};

enum class WrappingMode : std::uint8_t
{
  Repeat = 0,
  MirroredRepeat = 1,
  ClampToEdge = 2,
  ClampToBorder = 3,
  MirrorClampToEdge = 4,
};

enum class FilterMode : std::uint8_t
{
  Nearest = 0,
  Linear = 1,
};

enum class BorderColor : std::uint8_t
{
  FloatTransparentBlack = 0,
  IntTransparentBlack = 1,
  FloatOpaqueBlack = 2,
  IntOpaqueBlack = 3,
  FloatOpaqueWhite = 4,
  IntOpaqueWhite = 5,
};

using MipMapMode = FilterMode;

struct SamplerDescription
{
  WrappingMode wrap_u { WrappingMode::Repeat };
  WrappingMode wrap_v { WrappingMode::Repeat };
  WrappingMode wrap_w { WrappingMode::Repeat };
  FilterMode min_filter { FilterMode::Linear };
  FilterMode mag_filter { FilterMode::Linear };
  MipMapMode mipmap_mode { MipMapMode::Linear };
  float min_lod { 0.0f };
  float max_lod { 1.0f };
  BorderColor border_color { BorderColor::FloatOpaqueBlack };
  std::optional<CompareOp> compare_op { std::nullopt };
};

class VkTextureSampler
{
public:
  static auto create(IContext&, const SamplerDescription&)
    -> Holder<SamplerHandle>;
};

} // namespace VkBindless