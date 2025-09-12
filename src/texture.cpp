#include "vk-bindless/texture.hpp"

#include "ktx.h"
#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/object_pool.hpp"
#include "vk-bindless/types.hpp"
#include "vk-bindless/vulkan_context.hpp"

#include <assimp/texture.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <print>
#include <type_traits>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <stb_image.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef _MSC_VER
#define STBIW_SPRINTF sprintf_s
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include <stb_image_write.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace VkBindless {

template<typename T0, typename... Ts>
constexpr auto
max(T0&& first, Ts&&... rest)
{
  auto max_value = std::forward<T0>(first);
  ((max_value = max_value < rest ? rest : max_value), ...);
  return max_value;
}

namespace {
auto
make_ktx_copies(const ktxTexture2* ktx,
                uint32_t base_level,
                uint32_t level_count,
                uint32_t base_layer)
{
  const uint32_t levels =
    level_count ? level_count : ktx->numLevels - base_level;
  const uint32_t layers = ktx->numLayers;
  const uint32_t faces = ktx->numFaces ? ktx->numFaces : 1;

  std::vector<VkBufferImageCopy> copies;
  copies.reserve(levels * layers * faces);

  for (uint32_t ml = 0; ml < levels; ++ml) {
    const uint32_t level = base_level + ml;
    const uint32_t mip_w = std::max(1u, ktx->baseWidth >> level);
    const uint32_t mip_h = std::max(1u, ktx->baseHeight >> level);

    for (uint32_t la = 0; la < layers; ++la) {
      for (uint32_t fa = 0; fa < faces; ++fa) {
        ktx_size_t byte_offset = 0;
        KTX_error_code rc =
          ktxTexture_GetImageOffset(ktxTexture(const_cast<ktxTexture2*>(ktx)),
                                    level,
                                    la,
                                    fa,
                                    &byte_offset);
        if (rc != KTX_SUCCESS)
          continue;

        const uint32_t array_layer = base_layer + la * faces + fa;

        VkBufferImageCopy c{};
        c.bufferOffset = static_cast<VkDeviceSize>(byte_offset);
        c.bufferRowLength = 0;
        c.bufferImageHeight = 0;
        c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        c.imageSubresource.mipLevel = level;
        c.imageSubresource.baseArrayLayer = array_layer;
        c.imageSubresource.layerCount = 1;
        c.imageOffset = { 0, 0, 0 };
        c.imageExtent = { mip_w, mip_h, 1 };

        copies.push_back(c);
      }
    }
  }

  return copies;
}
}

auto
VkTexture::create_internal_image(IContext& ctx,
                                 const VkTextureDescription& description)
  -> void
{
  auto& allocator = ctx.get_allocator_implementation();

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format_to_vk_format(description.format);
  image_info.extent = description.extent;
  image_info.mipLevels = description.mip_levels.value_or(
    static_cast<std::uint32_t>(std::log2(max(description.extent.width,
                                             description.extent.height,
                                             description.extent.depth)) +
                               1));
  image_info.arrayLayers = description.layers;
  image_info.samples = description.sample_count;
  image_info.tiling = description.tiling;
  image_info.usage = static_cast<VkImageUsageFlags>(description.usage_flags);
  if (!description.data.empty()) {
    image_info.usage |=
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = description.initial_layout;
  image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

  assert(image_info.mipLevels > 0 && image_info.arrayLayers > 0);

  assert(!description.debug_name.empty());

  const AllocationCreateInfo alloc_info{
    .usage = MemoryUsage::AutoPreferDevice,
    .map_memory = false,
    .preferred_memory_bits = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    .required_memory_bits = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    .debug_name = std::string{ description.debug_name },
  };

  auto could_allocate = allocator.allocate_image(image_info, alloc_info);
  if (!could_allocate) {
    throw std::runtime_error("Failed to allocate image: " +
                             could_allocate.error().message);
  }

  auto&& [img, alloc] = std::move(could_allocate.value());
  image = img;
  image_allocation = alloc;
  mip_levels = image_info.mipLevels;
  array_layers = image_info.arrayLayers;

  if (!description.debug_name.empty()) {
    set_name_for_object(ctx.get_device(),
                        VK_OBJECT_TYPE_IMAGE,
                        image,
                        std::format("{}-[{}x{}]",
                                    description.debug_name,
                                    description.extent.width,
                                    description.extent.height));
  }

  const auto& vulkan_context = dynamic_cast<Context&>(ctx);

  const auto& data = description.data;
  if (!data.empty()) {
    static constexpr auto calculate_mip_levels =
      [](std::unsigned_integral auto width,
         std::unsigned_integral auto height) constexpr {
        using T = std::common_type_t<decltype(width), decltype(height)>;
        // Guard against zero (bit_width(0) is undefined)
        if (width == 0 || height == 0) {
          return T{ 0 };
        }

        const T max_dimension = std::max(width, height);

        return static_cast<T>(std::bit_width(max_dimension));
      };
    const auto buffer_row_length = 0;
    const auto computed_mip_levels =
      calculate_mip_levels(description.extent.width, description.extent.height);

    // Original single-image upload (unchanged)
    vulkan_context.staging_allocator->upload(
      *this,
      VkRect2D{
        .offset = { 0, 0 },
        .extent = { description.extent.width, description.extent.height } },
      0,
      1,
      0,
      description.layers,
      image_info.format,
      data.data(),
      buffer_row_length);

    if (mip_levels > 1 && computed_mip_levels > 1) {
      vulkan_context.staging_allocator->generate_mipmaps(*this,
                                                         get_extent().width,
                                                         get_extent().height,
                                                         mip_levels,
                                                         array_layers);
      set_layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    return;
  }

  if (description.fully_specified_data != nullptr) {
    auto ktx = description.fully_specified_data;

    auto* texture_data = ktxTexture_GetData(ktxTexture(ktx));
    auto size_bytes =
      ktxTexture_GetDataSize(ktxTexture(ktx)); // after transcode

    auto copies = make_ktx_copies(ktx, 0, ktx->numLevels, 0);

    vulkan_context.staging_allocator->upload(
      *this, texture_data, size_bytes, copies);
  }
}

VkTexture::VkTexture(IContext& ctx, const VkTextureDescription& description)
  : sample_count{ description.sample_count }
  , extent{ description.extent }
  , format{ description.format }
  , image_owns_itself{ description.is_owning }
  , is_swapchain{ description.is_swapchain }
  , sampled{ static_cast<bool>(description.usage_flags &
                               TextureUsageFlags::Sampled) }
  , storage{ static_cast<bool>(description.usage_flags &
                               TextureUsageFlags::Storage) }
  , is_depth{ static_cast<bool>(description.usage_flags &
                                TextureUsageFlags::DepthStencilAttachment) }
  , debug_name{ description.debug_name }
{
  assert(!debug_name.empty());

  if (!description.externally_created_image) {
    create_internal_image(ctx, description);
  } else {
    image = *description.externally_created_image;

    set_name_for_object(ctx.get_device(),
                        VK_OBJECT_TYPE_IMAGE,
                        image,
                        std::format("External_Image_{}-[{}x{}]",
                                    description.debug_name,
                                    description.extent.width,
                                    description.extent.height));
  }
  if (is_swapchain)
    return;

  // Lets create all views
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.pNext = nullptr;
  view_info.image = image;
  view_info.viewType = (description.extent.width == description.extent.height &&
                        description.layers == 6)
                         ? VK_IMAGE_VIEW_TYPE_CUBE
                         : VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format_to_vk_format(description.format);
  view_info.subresourceRange.aspectMask =
    is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = array_layers;
  view_info.components.r = VK_COMPONENT_SWIZZLE_R;
  view_info.components.g = VK_COMPONENT_SWIZZLE_G;
  view_info.components.b = VK_COMPONENT_SWIZZLE_B;
  view_info.components.a = VK_COMPONENT_SWIZZLE_A;

  VK_VERIFY(
    vkCreateImageView(ctx.get_device(), &view_info, nullptr, &image_view));
  set_name_for_object(ctx.get_device(),
                      VK_OBJECT_TYPE_IMAGE_VIEW,
                      image_view,
                      std::format("{} View", description.debug_name));

  mip_layer_views.resize(mip_levels * array_layers);

  // Less than two views means just one view for the entire image
  if (!(mip_levels == 1 && array_layers == 1)) {
    for (auto mip = 0U; mip < mip_levels; ++mip) {
      for (auto layer = 0U; layer < array_layers; ++layer) {
        const auto index = mip * array_layers + layer;
        view_info.subresourceRange.baseMipLevel = mip;
        view_info.subresourceRange.baseArrayLayer = layer;
        view_info.subresourceRange.layerCount = 1;
        view_info.subresourceRange.levelCount = 1;

        auto& mip_layer_view = mip_layer_views[index];
        VK_VERIFY(vkCreateImageView(
          ctx.get_device(), &view_info, nullptr, &mip_layer_view));

        set_name_for_object(
          ctx.get_device(),
          VK_OBJECT_TYPE_IMAGE_VIEW,
          mip_layer_view,
          std::format(
            "{} View Mip[{}] Layer[{}]", description.debug_name, mip, layer));
      }
    }
  }
}

auto
VkTexture::create(IContext& context, const VkTextureDescription& description)
  -> Holder<TextureHandle>
{
  auto& pool = context.get_texture_pool();
  auto handle = pool.create(VkTexture{
    context,
    description,
  });

  if (!handle.valid()) {
    return Holder<TextureHandle>::invalid();
  }

  context.needs_update() = true;

  return Holder{
    &context,
    std::move(handle),
  };
}

auto
load_image_file(const std::string_view path)
{
  // stbi
  auto stream = std::ifstream{ path.data(), std::ios::binary };
  if (!stream) {
    throw std::runtime_error("Failed to open file: " + std::string{ path });
  }
  std::int32_t width = 0;
  std::int32_t height = 0;
  std::int32_t channels = 0;

  auto buffer = std::vector<char>(std::istreambuf_iterator<char>(stream),
                                  std::istreambuf_iterator<char>());

  auto res =
    stbi_load_from_memory(reinterpret_cast<stbi_uc const*>(buffer.data()),
                          static_cast<int>(buffer.size()),
                          &width,
                          &height,
                          &channels,
                          STBI_rgb_alpha);
  if (!res) {
    throw std::runtime_error("Failed to load image: " + std::string{ path });
  }
  auto size = width * height * 4; // 4 channels (RGBA)
  std::vector<std::uint8_t> data(size);
  std::memcpy(data.data(), res, size);
  stbi_image_free(res);

  struct Output
  {
    std::vector<std::uint8_t> data;
    int width;
    int height;
    int channels;
  };

  return Output{
    .data = std::move(data),
    .width = width,
    .height = height,
    .channels = channels,
  };
}

auto
VkTexture::from_file(IContext& ctx,
                     const std::string_view path,
                     const VkTextureDescription& desc) -> Holder<TextureHandle>
{
  auto copy = desc;
  auto&& [data, w, h, channels] = load_image_file(path);
  if (data.empty())
    return Holder<TextureHandle>::invalid();
  copy.data = std::span(data);
  copy.extent = VkExtent3D{
    static_cast<std::uint32_t>(w),
    static_cast<std::uint32_t>(h),
    1,
  };

  return create(ctx, copy);
}

auto
VkTexture::from_memory(IContext& ctx,
                       const std::span<const std::uint8_t> bytes,
                       const VkTextureDescription& desc)
  -> Holder<TextureHandle>
{
  if (bytes.empty())
    return Holder<TextureHandle>::invalid();
  auto copy = desc;
  copy.data = bytes;
  return create(ctx, copy);
}

VkFilter
filter_mode_to_vk_filter_mode(FilterMode filter_mode)
{
  switch (filter_mode) {
    case FilterMode::Nearest:
      return VK_FILTER_NEAREST;
    case FilterMode::Linear:
      return VK_FILTER_LINEAR;
    default:
      throw std::invalid_argument("Unsupported filter mode");
  }
}

VkSamplerMipmapMode
filter_mode_to_vk_mip_map_mode(MipMapMode mip_map_mode)
{
  switch (mip_map_mode) {
    case MipMapMode::Nearest:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case MipMapMode::Linear:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    default:
      throw std::invalid_argument("Unsupported mip map mode");
  }
}

VkSamplerAddressMode
address_mode_to_vk_address_mode(WrappingMode wrapping_mode)
{
  switch (wrapping_mode) {
    case WrappingMode::Repeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case WrappingMode::MirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case WrappingMode::ClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case WrappingMode::ClampToBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case WrappingMode::MirrorClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default:
      throw std::invalid_argument("Unsupported wrapping mode");
  }
}
VkBorderColor
border_color_to_vk_border_color(BorderColor border_color)
{
  switch (border_color) {
    case BorderColor::FloatTransparentBlack:
      return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    case BorderColor::IntTransparentBlack:
      return VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
    case BorderColor::FloatOpaqueBlack:
      return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    case BorderColor::IntOpaqueBlack:
      return VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    case BorderColor::FloatOpaqueWhite:
      return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    case BorderColor::IntOpaqueWhite:
      return VK_BORDER_COLOR_INT_OPAQUE_WHITE;
    default:
      throw std::invalid_argument("Unsupported border color");
  }
}
auto
VkTextureSampler::create(IContext& context, const SamplerDescription& info)
  -> Holder<SamplerHandle>
{
  auto& pool = context.get_sampler_pool();
  VkSampler sampler{ VK_NULL_HANDLE };
  const VkSamplerCreateInfo sampler_info{
    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .magFilter = filter_mode_to_vk_filter_mode(info.mag_filter),
    .minFilter = filter_mode_to_vk_filter_mode(info.min_filter),
    .mipmapMode = filter_mode_to_vk_mip_map_mode(info.mipmap_mode),
    .addressModeU = address_mode_to_vk_address_mode(info.wrap_u),
    .addressModeV = address_mode_to_vk_address_mode(info.wrap_v),
    .addressModeW = address_mode_to_vk_address_mode(info.wrap_w),
    .mipLodBias = 0.0f,
    .anisotropyEnable = false,
    .maxAnisotropy = 1.0f,
    .compareEnable = info.compare_op.has_value(),
    .compareOp = info.compare_op.has_value()
                   ? static_cast<VkCompareOp>(*info.compare_op)
                   : VK_COMPARE_OP_NEVER,
    .minLod = info.min_lod,
    .maxLod = info.max_lod,
    .borderColor = border_color_to_vk_border_color(info.border_color),
    .unnormalizedCoordinates = false,
  };
  VK_VERIFY(
    vkCreateSampler(context.get_device(), &sampler_info, nullptr, &sampler));

  auto handle = pool.create(std::move(sampler));

  context.needs_update() = true;

  return Holder{
    &context,
    std::move(handle),
  };
}

auto
VkTexture::create_image_view(const VkDevice device,
                             const VkImageViewCreateInfo& view_info) -> void
{
  VkImageViewCreateInfo copy = view_info;
  copy.image = image;
  copy.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  VK_VERIFY(vkCreateImageView(device, &copy, nullptr, &image_view));
}
auto
VkTexture::write_hdr(std::string_view path,
                     const std::uint32_t width,
                     const std::uint32_t height,
                     const std::span<const float> data) -> bool
{
  return stbi_write_hdr(path.data(),
                        width,
                        height,
                        4, // 4 components
                        data.data());
}

auto
VkTexture::get_or_create_framebuffer_view(IContext& context,
                                          std::uint32_t mip,
                                          std::uint32_t layer) -> VkImageView
{
  if (mip >= max_mip_levels || layer >= cube_array_layers) {
    return VK_NULL_HANDLE;
  }

  if (VK_NULL_HANDLE !=
      cached_framebuffer_views.at(mip * cube_array_layers + layer)) {
    return cached_framebuffer_views.at(mip * cube_array_layers + layer);
  }
  const VkImageViewCreateInfo view_info{
    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .image = image,
    .viewType = (extent.width == extent.height && array_layers == 6)
                  ? VK_IMAGE_VIEW_TYPE_CUBE
                  : VK_IMAGE_VIEW_TYPE_2D,
    .format =format_to_vk_format( format),
    .components = {
      .r = VK_COMPONENT_SWIZZLE_R,
      .g = VK_COMPONENT_SWIZZLE_G,
      .b = VK_COMPONENT_SWIZZLE_B,
      .a = VK_COMPONENT_SWIZZLE_A,
    },
    .subresourceRange = {
      .aspectMask = is_depth
                     ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT)
                     : static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT),
      .baseMipLevel = mip,
      .levelCount = 1,
      .baseArrayLayer = layer,
      .layerCount = 1,
    },
  };

  VK_VERIFY(vkCreateImageView(
    context.get_device(),
    &view_info,
    nullptr,
    &cached_framebuffer_views[mip * cube_array_layers + layer]));

  const auto name =
    std::format("{}_FBView (mip: {}, layer: {})", debug_name, mip, layer);
  set_name_for_object(
    context.get_device(),
    VK_OBJECT_TYPE_IMAGE_VIEW,
    cached_framebuffer_views.at(mip * cube_array_layers + layer),
    name);

  return cached_framebuffer_views[mip * cube_array_layers + layer];
}

} // namespace VkBindless
