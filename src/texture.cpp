#include "vk-bindless/texture.hpp"

#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/object_pool.hpp"
#include "vk-bindless/types.hpp"
#include "vk-bindless/vulkan_context.hpp"

#include <cmath>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

namespace VkBindless {

template<typename T0, typename... Ts>
constexpr auto
max(T0&& first, Ts&&... rest)
{
  auto max_value = std::forward<T0>(first);
  ((max_value = max_value < rest ? rest : max_value), ...);
  return max_value;
}

auto VkTexture::create_internal_image(IContext& ctx,
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
                                             description.extent.depth)) + 1));
  image_info.arrayLayers = description.layers;
  image_info.samples = description.sample_count;
  image_info.tiling = description.tiling;
  image_info.usage = static_cast<VkImageUsageFlags>(description.usage_flags);
  if (!description.data.empty()) {
    image_info.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = description.initial_layout;
  image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

  assert(image_info.mipLevels > 0 && image_info.arrayLayers > 0);

  if (description.debug_name.empty()) {
    set_name_for_object(ctx.get_device(),
                        VK_OBJECT_TYPE_IMAGE,
                        image,
                        std::format("{}-[{}x{}]",
                                    description.debug_name,
                                    description.extent.width,
                                    description.extent.height));
  }

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

  const auto& data = description.data;
  if (data.empty()) return;

  VkFormatProperties props{};
  vkGetPhysicalDeviceFormatProperties(ctx.get_physical_device(), image_info.format, &props);
  if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
    throw std::runtime_error("Format does not support linear blit for mip generation");
  }

  auto mip_extent = [](VkExtent3D e, std::uint32_t level) -> VkExtent3D {
    return VkExtent3D{
      .width  = std::max(1u, e.width  >> level),
      .height = std::max(1u, e.height >> level),
      .depth  = std::max(1u, e.depth  >> level),
    };
  };

  auto staging_buffer = VkDataBuffer::create(
    ctx,
    BufferDescription{
      .size = data.size_bytes(),
      .storage = StorageType::DeviceLocal,
      .usage = BufferUsageFlags::TransferSrc | BufferUsageFlags::TransferDst,
      .debug_name = std::format("StagingBuffer for Texture '{}'", description.debug_name),
    });

  auto get_buffer = [](auto& context, auto& buffer) {
    if (const VkDataBuffer* b = *context.get_buffer_pool().get(buffer)) {
      return b->get_buffer();
    }
    return static_cast<VkBuffer>(VK_NULL_HANDLE);
  };

  if (auto mapped = ctx.get_mapped_pointer(*staging_buffer); mapped) {
    std::memcpy(mapped, data.data(), data.size_bytes());
    ctx.flush_mapped_memory(*staging_buffer, 0, data.size_bytes());
  } else {
    throw std::runtime_error("Failed to map staging buffer");
  }

  auto& cmd = ctx.acquire_command_buffer();
  auto cb = dynamic_cast<CommandBuffer&>(cmd).get_command_buffer();

  auto barrier = [&](VkImageLayout old_layout,
                     VkImageLayout new_layout,
                     VkPipelineStageFlags2 src_stage,
                     VkPipelineStageFlags2 dst_stage,
                     VkAccessFlags2 src_access,
                     VkAccessFlags2 dst_access,
                     std::uint32_t base_mip,
                     std::uint32_t level_count) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = src_stage;
    b.dstStageMask = dst_stage;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, base_mip, level_count, 0u, image_info.arrayLayers };

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;

    vkCmdPipelineBarrier2(cb, &dep);
  };

  barrier(image_info.initialLayout,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          0,
          VK_ACCESS_2_TRANSFER_WRITE_BIT,
          0u,
          image_info.mipLevels);

  VkBufferImageCopy2 region{};
  region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
  region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, image_info.arrayLayers };
  region.imageOffset = { 0, 0, 0 };
  region.imageExtent = image_info.extent;

  VkCopyBufferToImageInfo2 copy{};
  copy.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
  copy.srcBuffer = get_buffer(ctx, staging_buffer);
  copy.dstImage = image;
  copy.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  copy.regionCount = 1;
  copy.pRegions = &region;

  vkCmdCopyBufferToImage2(cb, &copy);

  barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT,
          VK_ACCESS_2_TRANSFER_READ_BIT,
          0u,
          1u);

  for (std::uint32_t mip = 1; mip < image_info.mipLevels; ++mip) {
    auto src_e = mip_extent(image_info.extent, mip - 1);
    auto dst_e = mip_extent(image_info.extent, mip);

    for (std::uint32_t layer = 0; layer < image_info.arrayLayers; ++layer) {
      VkImageBlit2 blit{};
      blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
      blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, layer, 1 };
      blit.srcOffsets[0] = { 0, 0, 0 };
      blit.srcOffsets[1] = { static_cast<int32_t>(src_e.width),
                             static_cast<int32_t>(src_e.height),
                             static_cast<int32_t>(src_e.depth) };
      blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1 };
      blit.dstOffsets[0] = { 0, 0, 0 };
      blit.dstOffsets[1] = { static_cast<int32_t>(dst_e.width),
                             static_cast<int32_t>(dst_e.height),
                             static_cast<int32_t>(dst_e.depth) };

      VkBlitImageInfo2 bi{};
      bi.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
      bi.srcImage = image;
      bi.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      bi.dstImage = image;
      bi.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      bi.regionCount = 1;
      bi.pRegions = &blit;
      bi.filter = VK_FILTER_LINEAR;

      vkCmdBlitImage2(cb, &bi);
    }

    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            mip,
            1u);
  }

  const auto final_layout =
    description.final_layout.value_or(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          final_layout,
          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_TRANSFER_READ_BIT,
          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
          0u,
          image_info.mipLevels);

  auto submitted = ctx.submit(cmd, {});
  if (!submitted) throw std::runtime_error(submitted.error());
  ctx.wait_for(submitted.value());
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
                               TextureUsageFlags::Storage) },
is_depth{ static_cast<bool>(description.usage_flags &
                               TextureUsageFlags::DepthStencilAttachment) }
{
  if (!description.externally_created_image) {
    create_internal_image(ctx, description);
  } else {
    image = *description.externally_created_image;
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
  view_info.subresourceRange.aspectMask = is_depth
                                           ? VK_IMAGE_ASPECT_DEPTH_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
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

  mip_layer_views.resize(mip_levels * array_layers);

  // Less than two views means just one view for the entire image
  const auto has_only_one_view =
    mip_levels == 1 && array_layers == 1;
  if (has_only_one_view) {
    mip_layer_views.at(0) = image_view;
  } else {
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

VkSamplerMipmapMode filter_mode_to_vk_mip_map_mode(
  MipMapMode mip_map_mode)
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
  const auto name = std::format("Framebuffer_({})_view (mip: {}, layer: {})",
                          static_cast<const void*>(image),
                          mip,
                          layer);
  set_name_for_object(
    context.get_device(),
    VK_OBJECT_TYPE_IMAGE_VIEW,
    cached_framebuffer_views.at(mip * cube_array_layers + layer),
    name);

  return cached_framebuffer_views[mip * cube_array_layers + layer];
}

} // namespace VkBindless
