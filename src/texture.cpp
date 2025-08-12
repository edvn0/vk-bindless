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

template <typename T0, typename... Ts>
constexpr auto max(T0 &&first, Ts &&...rest) {
  auto max_value = std::forward<T0>(first);
  ((max_value = max_value < rest ? rest : max_value), ...);
  return max_value;
}

VkTexture::VkTexture(IContext &ctx, const VkTextureDescription &description)
    : sample_count{description.sample_count},
      image_owns_itself{description.is_owning},
      is_swapchain{description.is_swapchain},
      sampled{static_cast<bool>(description.usage_flags &
                                TextureUsageFlags::Sampled)},
      storage{static_cast<bool>(description.usage_flags &
                                TextureUsageFlags::Storage)} {
  auto &allocator = ctx.get_allocator_implementation();

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = nullptr;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = description.format;
  image_info.extent = description.extent;
  image_info.mipLevels =
      description.mip_levels.value_or(static_cast<std::uint32_t>(
          std::log2(max(description.extent.width, description.extent.height,
                        description.extent.depth)) +
          1));
  image_info.arrayLayers = description.layers;
  image_info.samples = description.sample_count;
  image_info.tiling = description.tiling;
  image_info.usage = static_cast<VkImageUsageFlags>(description.usage_flags);
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = description.initial_layout;
  image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

  assert(image_info.mipLevels > 0 && image_info.arrayLayers > 0 &&
         "Image must have at least one mip level and one layer");

  if (description.debug_name.empty()) {
    set_name_for_object(
        ctx.get_device(), VK_OBJECT_TYPE_IMAGE, image, std::format("{}-[{}x{}]", description.debug_name, description.extent.width, description.extent.height));
      }

  AllocationCreateInfo alloc_info{
      .usage = MemoryUsage::AutoPreferDevice,
      .map_memory = false,
      .preferred_memory_bits = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .required_memory_bits = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .debug_name = std::string{description.debug_name},
  };

  auto could_allocate = allocator.allocate_image(image_info, alloc_info);
  if (!could_allocate) {
    throw std::runtime_error("Failed to allocate image: " +
                             could_allocate.error().message);
  }

  auto &&[img, alloc] = std::move(could_allocate.value());
  image = img;
  image_allocation = alloc;

  // We don't want to create swapchain views here.
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
  view_info.format = description.format;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = image_info.mipLevels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = image_info.arrayLayers;
  view_info.components.r = VK_COMPONENT_SWIZZLE_R;
  view_info.components.g = VK_COMPONENT_SWIZZLE_G;
  view_info.components.b = VK_COMPONENT_SWIZZLE_B;
  view_info.components.a = VK_COMPONENT_SWIZZLE_A;

  VK_VERIFY(
      vkCreateImageView(ctx.get_device(), &view_info, nullptr, &image_view));

  mip_layer_views.resize(image_info.mipLevels * image_info.arrayLayers);

  if (mip_layer_views.size() < 2) {
    mip_layer_views.at(0) = image_view;
  } else {
    for (std::uint32_t mip = 0; mip < image_info.mipLevels; ++mip) {
      for (std::uint32_t layer = 0; layer < image_info.arrayLayers; ++layer) {
        const auto index = mip * image_info.arrayLayers + layer;
        view_info.subresourceRange.baseMipLevel = mip;
        view_info.subresourceRange.baseArrayLayer = layer;
        view_info.subresourceRange.layerCount = 1;
        view_info.subresourceRange.levelCount = 1;

        auto &mip_layer_view = mip_layer_views[index];
        VK_VERIFY(vkCreateImageView(ctx.get_device(), &view_info, nullptr,
                                    &mip_layer_view));
      }
    }
  }
}

auto VkTexture::create(IContext &context,
                       const VkTextureDescription &description)
    -> Holder<TextureHandle> {
  auto &pool = context.get_texture_pool();
  auto handle = pool.create(VkTexture{
      context,
      description,
  });

  if (!handle.valid()) {
    return Holder<TextureHandle>::invalid();
  }

  return Holder{
      &context,
      std::move(handle),
  };
}

auto VkTextureSampler::create(IContext &context,
                              const VkSamplerCreateInfo &info)
    -> Holder<SamplerHandle> {
  auto &pool = context.get_sampler_pool();
  VkSampler sampler{VK_NULL_HANDLE};
  VkSamplerCreateInfo sampler_info{info};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.pNext = nullptr;
  VK_VERIFY(
      vkCreateSampler(context.get_device(), &sampler_info, nullptr, &sampler));

  auto handle = pool.create(std::move(sampler));
  return Holder{
      &context,
      std::move(handle),
  };
}

auto VkTexture::create_image_view(VkDevice device,
                                  const VkImageViewCreateInfo &view_info)
    -> void {
  VkImageViewCreateInfo copy = view_info;
  copy.image = image;
  copy.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  VK_VERIFY(vkCreateImageView(device, &copy, nullptr, &image_view));
}

} // namespace VkBindless
