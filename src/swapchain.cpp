#include "vk-bindless/swapchain.hpp"

#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/texture.hpp"
#include "vk-bindless/vulkan_context.hpp"

#include <bit>
#include <vulkan/vulkan.h>

namespace VkBindless {

namespace {
auto
choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& formats,
                           const ColorSpace requested_colour_space,
                           bool has_swapchain_ext)
{

  auto is_native_swapchain_bgr =
    [](const std::vector<VkSurfaceFormatKHR>& formats) -> bool {
    for (const VkSurfaceFormatKHR& fmt : formats) {
      // The preferred format should be the one which is closer to the beginning
      // of the formats container. If BGR is encountered earlier, it should be
      // picked as the format of choice. If RGB happens to be earlier, take it.
      if (fmt.format == VK_FORMAT_R8G8B8A8_UNORM ||
          fmt.format == VK_FORMAT_R8G8B8A8_SRGB ||
          fmt.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32) {
        return false;
      }
      if (fmt.format == VK_FORMAT_B8G8R8A8_UNORM ||
          fmt.format == VK_FORMAT_B8G8R8A8_SRGB ||
          fmt.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
        return true;
      }
    }
    return false;
  };

  auto colour_space_to_vk_surface_format =
    [](ColorSpace colorSpace,
       bool is_bgr,
       bool has_swapchain_ext) -> VkSurfaceFormatKHR {
    switch (colorSpace) {
      case ColorSpace::SRGB_NONLINEAR:
        return VkSurfaceFormatKHR{ is_bgr ? VK_FORMAT_B8G8R8A8_UNORM
                                          : VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
      case ColorSpace::SRGB_EXTENDED_LINEAR:
        if (has_swapchain_ext)
          return VkSurfaceFormatKHR{ VK_FORMAT_R16G16B16A16_SFLOAT,
                                     VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT };
        [[fallthrough]];
      case ColorSpace::HDR10:
        if (has_swapchain_ext) {
          return VkSurfaceFormatKHR{ is_bgr
                                       ? VK_FORMAT_A2B10G10R10_UNORM_PACK32
                                       : VK_FORMAT_A2R10G10B10_UNORM_PACK32,
                                     VK_COLOR_SPACE_HDR10_ST2084_EXT };
        }
        [[fallthrough]];
      default:
        // default to normal sRGB non linear.
        return VkSurfaceFormatKHR{ is_bgr ? VK_FORMAT_B8G8R8A8_SRGB
                                          : VK_FORMAT_R8G8B8A8_SRGB,
                                   VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    }
  };

  const VkSurfaceFormatKHR preferred =
    colour_space_to_vk_surface_format(requested_colour_space,
                                      is_native_swapchain_bgr(formats),
                                      has_swapchain_ext);

  for (const VkSurfaceFormatKHR& fmt : formats) {
    if (fmt.format == preferred.format &&
        fmt.colorSpace == preferred.colorSpace) {
      return fmt;
    }
  }

  for (const VkSurfaceFormatKHR& fmt : formats) {
    if (fmt.format == preferred.format) {
      return fmt;
    }
  }

  return formats[0];
}
} // namespace

auto
Swapchain::swapchain_image_count() const -> std::uint32_t
{
  return image_count;
}

auto
Swapchain::context() -> IContext&
{
  return context_ref;
}

auto
Swapchain::current_texture() -> TextureHandle
{
  auto& vulkan_context = static_cast<Context&>(context_ref);

  if (need_next_image) {
    if (present_fences[swapchain_current_image_index()]) {
      // VK_EXT_swapchain_maintenance1: before acquiring again, wait for the
      // presentation operation to finish
      VK_VERIFY(
        vkWaitForFences(context_ref.vkb_device,
                        1,
                        &present_fences[swapchain_current_image_index()],
                        VK_TRUE,
                        UINT64_MAX));
      VK_VERIFY(
        vkResetFences(context_ref.vkb_device,
                      1,
                      &present_fences[swapchain_current_image_index()]));
    }
    const VkSemaphoreWaitInfo wait_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .pNext = nullptr,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &vulkan_context.timeline_semaphore,
      .pValues = &timeline_wait_values[swapchain_image_index],
    };
    vkWaitSemaphores(vulkan_context.get_device(), &wait_info, UINT64_MAX);
    const auto& acquire_semaphore = acquire_semaphores[swapchain_image_index];
    auto result = vkAcquireNextImageKHR(vulkan_context.get_device(),
                                        swapchain_khr,
                                        UINT64_MAX,
                                        acquire_semaphore,
                                        VK_NULL_HANDLE,
                                        &swapchain_image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      return TextureHandle{};
    }
    need_next_image = false;
    vulkan_context.immediate_commands->wait_semaphore(acquire_semaphore);
  }

  if (swapchain_image_index < swapchain_image_count()) {
    return swapchain_textures[swapchain_image_index];
  }

  return TextureHandle{};
}

auto
Swapchain::current_vk_image() const -> VkImage
{
  if (swapchain_current_image_index() < image_count) {
    auto* tex = context_ref.texture_pool
                  .get(swapchain_textures.at(swapchain_current_image_index()))
                  .value();
    return tex->get_image();
  }
  return VK_NULL_HANDLE;
}

auto
Swapchain::current_vk_image_view() const -> VkImageView
{
  if (swapchain_current_image_index() < image_count) {
    auto* tex = context_ref.texture_pool
                  .get(swapchain_textures.at(swapchain_current_image_index()))
                  .value();
    return tex->get_image_view();
  }
  return VK_NULL_HANDLE;
}

auto
Swapchain::create_swapchain_impl(std::uint32_t width,
                                 std::uint32_t height,
                                 VkSwapchainKHR old_swapchain) -> void
{
  VK_VERIFY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    context_ref.get_physical_device(),
    context_ref.surface,
    &context_ref.device_surface_capabilities));

  const auto& caps = context_ref.device_surface_capabilities;
  swapchain_width =
    std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
  swapchain_height =
    std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);

  swapchain_surface_format =
    choose_swap_surface_format(context_ref.device_surface_formats,
                               context_ref.swapchain_requested_colour_space,
                               true);

  VkBool32 queue_family_supports_presentation = VK_FALSE;
  VK_VERIFY(vkGetPhysicalDeviceSurfaceSupportKHR(
    context_ref.get_physical_device(),
    context_ref.get_queue_family_index_unsafe(Queue::Graphics),
    context_ref.surface,
    &queue_family_supports_presentation));
  assert(queue_family_supports_presentation);

  auto choose_swapchain_image_count =
    [](const VkSurfaceCapabilitiesKHR& caps) -> uint32_t {
    const uint32_t desired = caps.minImageCount + 1;
    const bool exceeded =
      caps.maxImageCount > 0 && desired > caps.maxImageCount;
    return exceeded ? caps.maxImageCount : desired;
  };

  auto choose_swapchain_present_mode =
    [](const std::vector<VkPresentModeKHR>& modes) -> VkPresentModeKHR {
#if defined(__linux__) || defined(_M_ARM64)
    if (std::find(modes.cbegin(),
                  modes.cend(),
                  VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.cend()) {
      return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
#endif // __linux__
    if (std::find(modes.cbegin(), modes.cend(), VK_PRESENT_MODE_MAILBOX_KHR) !=
        modes.cend()) {
      return VK_PRESENT_MODE_MAILBOX_KHR;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
  };

  auto choose_usage_flags = [](VkPhysicalDevice pd,
                               VkSurfaceKHR surface,
                               VkFormat format) -> VkImageUsageFlags {
    VkImageUsageFlags usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkSurfaceCapabilitiesKHR caps = {};
    VK_VERIFY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));

    VkFormatProperties props = {};
    vkGetPhysicalDeviceFormatProperties(pd, format, &props);

    const bool is_storage_supported =
      (caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) > 0;
    const bool is_tiling_optimal_supported =
      (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) > 0;

    if (is_storage_supported && is_tiling_optimal_supported) {
      usage_flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    return usage_flags;
  };

  const VkImageUsageFlags usage_flags =
    choose_usage_flags(context_ref.get_physical_device(),
                       context_ref.surface,
                       swapchain_surface_format.format);
  const bool is_composite_alpha_opaque_supported =
    (context_ref.device_surface_capabilities.supportedCompositeAlpha &
     VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0;
  std::array<std::uint32_t, 1> queue_family_indices = {
    context_ref.get_queue_family_index_unsafe(Queue::Graphics),
  };

  const VkSwapchainCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = nullptr,
      .flags = 0,
      .surface = context_ref.surface,
      .minImageCount =
          choose_swapchain_image_count(context_ref.device_surface_capabilities),
      .imageFormat = swapchain_surface_format.format,
      .imageColorSpace = swapchain_surface_format.colorSpace,
      .imageExtent =
          {
              .width = swapchain_width,
              .height = swapchain_height,
          },
      .imageArrayLayers = 1,
      .imageUsage = usage_flags,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = queue_family_indices.data(),
      .preTransform = context_ref.device_surface_capabilities.currentTransform,
      .compositeAlpha = is_composite_alpha_opaque_supported
                            ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                            : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      .presentMode =
          choose_swapchain_present_mode(context_ref.device_present_modes),
      .clipped = VK_TRUE,
      .oldSwapchain = old_swapchain,
  };
  VK_VERIFY(vkCreateSwapchainKHR(
    context_ref.get_device(), &ci, nullptr, &swapchain_khr));

  /*
if (context_ref.has_EXT_hdr_metadata_) {
const VkHdrMetadataEXT metadata = {
.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
.displayPrimaryRed = {.x = 0.680f, .y = 0.320f},
.displayPrimaryGreen = {.x = 0.265f, .y = 0.690f},
.displayPrimaryBlue = {.x = 0.150f, .y = 0.060f},
.whitePoint = {.x = 0.3127f, .y = 0.3290f},
.maxLuminance = 80.0f,
.minLuminance = 0.001f,
.maxContentLightLevel = 2000.0f,
.maxFrameAverageLightLevel = 500.0f,
};
vkSetHdrMetadataEXT(device_, 1, &swapchain_, &metadata);
}
*/

  if (old_swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(context_ref.get_device(), old_swapchain, nullptr);
  }

  std::array<VkImage, max_swapchain_images> swapchain_images{};
  VK_VERIFY(vkGetSwapchainImagesKHR(
    context_ref.get_device(), swapchain_khr, &image_count, nullptr));
  if (image_count > max_swapchain_images) {
    image_count = max_swapchain_images;
  }
  VK_VERIFY(vkGetSwapchainImagesKHR(context_ref.get_device(),
                                    swapchain_khr,
                                    &image_count,
                                    swapchain_images.data()));

  static constexpr auto create_semaphore = [](VkDevice device,
                                              std::string_view) {
    const VkSemaphoreCreateInfo semaphore_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
    };
    VkSemaphore semaphore;
    VK_VERIFY(vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore));
    set_name_for_object(device,
                        VK_OBJECT_TYPE_SEMAPHORE,
                        (uint64_t)semaphore,
                        std::format("Semaphore: swapchain-acquire"));
    return semaphore;
  };

  // create images, image views and framebuffers
  for (std::uint32_t i = 0; i < swapchain_image_count(); i++) {
    acquire_semaphores[i] = create_semaphore(context_ref.get_device(),
                                             "Semaphore: swapchain-acquire");

    VkTexture image{
      context_ref,
      VkTextureDescription{
        .format = vk_format_to_format(swapchain_surface_format.format),
        .extent = { swapchain_width, swapchain_height, 1 },
        .usage_flags = TextureUsageFlags::ColourAttachment |
                       TextureUsageFlags::TransferSource |
                       TextureUsageFlags::TransferDestination,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
        .is_owning = true,
        .is_swapchain = true,
        .externally_created_image = swapchain_images[i],
        .debug_name = std::format("Swapchain_Image_{}", i),
      },
    };

    image.create_image_view(
        context_ref.get_device(),
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = VK_NULL_HANDLE, 
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = swapchain_surface_format.format,
            .components =
                {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        });

    swapchain_textures[i] = context_ref.texture_pool.create(std::move(image));
  }
}

Swapchain::Swapchain(IContext& ctx, uint32_t width, uint32_t height)
  : context_ref(static_cast<Context&>(ctx))
  , graphics_queue_handle(context_ref.get_queue(Queue::Graphics).value())
  , swapchain_width(width)
  , swapchain_height(height)
{
  create_swapchain_impl(width, height, VK_NULL_HANDLE);
}

Swapchain::~Swapchain()
{
  for (TextureHandle handle : swapchain_textures) {
    if (handle.valid()) {
      context().destroy(handle);
    }
  }
  vkDestroySwapchainKHR(context_ref.get_device(), swapchain_khr, nullptr);
  for (VkSemaphore sem : acquire_semaphores) {
    vkDestroySemaphore(context_ref.get_device(), sem, nullptr);
  }
  for (VkFence fence : present_fences) {
    if (fence)
      vkDestroyFence(context_ref.get_device(), fence, nullptr);
  }
}

auto
Swapchain::swapchain_current_image_index() const -> std::uint32_t
{
  return swapchain_image_index;
}

auto
Swapchain::set_next_image_needed(bool val) -> void
{
  need_next_image = val;
}

auto
Swapchain::resize(std::uint32_t new_width, std::uint32_t new_height) -> void
{
  if (new_width == swapchain_width && new_height == swapchain_height) {
    return;
  }

  vkDeviceWaitIdle(context_ref.get_device());

  for (TextureHandle handle : swapchain_textures) {
    if (handle.valid()) {
      context().destroy(handle);
    }
  }
  swapchain_textures.fill(TextureHandle{});

  for (VkSemaphore sem : acquire_semaphores) {
    if (sem != VK_NULL_HANDLE) {
      context().pre_frame_task(
        [sem](auto dev, auto) { vkDestroySemaphore(dev, sem, nullptr); });
      sem = VK_NULL_HANDLE;
    }
  }

  for (VkFence fence : present_fences) {
    if (fence != VK_NULL_HANDLE) {
      context().pre_frame_task(
        [fence](auto dev, auto) { vkDestroyFence(dev, fence, nullptr); });
      fence = VK_NULL_HANDLE;
    }
  }

  VkSwapchainKHR old_swapchain = swapchain_khr;
  swapchain_khr = VK_NULL_HANDLE;

  swapchain_image_index = 0;
  need_next_image = true;
  frame_index = 0;

  create_swapchain_impl(new_width, new_height, old_swapchain);
}

auto
Swapchain::present(VkSemaphore wait_semaphore)
  -> Expected<void, SwapchainPresentFailure>
{

  const VkSwapchainPresentFenceInfoEXT fence_info = {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
    .pNext = nullptr,
    .swapchainCount = 1,
    .pFences = &present_fences[swapchain_image_index],
  };
  const VkPresentInfoKHR pi = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .pNext = context_ref.has_swapchain_maintenance_1 ? &fence_info : nullptr,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &wait_semaphore,
    .swapchainCount = 1u,
    .pSwapchains = &swapchain_khr,
    .pImageIndices = &swapchain_image_index,
    .pResults = nullptr,
  };

  static auto create_fence = [](VkDevice device, const std::string_view name) {
    VkFenceCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
    };
    info.flags = 0; // start unsignaled
    VkFence fence{};
    if (vkCreateFence(device, &info, nullptr, &fence) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create fence");
    }
    // If you have debug utils naming:
    if (!name.empty()) {
      set_name_for_object(device, VK_OBJECT_TYPE_FENCE, fence, name);
    }
    return fence;
  };

  if (context_ref.has_swapchain_maintenance_1 &&
      !present_fences[swapchain_image_index]) {
    present_fences[swapchain_image_index] =
      create_fence(context_ref.get_device(), "Fence: present-fence");
  }
  VkResult r = vkQueuePresentKHR(graphics_queue_handle, &pi);
  if (r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_OUT_OF_DATE_KHR) {
    return unexpected<SwapchainPresentFailure>(
      r == VK_ERROR_OUT_OF_DATE_KHR ? SwapchainPresentFailure::OutOfDate
                                    : SwapchainPresentFailure::Suboptimal);
  }

  if (r != VK_SUCCESS) {
    assert(false && "Failed to present swapchain image");
  }
  set_next_image_needed(true);
  frame_index++;

  return {};
}

} // namespace VkBindless