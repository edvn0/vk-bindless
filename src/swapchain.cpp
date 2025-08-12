#include "vk-bindless/swapchain.hpp"

#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/texture.hpp"
#include "vk-bindless/vulkan_context.hpp"

namespace VkBindless {

namespace {
auto choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR> &formats,
                                ColorSpace requestedColorSpace,
                                bool hasSwapchainColorspaceExt) {

  auto isNativeSwapChainBGR =
      [](const std::vector<VkSurfaceFormatKHR> &formats) -> bool {
    for (const VkSurfaceFormatKHR &fmt : formats) {
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

  auto colorSpaceToVkSurfaceFormat =
      [](ColorSpace colorSpace, bool isBGR,
         bool hasSwapchainColorspaceExt) -> VkSurfaceFormatKHR {
    switch (colorSpace) {
    case ColorSpace::SRGB_NONLINEAR:
      return VkSurfaceFormatKHR{isBGR ? VK_FORMAT_B8G8R8A8_UNORM
                                      : VK_FORMAT_R8G8B8A8_UNORM,
                                VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    case ColorSpace::SRGB_EXTENDED_LINEAR:
      if (hasSwapchainColorspaceExt)
        return VkSurfaceFormatKHR{VK_FORMAT_R16G16B16A16_SFLOAT,
                                  VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};
      [[fallthrough]];
    case ColorSpace::HDR10:
      if (hasSwapchainColorspaceExt) {
        return VkSurfaceFormatKHR{isBGR ? VK_FORMAT_A2B10G10R10_UNORM_PACK32
                                        : VK_FORMAT_A2R10G10B10_UNORM_PACK32,
                                  VK_COLOR_SPACE_HDR10_ST2084_EXT};
      }
      [[fallthrough]];
    default:
      // default to normal sRGB non linear.
      return VkSurfaceFormatKHR{isBGR ? VK_FORMAT_B8G8R8A8_SRGB
                                      : VK_FORMAT_R8G8B8A8_SRGB,
                                VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }
  };

  const VkSurfaceFormatKHR preferred = colorSpaceToVkSurfaceFormat(
      requestedColorSpace, isNativeSwapChainBGR(formats),
      hasSwapchainColorspaceExt);

  for (const VkSurfaceFormatKHR &fmt : formats) {
    if (fmt.format == preferred.format &&
        fmt.colorSpace == preferred.colorSpace) {
      return fmt;
    }
  }

  for (const VkSurfaceFormatKHR &fmt : formats) {
    if (fmt.format == preferred.format) {
      return fmt;
    }
  }

  return formats[0];
}
} // namespace

auto Swapchain::current_texture() -> TextureHandle {
  auto &vulkan_context = static_cast<Context &>(context_ref);

  if (need_next_image) {
    const VkSemaphoreWaitInfo waitInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &vulkan_context.timeline_semaphore,
        .pValues = &timeline_wait_values[swapchain_image_index],
    };
    vkWaitSemaphores(vulkan_context.get_device(), &waitInfo, UINT64_MAX);
    VkSemaphore acquireSemaphore = acquire_semaphores[swapchain_image_index];
    vkAcquireNextImageKHR(vulkan_context.get_device(), swapchain_khr,
                          UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE,
                          &swapchain_image_index);
    need_next_image = false;
    vulkan_context.immediate_commands.wait_semaphore(acquireSemaphore);
  }

  if (swapchain_image_index < swapchain_image_count()) {
    return swapchain_textures[swapchain_image_index];
  }

  return TextureHandle{};
}

Swapchain::Swapchain(IContext &ctx, uint32_t width, uint32_t height)
    : context_ref(static_cast<Context &>(ctx)),
      graphics_queue_handle(context_ref.get_queue(Queue::Graphics).value()),
      swapchain_width(width), swapchain_height(height) {
  swapchain_surface_format = choose_swap_surface_format(
      context_ref.device_surface_formats,
      context_ref.swapchain_requested_colour_space, true);

  VkBool32 queueFamilySupportsPresentation = VK_FALSE;
  VK_VERIFY(vkGetPhysicalDeviceSurfaceSupportKHR(
      ctx.get_physical_device(),
      context_ref.get_queue_family_index_unsafe(Queue::Graphics),
      context_ref.surface, &queueFamilySupportsPresentation));
  assert(queueFamilySupportsPresentation);

  auto chooseSwapImageCount =
      [](const VkSurfaceCapabilitiesKHR &caps) -> uint32_t {
    const uint32_t desired = caps.minImageCount + 1;
    const bool exceeded =
        caps.maxImageCount > 0 && desired > caps.maxImageCount;
    return exceeded ? caps.maxImageCount : desired;
  };

  auto chooseSwapPresentMode =
      [](const std::vector<VkPresentModeKHR> &modes) -> VkPresentModeKHR {
#if defined(__linux__) || defined(_M_ARM64)
    if (std::find(modes.cbegin(), modes.cend(),
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

  auto chooseUsageFlags = [](VkPhysicalDevice pd, VkSurfaceKHR surface,
                             VkFormat format) -> VkImageUsageFlags {
    VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkSurfaceCapabilitiesKHR caps = {};
    VK_VERIFY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));

    VkFormatProperties props = {};
    vkGetPhysicalDeviceFormatProperties(pd, format, &props);

    const bool isStorageSupported =
        (caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) > 0;
    const bool isTilingOptimalSupported =
        (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) > 0;

    if (isStorageSupported && isTilingOptimalSupported) {
      usageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    return usageFlags;
  };

  const VkImageUsageFlags usageFlags =
      chooseUsageFlags(ctx.get_physical_device(), context_ref.surface,
                       swapchain_surface_format.format);
  const bool is_composite_alpha_opaque_supported =
      (context_ref.device_surface_capabilities.supportedCompositeAlpha &
       VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0;
  std::array<std::uint32_t, 1> queueFamilyIndices = {
      context_ref.get_queue_family_index_unsafe(Queue::Graphics),
  };

  const VkSwapchainCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = context_ref.surface,
      .minImageCount =
          chooseSwapImageCount(context_ref.device_surface_capabilities),
      .imageFormat = swapchain_surface_format.format,
      .imageColorSpace = swapchain_surface_format.colorSpace,
      .imageExtent = {.width = width, .height = height},
      .imageArrayLayers = 1,
      .imageUsage = usageFlags,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = queueFamilyIndices.data(),
#if defined(ANDROID)
      .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
#else
      .preTransform = context_ref.device_surface_capabilities.currentTransform,
#endif
      .compositeAlpha = is_composite_alpha_opaque_supported
                            ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                            : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      .presentMode = chooseSwapPresentMode(context_ref.device_present_modes),
      .clipped = VK_TRUE,
      .oldSwapchain = VK_NULL_HANDLE,
  };
  VK_VERIFY(vkCreateSwapchainKHR(context_ref.get_device(), &ci, nullptr,
                                 &swapchain_khr));

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
  std::array<VkImage, max_swapchain_images> swapchain_images;
  VK_VERIFY(vkGetSwapchainImagesKHR(context_ref.get_device(), swapchain_khr,
                                    &image_count, nullptr));
  if (image_count > max_swapchain_images) {
    image_count = max_swapchain_images;
  }
  VK_VERIFY(vkGetSwapchainImagesKHR(context_ref.get_device(), swapchain_khr,
                                    &image_count, swapchain_images.data()));

  static constexpr auto create_semaphore =
      [](VkDevice device, std::string_view n) -> VkSemaphore {
    auto semaphore_info =
        VkSemaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore semaphore;
    VK_VERIFY(vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore));
    // set_debug_object_name(device, VK_OBJECT_TYPE_SEMAPHORE,
    //                        (uint64_t)semaphore, n);
    return semaphore;
  };

  // create images, image views and framebuffers
  for (std::uint32_t i = 0; i < swapchain_image_count(); i++) {
    acquire_semaphores[i] = create_semaphore(context_ref.get_device(),
                                             "Semaphore: swapchain-acquire");

    VkTexture image{
        context_ref,
        VkTextureDescription{
            .format = swapchain_surface_format.format,
            .extent = {swapchain_width, swapchain_height, 1},
            .usage_flags = TextureUsageFlags::ColourAttachment |
                           TextureUsageFlags::TransferSource |
                           TextureUsageFlags::TransferDestination,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            .is_owning = true,
            .is_swapchain = true,
            .debug_name = "Swapchain Image " + std::to_string(i),
        },
    };

    image.create_image_view(context_ref.get_device(),
                            {.viewType = VK_IMAGE_VIEW_TYPE_2D,
                             .format = swapchain_surface_format.format,
                             .subresourceRange = {
                                 .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0,
                                 .levelCount = VK_REMAINING_MIP_LEVELS,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1,
                             }});

    swapchain_textures[i] = context_ref.texture_pool.create(std::move(image));
  }
}

Swapchain::~Swapchain() {
  for (TextureHandle handle : swapchain_textures) {
    context().destroy(handle);
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

} // namespace VkBindless