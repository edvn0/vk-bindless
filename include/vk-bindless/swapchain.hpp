#pragma once

#include "vk-bindless/expected.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/handle.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vulkan/vulkan.h>

namespace VkBindless {

enum class SwapchainPresentFailure
{
  OutOfDate,
  Suboptimal,
};

class Swapchain final
{
  static constexpr auto max_swapchain_images = 8U;

public:
  Swapchain(IContext& context, std::uint32_t width, std::uint32_t height);
  ~Swapchain();

  Swapchain(const Swapchain&) = delete;
  auto operator=(const Swapchain&) -> Swapchain& = delete;
  Swapchain(Swapchain&&) = delete;
  auto operator=(Swapchain&&) -> Swapchain& = delete;

  auto present(VkSemaphore wait_semaphore)
    -> Expected<void, SwapchainPresentFailure>;
  auto resize(std::uint32_t, std::uint32_t) -> void;

  auto current_vk_image() const -> VkImage;
  auto current_vk_image_view() const -> VkImageView;
  auto current_texture() -> TextureHandle;

  auto surface_format() const -> const VkSurfaceFormatKHR&;
  auto swapchain_current_image_index() const -> std::uint32_t;
  auto swapchain_image_count() const -> std::uint32_t;

  auto context() -> IContext&;
  auto graphics_queue() const -> VkQueue;
  auto width() const -> std::uint32_t { return swapchain_width; }
  auto height() const -> std::uint32_t { return swapchain_height; }
  auto current_frame_index() const -> std::uint64_t { return frame_index; }
  auto next_image_needed() const -> bool;
  auto swapchain_handle() const -> VkSwapchainKHR { return swapchain_khr; }

  auto set_size(std::uint32_t width, std::uint32_t height) -> void;
  auto set_next_image_needed(bool value) -> void;

private:
  Context& context_ref;
  VkQueue graphics_queue_handle = VK_NULL_HANDLE;
  std::uint32_t swapchain_width = 0;
  std::uint32_t swapchain_height = 0;
  std::uint32_t image_count = 0;
  std::uint32_t swapchain_image_index = 0;
  std::uint64_t frame_index = 0;
  bool need_next_image = true;
  VkSwapchainKHR swapchain_khr = VK_NULL_HANDLE;
  VkSurfaceFormatKHR swapchain_surface_format{
    .format = VK_FORMAT_UNDEFINED,
    .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
  };
  std::array<TextureHandle, max_swapchain_images> swapchain_textures{};
  std::array<VkSemaphore, max_swapchain_images> acquire_semaphores{};
  std::array<VkFence, max_swapchain_images> present_fences{};
  std::array<std::uint64_t, max_swapchain_images> timeline_wait_values{};

  auto create_swapchain_impl(std::uint32_t, std::uint32_t, VkSwapchainKHR)
    -> void;
  auto wait_for_pending_timeline_operations() -> void;

  friend class Context;
};

} // namespace VkBindless
