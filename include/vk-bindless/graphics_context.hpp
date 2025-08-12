#pragma once

#include "vk-bindless/commands.hpp"
#include "vk-bindless/expected.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/handle.hpp"
#include "vk-bindless/texture.hpp"

#include <deque>
#include <functional>
#include <string>
#include <vulkan/vulkan.h>

namespace VkBindless {

enum struct Queue {
  Graphics,
  Compute,
  Transfer,
};

using PreFrameCallback = std::function<void(VkDevice, VkAllocationCallbacks *)>;

struct ContextError {
  std::string message;
};

using TexturePool = Pool<Texture, VkTexture>;
using SamplerPool = Pool<Sampler, VkSampler>;

struct IContext {
  virtual ~IContext() = default;
  [[nodiscard]] virtual auto get_device() const -> const VkDevice & = 0;
  [[nodiscard]] virtual auto get_physical_device() const
      -> const VkPhysicalDevice & = 0;
  [[nodiscard]] virtual auto get_instance() const -> const VkInstance & = 0;
  [[nodiscard]] virtual auto get_queue(Queue queue) const
      -> Expected<VkQueue, ContextError> = 0;
  [[nodiscard]] virtual auto get_queue_family_index(Queue queue) const
      -> Expected<std::uint32_t, ContextError> = 0;
  [[nodiscard]] virtual auto get_queue_unsafe(Queue queue) const
      -> const VkQueue & = 0;
  [[nodiscard]] virtual auto get_queue_family_index_unsafe(Queue queue) const
      -> std::uint32_t = 0;

  [[nodiscard]] virtual auto needs_update() -> bool & = 0;
  virtual auto update_resource_bindings() -> void {
    // Default implementation does nothing.
    // Derived classes can override this to provide specific behavior.
  }
  virtual auto pre_frame_task(PreFrameCallback &&) -> void = 0;
  virtual auto get_allocator_implementation() -> IAllocator & = 0;

#define DESTROY_HANDLE_X_MACRO(type)                                           \
  virtual auto destroy(type handle) -> void = 0;
  FOR_EACH_HANDLE_TYPE(DESTROY_HANDLE_X_MACRO)
#undef DESTROY_HANDLE_X_MACRO

  virtual auto get_texture_pool() -> TexturePool & = 0;
  virtual auto get_sampler_pool() -> SamplerPool & = 0;

  virtual auto acquire_command_buffer() -> ICommandBuffer & = 0;
  virtual auto submit(ICommandBuffer &, TextureHandle present)
      -> SubmitHandle = 0;
  virtual auto get_current_swapchain_texture() -> TextureHandle = 0;
};

} // namespace VkBindless