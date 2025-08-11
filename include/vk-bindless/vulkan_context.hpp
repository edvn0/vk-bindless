#pragma once

#include "vk-bindless/expected.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/object_pool.hpp"
#include "vk-bindless/texture.hpp"
#include "vk-bindless/types.hpp"

#include <functional>
#include <memory>

#include <VkBootstrap.h>

namespace VkBindless {

class Context final : public IContext {
public:
  Context() = default;
  ~Context() override;

  static auto create(std::function<VkSurfaceKHR(VkInstance)> &&surface_fn)
      -> Expected<std::unique_ptr<IContext>, ContextError>;

  [[nodiscard]] auto get_device() const -> const VkDevice & override;
  [[nodiscard]] auto get_physical_device() const
      -> const VkPhysicalDevice & override;
  [[nodiscard]] auto get_instance() const -> const VkInstance & override;

  [[nodiscard]] auto get_queue(Queue) const
      -> Expected<VkQueue, ContextError> override;
  [[nodiscard]] auto get_queue_family_index(Queue) const
      -> Expected<std::uint32_t, ContextError> override;

  [[nodiscard]] auto get_queue_unsafe(Queue) const -> const VkQueue & override;
  [[nodiscard]] auto get_queue_family_index_unsafe(Queue) const
      -> std::uint32_t override;

  [[nodiscard]] auto needs_update() -> bool & override {
    return resource_bindings_updated;
  }
  auto update_resource_bindings() -> void override;
  auto pre_frame_task(PreFrameCallback &&callback) -> void override {
    pre_frame_callbacks.push_back(std::move(callback));
  }
  auto get_allocator_implementation() -> IAllocator & override;

#define DESTROY_HANDLE_X_MACRO(type) auto destroy(type handle) -> void override;
  FOR_EACH_HANDLE_TYPE(DESTROY_HANDLE_X_MACRO)
#undef DESTROY_HANDLE_X_MACRO

  auto get_texture_pool() -> TexturePool & override { return texture_pool; }
  auto get_sampler_pool() -> SamplerPool & override { return sampler_pool; }

private:
  vkb::Instance vkb_instance{};
  vkb::PhysicalDevice vkb_physical_device{};
  vkb::Device vkb_device{};
  VkSurfaceKHR surface{};

  VkQueue graphics_queue{};
  VkQueue compute_queue{};
  VkQueue transfer_queue{};

  std::uint32_t graphics_queue_family{};
  std::uint32_t compute_queue_family{};
  std::uint32_t transfer_queue_family{};

  TexturePool texture_pool{};
  SamplerPool sampler_pool{};

  std::uint32_t current_max_textures{16};
  std::uint32_t current_max_samplers{16};
  std::uint32_t current_max_acceleration_structures{16};
  bool resource_bindings_updated = false;
  VkDescriptorSetLayout descriptor_set_layout{VK_NULL_HANDLE};
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
  Handle<Texture> dummy_texture;
  Handle<Sampler> dummy_sampler;

  Unique<IAllocator> allocator_impl{nullptr, default_deleter<IAllocator>};

  std::deque<PreFrameCallback> pre_frame_callbacks{};

  static auto get_dsl_binding(std::uint32_t, VkDescriptorType, uint32_t)
      -> VkDescriptorSetLayoutBinding;
  auto grow_descriptor_pool(std::uint32_t textures, std::uint32_t samplers)
      -> Expected<void, ContextError>;
  auto create_placeholder_resources() -> void;
  auto update_descriptor_sets() -> Expected<void, ContextError>;

  using base = IContext;
};

} // namespace VkBindless
