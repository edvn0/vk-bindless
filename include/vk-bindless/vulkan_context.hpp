#pragma once

#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/commands.hpp"
#include "vk-bindless/expected.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/object_pool.hpp"
#include "vk-bindless/swapchain.hpp"
#include "vk-bindless/texture.hpp"
#include "vk-bindless/types.hpp"

#include <functional>
#include <memory>

#include <VkBootstrap.h>
#include <vulkan/vulkan.h>

namespace VkBindless {

auto
format_to_vk_format(Format format) -> VkFormat;
auto
vk_format_to_format(VkFormat format) -> Format;

inline auto
set_name_for_object(auto device,
                    const auto type,
                    const auto handle,
                    const std::string_view name)
{
  VkDebugUtilsObjectNameInfoEXT name_info{
    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
    .pNext = nullptr,
    .objectType = type,
    .objectHandle = std::bit_cast<std::uint64_t>(handle),
    .pObjectName = name.data(),
  };
  static auto vkSetDebugUtilsObjectNameEXT =
    reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
      vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
  if (vkSetDebugUtilsObjectNameEXT) {
    vkSetDebugUtilsObjectNameEXT(device, &name_info);
  }
};

class StagingAllocator final
{public:
  explicit StagingAllocator(IContext& ctx);
  ~StagingAllocator() = default;

  StagingAllocator(const StagingAllocator&) = delete;
  StagingAllocator& operator=(const StagingAllocator&) = delete;

  void upload(VkDataBuffer& buffer, size_t dstOffset, size_t size, const void* data);
  void upload(VkTexture& image,
                   const VkRect2D& imageRegion,
                   std::uint32_t baseMipLevel,
                   std::uint32_t numMipLevels,
                   std::uint32_t layer,
                   std::uint32_t numLayers,
                   VkFormat format,
                   const void* data,
                   std::uint32_t bufferRowLength);
  /*void imageData3D(VkTexture& image, const VkOffset3D& offset, const VkExtent3D& extent, VkFormat format, const void* data);
  void getImageData(VkTexture& image,
                    const VkOffset3D& offset,
                    const VkExtent3D& extent,
                    VkImageSubresourceRange range,
                    VkFormat format,
                    void* outData);*/

private:
  static constexpr auto staging_buffer_alignment = 16;

  struct MemoryRegionDescription {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    SubmitHandle handle = {};
  };

  auto get_next_free_offset(std::uint32_t) -> MemoryRegionDescription;
  auto ensure_size(std::uint32_t) -> void;
  auto wait_and_reset() -> void;

  Context& context;
  Holder<BufferHandle> staging_buffer;
  VkDeviceSize staging_buffer_size = 0;
  uint32_t staging_buffer_count = 0;
  // the staging buffer grows from minBufferSize up to maxBufferSize as needed
  VkDeviceSize max_buffer_size = 0;
  VkDeviceSize min_buffer_size = 4ULL * 2048ULL * 2048ULL;
  std::vector<MemoryRegionDescription> regions{};
};

class Context final : public IContext
{
public:
  Context() = default;
  ~Context() override;

  static auto create(std::function<VkSurfaceKHR(VkInstance)>&& surface_fn)
    -> Expected<std::unique_ptr<IContext>, ContextError>;

  [[nodiscard]] auto get_device() const -> const VkDevice& override;
  [[nodiscard]] auto get_physical_device() const
    -> const VkPhysicalDevice& override;
  [[nodiscard]] auto get_instance() const -> const VkInstance& override;

  [[nodiscard]] auto get_queue(Queue) const
    -> Expected<VkQueue, ContextError> override;
  [[nodiscard]] auto get_queue_family_index(Queue) const
    -> Expected<std::uint32_t, ContextError> override;

  [[nodiscard]] auto get_queue_unsafe(Queue) const -> const VkQueue& override;
  [[nodiscard]] auto get_queue_family_index_unsafe(Queue) const
    -> std::uint32_t override;

  [[nodiscard]] auto needs_update() -> bool& override
  {
    return resource_bindings_updated;
  }

  [[nodiscard]] auto get_frame_index() const -> std::uint64_t override
  {
    return swapchain->current_frame_index();
  }
  auto get_swapchain() -> Swapchain& override { return *swapchain; }
  auto resize_swapchain(std::uint32_t, std::uint32_t) -> void override;

  auto update_resource_bindings() -> void override;
  auto pre_frame_task(PreFrameCallback&& callback) -> void override
  {
    pre_frame_callbacks.push_back(std::move(callback));
  }
  auto get_allocator_implementation() -> IAllocator& override;

#define DESTROY_HANDLE_X_MACRO(type) auto destroy(type handle) -> void override;
  FOR_EACH_HANDLE_TYPE(DESTROY_HANDLE_X_MACRO)
#undef DESTROY_HANDLE_X_MACRO

  auto get_texture_pool() -> TexturePool& override { return texture_pool; }
  auto get_sampler_pool() -> SamplerPool& override { return sampler_pool; }
  auto get_compute_pipeline_pool() -> ComputePipelinePool& override
  {
    return compute_pipeline_pool;
  }
  auto get_graphics_pipeline_pool() -> GraphicsPipelinePool& override
  {
    return graphics_pipeline_pool;
  }
  auto get_shader_module_pool() -> ShaderModulePool& override
  {
    return shader_module_pool;
  }
  auto get_buffer_pool() -> BufferPool& override { return buffer_pool; }

  auto acquire_command_buffer() -> ICommandBuffer& override;
  auto acquire_immediate_command_buffer() -> CommandBufferWrapper& override;
  auto submit(ICommandBuffer& cmd_buffer, TextureHandle present)
    -> Expected<SubmitHandle, std::string> override;
  auto get_current_swapchain_texture() -> TextureHandle override;
  auto get_dimensions(TextureHandle) const -> Dimensions override;
  auto get_device_address(BufferHandle) -> std::uint64_t override;
  auto get_mapped_pointer(BufferHandle) -> void* override;
  auto flush_mapped_memory(BufferHandle,
                           std::uint64_t offset,
                           std::uint64_t size) -> void override;
  [[nodiscard]] auto use_staging() const -> bool override { return use_staging_system; }
        auto wait_for(const SubmitHandle value) -> void override
  {
    immediate_commands->wait(value);
  }
  [[nodiscard]] auto get_immediate_commands() const -> auto&
  {
    return *immediate_commands;
  }

  auto bind_default_descriptor_sets(const VkCommandBuffer cmd,
                                    const VkPipelineBindPoint bind_point,
                                    const VkPipelineLayout layout) const -> void
  {
    const std::array dsets{
      descriptor_set, descriptor_set, descriptor_set, descriptor_set
    };
    vkCmdBindDescriptorSets(cmd,
                            bind_point,
                            layout,
                            0,
                            static_cast<std::uint32_t>(dsets.size()),
                            dsets.data(),
                            0,
                            nullptr);
  }
  auto get_pipeline(GraphicsPipelineHandle, std::uint32_t) -> VkPipeline;
  auto get_pipeline(ComputePipelineHandle) -> VkPipeline;

private:
  vkb::Instance vkb_instance{};
  vkb::PhysicalDevice vkb_physical_device{};
  vkb::Device vkb_device{};
  VkSurfaceKHR surface{};
  Unique<Swapchain> swapchain{};
  Unique<StagingAllocator> staging_allocator{  };
  VkSemaphore timeline_semaphore{ VK_NULL_HANDLE };
  bool use_staging_system{ true };

  VkQueue graphics_queue{};
  VkQueue compute_queue{};
  VkQueue transfer_queue{};

  std::uint32_t graphics_queue_family{};
  std::uint32_t compute_queue_family{};
  std::uint32_t transfer_queue_family{};

  TexturePool texture_pool{};
  SamplerPool sampler_pool{};
  ComputePipelinePool compute_pipeline_pool{};
  GraphicsPipelinePool graphics_pipeline_pool{};
  ShaderModulePool shader_module_pool{};
  BufferPool buffer_pool{};

  std::uint32_t current_max_textures{ 16 };
  std::uint32_t current_max_samplers{ 16 };
  std::uint32_t current_max_acceleration_structures{ 16 };
  bool resource_bindings_updated =
    true; // Flag to indicate if resource bindings need to be updated. True
          // initially to trigger first
  VkDescriptorSetLayout descriptor_set_layout{ VK_NULL_HANDLE };
  VkDescriptorSet descriptor_set{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
  Handle<Texture> dummy_texture;
  Handle<Sampler> dummy_sampler;

  std::unique_ptr<ImmediateCommands> immediate_commands{ nullptr };
  bool is_headless{ false };
  CommandBuffer command_buffer{};
  Unique<IAllocator> allocator_impl{ nullptr, default_deleter<IAllocator> };
  std::vector<VkSurfaceFormatKHR> device_surface_formats;
  std::vector<VkFormat> device_depth_formats;
  std::vector<VkPresentModeKHR> device_present_modes;
  ColorSpace swapchain_requested_colour_space{ ColorSpace::SRGB_NONLINEAR };
  VkSurfaceCapabilitiesKHR device_surface_capabilities{};
  struct VulkanProperties
  {
    VkPhysicalDeviceProperties base{};
    VkPhysicalDeviceVulkan11Properties eleven{};
    VkPhysicalDeviceVulkan12Properties twelve{};
    VkPhysicalDeviceVulkan13Properties thirteen{};
    VkPhysicalDeviceVulkan14Properties fourteen{};
  };
  VulkanProperties vulkan_properties{};
  bool has_swapchain_maintenance_1{ false };

  std::deque<PreFrameCallback> pre_frame_callbacks{};
  auto process_callbacks() -> void
  {
    while (!pre_frame_callbacks.empty()) {
      auto callback = std::move(pre_frame_callbacks.front());
      VkAllocationCallbacks* allocation_callbacks = nullptr;
      pre_frame_callbacks.pop_front();
      callback(vkb_device.device, allocation_callbacks);
    }
  }

  static auto get_dsl_binding(std::uint32_t, VkDescriptorType, uint32_t)
    -> VkDescriptorSetLayoutBinding;
  auto grow_descriptor_pool(std::uint32_t textures, std::uint32_t samplers)
    -> Expected<void, ContextError>;
  auto create_placeholder_resources() -> void;
  auto update_descriptor_sets() -> Expected<void, ContextError>;

  using base = IContext;

  friend class Swapchain;
  friend class CommandBuffer;
  friend class StagingAllocator;
  friend class VkTexture;
};

} // namespace VkBindless
