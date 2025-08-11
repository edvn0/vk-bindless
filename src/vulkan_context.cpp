#include "vk-bindless/vulkan_context.hpp"

#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/scope_exit.hpp"
#include "vk-bindless/texture.hpp"

#include <vk_mem_alloc.h>

#include <VkBootstrap.h>
#include <iostream>
#include <memory>
#include <queue>
#include <vulkan/vulkan_core.h>

#define TODO(message)                                                          \
  do {                                                                         \
    throw std::runtime_error("TODO: " message);                                \
  } while (false)

namespace VkBindless {

constexpr VkShaderStageFlags all_stages_flags =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

Context::~Context() {
  vkDeviceWaitIdle(vkb_device.device);

  destroy(dummy_texture);
  destroy(dummy_sampler);

  while (!pre_frame_callbacks.empty()) {
    auto callback = std::move(pre_frame_callbacks.front());
    pre_frame_callbacks.pop_front();
    callback(vkb_device.device, nullptr);
  }

  // Destroy all resources
  texture_pool.clear();
  sampler_pool.clear();
}

auto Context::create(std::function<VkSurfaceKHR(VkInstance)> &&surface_fn)
    -> Expected<std::unique_ptr<IContext>, ContextError> {
  vkb::InstanceBuilder builder;
  auto inst_ret = builder.set_app_name("Bindless Vulkan")
                      .request_validation_layers()
                      .use_default_debug_messenger()
                      .require_api_version(1, 4, 0)
                      .build();
  if (!inst_ret) {
    return unexpected(ContextError{"Failed to create Vulkan instance"});
  }

  vkb::Instance vkb_instance = inst_ret.value();

  VkSurfaceKHR surf = surface_fn(vkb_instance.instance);

  if (VK_NULL_HANDLE == surf) {
    std::cerr << "Headless rendering is enabled." << std::endl;
  }

  vkb::PhysicalDeviceSelector selector{vkb_instance, surf};
  auto phys_ret =
      selector
          .set_minimum_version(1, 3) // You can adjust from 1.0 to 1.4 here
          /*  .add_required_extension("VK_KHR_shader_draw_parameters")
            .add_required_extension("VK_EXT_descriptor_indexing")
            .add_required_extension("VK_KHR_dynamic_rendering")
            .add_required_extension("VK_KHR_depth_stencil_resolve")
            .add_required_extension("VK_KHR_create_renderpass2")
            .add_required_extension(VK_KHR_DEVICE_GROUP_EXTENSION_NAME)
            .add_required_extension(VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME)
            .add_required_extension("VK_KHR_multiview")
            .add_required_extension("VK_KHR_maintenance2")
            .add_required_extension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME)
            .add_required_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
            .add_required_extension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME)
            .add_required_extension("VK_KHR_maintenance3")
            */
          .select();
  if (!phys_ret) {
    std::cout << "Failed to select Vulkan physical device: "
              << phys_ret.error().message() << std::endl;
    return unexpected(ContextError{"Failed to select Vulkan physical device"});
  }

  const vkb::PhysicalDevice &vkb_physical = phys_ret.value();

  VkPhysicalDeviceVulkan11Features vk11_features{};
  vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  vk11_features.shaderDrawParameters = VK_TRUE;

  VkPhysicalDeviceVulkan12Features vk12_features{};
  vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  vk12_features.pNext = &vk11_features;
  vk12_features.descriptorIndexing = VK_TRUE;
  vk12_features.runtimeDescriptorArray = VK_TRUE;
  vk12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  vk12_features.descriptorBindingPartiallyBound = VK_TRUE;
  vk12_features.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
  vk12_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
  vk12_features.bufferDeviceAddress = VK_TRUE;
  vk12_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
  vk12_features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
  vk12_features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
  vk12_features.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
  vk12_features.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
  vk12_features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;

  VkPhysicalDeviceVulkan13Features vk13_features{};
  vk13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  vk13_features.pNext = &vk12_features;
  vk13_features.dynamicRendering = VK_TRUE;

  VkPhysicalDeviceVulkan14Features vk14_features{};
  vk14_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
  vk14_features.pNext = &vk13_features;

  vkb::DeviceBuilder dev_builder{vkb_physical};
  dev_builder.add_pNext(&vk14_features);

  auto dev_ret = dev_builder.build();
  if (!dev_ret) {
    return unexpected(ContextError{"Failed to create logical device"});
  }

  const vkb::Device &vkb_device = dev_ret.value();

  auto context = std::make_unique<Context>();
  context->vkb_instance = vkb_instance;
  context->vkb_physical_device = vkb_physical;
  context->vkb_device = vkb_device;
  context->surface = surf;

  {
    auto q = vkb_device.get_queue(vkb::QueueType::graphics);
    if (!q)
      return unexpected(ContextError{"Missing graphics queue"});
    context->graphics_queue = q.value();
    context->graphics_queue_family =
        vkb_device.get_queue_index(vkb::QueueType::graphics).value();
  }

  {
    if (auto q = vkb_device.get_queue(vkb::QueueType::compute); !q) {
      context->compute_queue = context->graphics_queue;
      context->compute_queue_family = context->graphics_queue_family;
    } else {

      context->compute_queue = q.value();
      context->compute_queue_family =
          vkb_device.get_queue_index(vkb::QueueType::compute).value();
    }
  }

  {
    if (auto q = vkb_device.get_queue(vkb::QueueType::transfer); !q) {
      context->transfer_queue = context->graphics_queue;
      context->transfer_queue_family = context->graphics_queue_family;
    } else {
      context->transfer_queue = q.value();
      context->transfer_queue_family =
          vkb_device.get_queue_index(vkb::QueueType::transfer).value();
    }
  }

  auto allocator = IAllocator::create_allocator(
      vkb_instance.instance, vkb_physical.physical_device, vkb_device.device);

  context->allocator_impl = std::move(allocator);

  context->create_placeholder_resources();
  context->update_resource_bindings();

  return context;
}

auto Context::get_queue(const Queue queue) const
    -> Expected<VkQueue, ContextError> {
  switch (queue) {
    using enum VkBindless::Queue;
  case Graphics:
    return graphics_queue;
  case Compute:
    return compute_queue;
  case Transfer:
    return transfer_queue;
  }
  return unexpected(ContextError{"Invalid queue requested"});
}

auto Context::get_queue_family_index(const Queue queue) const
    -> Expected<std::uint32_t, ContextError> {
  switch (queue) {
    using enum VkBindless::Queue;
  case Graphics:
    return graphics_queue_family;
  case Compute:
    return compute_queue_family;
  case Transfer:
    return transfer_queue_family;
  }
  return unexpected(ContextError{"Invalid queue family requested"});
}

auto Context::get_device() const -> const VkDevice & {
  return vkb_device.device;
}

auto Context::get_physical_device() const -> const VkPhysicalDevice & {
  return vkb_physical_device.physical_device;
}

auto Context::get_instance() const -> const VkInstance & {
  return vkb_instance.instance;
}

auto Context::get_queue_unsafe(Queue queue) const -> const VkQueue & {
  switch (queue) {
    using enum VkBindless::Queue;
  case Graphics:
    return graphics_queue;
  case Compute:
    return compute_queue;
  case Transfer:
    return transfer_queue;
  }
  return graphics_queue;
}

auto Context::get_queue_family_index_unsafe(Queue queue) const
    -> std::uint32_t {
  switch (queue) {
    using enum Queue;
  case Graphics:
    return graphics_queue_family;
  case Compute:
    return compute_queue_family;
  case Transfer:
    return transfer_queue_family;
  }
  return graphics_queue_family;
}

void Context::update_resource_bindings() {
  base::update_resource_bindings();

  if (const auto &should_update = needs_update(); should_update) [[likely]] {
    return;
  }

  constexpr auto grow_factor = 1.5F;
  auto current_textures = 1U;
  auto current_samplers = 1U;

  constexpr auto grow_pool = [](const auto &pool, auto &out_max_textures) {
    while (pool.size() > out_max_textures) {
      out_max_textures = static_cast<std::uint32_t>(
          static_cast<float>(out_max_textures) * grow_factor);
    }
  };

  grow_pool(texture_pool, current_textures);
  grow_pool(sampler_pool, current_samplers);

  if (current_textures != current_max_textures ||
      current_samplers != current_max_samplers) {
    if (auto err = grow_descriptor_pool(current_textures, current_samplers);
        !err.has_value()) {
      std::cerr << "Failed to grow descriptor pool: " << err.error().message
                << std::endl;
      std::terminate();
    }
  }

  std::vector<VkDescriptorImageInfo> sampled_images;
  std::vector<VkDescriptorImageInfo> storage_images;

  sampled_images.reserve(texture_pool.size());
  storage_images.reserve(texture_pool.size());

  // Need a white texture for VkImageView dummy-ing
  const auto *realised_image = texture_pool.get(dummy_texture).value();
  const auto dummy_sampler = sampler_pool.at(0);
  const auto &dummy_image_view = realised_image->get_image_view();

  for (const auto &object : texture_pool) {
    const auto &view = object.get_image_view();
    const auto &storage_view = object.get_storage_image_view()
                                   ? object.get_storage_image_view()
                                   : object.get_image_view();

    const auto is_available =
        VK_SAMPLE_COUNT_1_BIT ==
        (object.get_sample_count() & VK_SAMPLE_COUNT_1_BIT);
    const auto is_sampled = object.is_sampled() && is_available;
    const auto is_storage = object.is_storage() && is_available;

    sampled_images.emplace_back(VK_NULL_HANDLE,
                                is_sampled ? view : dummy_image_view,
                                VK_IMAGE_LAYOUT_GENERAL);

    storage_images.emplace_back(VK_NULL_HANDLE,
                                is_storage ? storage_view : dummy_image_view,
                                VK_IMAGE_LAYOUT_GENERAL);
  }

  std::vector<VkDescriptorImageInfo> sampler_infos;
  sampler_infos.reserve(sampler_pool.size());

  for (const auto &object : sampler_pool) {
    sampler_infos.emplace_back(object ? object : dummy_sampler, VK_NULL_HANDLE,
                               VK_IMAGE_LAYOUT_UNDEFINED);
  }

  std::array<VkWriteDescriptorSet, 3> writes{};
  auto write_count = 0U;
  if (!sampled_images.empty()) {
    writes[0] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<std::uint32_t>(sampled_images.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = sampled_images.data(),
    };
    write_count++;
  }
  if (!sampler_infos.empty()) {
    writes[1] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<std::uint32_t>(sampler_infos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = sampler_infos.data(),
    };
    write_count++;
  }
  if (!storage_images.empty()) {
    writes[2] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<std::uint32_t>(storage_images.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = storage_images.data(),
    };
    write_count++;
  }

  if (write_count != 0) {
    vkQueueWaitIdle(get_queue_unsafe(Queue::Graphics));
    vkQueueWaitIdle(get_queue_unsafe(Queue::Compute));
    vkUpdateDescriptorSets(vkb_device.device, write_count, writes.data(), 0,
                           nullptr);
  }

  needs_update() = false;
}

auto Context::get_dsl_binding(const std::uint32_t index,
                              const VkDescriptorType descriptor_type,
                              const uint32_t max_count)
    -> VkDescriptorSetLayoutBinding {
  const auto binding = VkDescriptorSetLayoutBinding{
      .binding = index,
      .descriptorType = descriptor_type,
      .descriptorCount = max_count,
      .stageFlags = all_stages_flags,
      .pImmutableSamplers = nullptr,
  };

  return binding;
}

auto Context::grow_descriptor_pool(std::uint32_t new_max_textures,
                                   std::uint32_t new_max_samplers)
    -> Expected<void, ContextError> {
  static auto vkGetPhysicalDeviceProperties2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
          vkGetInstanceProcAddr(vkb_instance.instance,
                                "vkGetPhysicalDeviceProperties2"));

  current_max_textures = new_max_textures;
  current_max_samplers = new_max_samplers;

  if (vkGetPhysicalDeviceProperties2 == nullptr) {
    return unexpected(
        ContextError{"Failed to get vkGetPhysicalDeviceProperties2"});
  }

  VkPhysicalDeviceVulkan12Properties props12{};
  props12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;

  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &props12;

  vkGetPhysicalDeviceProperties2(vkb_physical_device, &props2);

#define VERIFY(condition, message)                                             \
  if (!(condition)) {                                                          \
    return unexpected(ContextError{message});                                  \
  }

  VERIFY(current_max_samplers <=
             props12.maxDescriptorSetUpdateAfterBindSamplers,
         "Maximum number of samplers exceeds device limit");
  VERIFY(current_max_textures <=
             props12.maxDescriptorSetUpdateAfterBindSampledImages,
         "Maximum number of sampled images exceeds device limit");

  if (const auto could = update_descriptor_sets(); !could.has_value()) {
    return unexpected(could.error());
  }

  return {};
}

auto Context::update_descriptor_sets() -> Expected<void, ContextError> {
  if (descriptor_pool != VK_NULL_HANDLE) {
    pre_frame_task(
        [pool = descriptor_pool](auto dev, auto *allocation_callbacks) {
          vkDestroyDescriptorPool(dev, pool, allocation_callbacks);
        });
  }
  if (descriptor_set_layout != VK_NULL_HANDLE) {
    pre_frame_task(
        [layout = descriptor_set_layout](auto dev, auto *allocation_callbacks) {
          vkDestroyDescriptorSetLayout(dev, layout, allocation_callbacks);
        });
  }

  const auto bindings = std::array{
      get_dsl_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                      current_max_textures),
      get_dsl_binding(1, VK_DESCRIPTOR_TYPE_SAMPLER, current_max_samplers),
      get_dsl_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                      current_max_textures),
      get_dsl_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      current_max_textures),
  };

  constexpr VkDescriptorSetLayoutCreateFlags flags =
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
      VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

  constexpr std::array<VkDescriptorSetLayoutCreateFlags, 4> binding_flags =
      std::array{flags, flags, flags, flags};

  const VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_create_info =
      {
          .sType =
              VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
          .bindingCount = static_cast<std::uint32_t>(binding_flags.size()),
          .pBindingFlags = binding_flags.data(),
      };

  const VkDescriptorSetLayoutCreateInfo dsl_create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &binding_flags_create_info,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      .bindingCount = static_cast<std::uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

  if (vkCreateDescriptorSetLayout(vkb_device.device, &dsl_create_info, nullptr,
                                  &descriptor_set_layout) != VK_SUCCESS) {
    return unexpected(ContextError{"Failed to create descriptor set layout"});
  };

  const std::array pool_sizes = {
      VkDescriptorPoolSize{
          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          current_max_textures,
      },
      VkDescriptorPoolSize{
          VK_DESCRIPTOR_TYPE_SAMPLER,
          current_max_samplers,
      },
      VkDescriptorPoolSize{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          current_max_textures,
      },
      VkDescriptorPoolSize{
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          current_max_textures,
      },
  };

  const VkDescriptorPoolCreateInfo pool_create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 1,
      .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
  };
  if (vkCreateDescriptorPool(vkb_device.device, &pool_create_info, nullptr,
                             &descriptor_pool) != VK_SUCCESS) {
    return unexpected(ContextError{"Failed to create descriptor pool"});
  }

  // Allocate the descriptor set
  VkDescriptorSetAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &descriptor_set_layout};
  if (vkAllocateDescriptorSets(vkb_device.device, &alloc_info,
                               &descriptor_set) != VK_SUCCESS) {
    return unexpected(ContextError{"Failed to allocate descriptor set"});
  }

  return {};
}

auto Context::create_placeholder_resources() -> void {
  const std::array<const std::uint8_t, 4> dummy_white_texture = {
      255,
      255,
      255,
      255,
  };
  auto image =
      VkTexture::create(*this, VkTextureDescription{
                                   .data = std::span(dummy_white_texture),
                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .extent = {1, 1, 1},
                                   .usage_flags = TextureUsageFlags::Sampled |
                                                  TextureUsageFlags::Storage,
                                   .debug_name = "Dummy White Texture (1x1)",
                               });
  assert(image.valid());
  dummy_texture = image.release();

  dummy_sampler = VkTextureSampler::create(
                      *this,
                      VkSamplerCreateInfo{
                          .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                          .pNext = nullptr,
                          .flags = {},
                          .magFilter = VK_FILTER_LINEAR,
                          .minFilter = VK_FILTER_LINEAR,
                          .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                          .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                          .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                          .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                          .mipLodBias = 0.0F,
                          .anisotropyEnable = VK_FALSE,
                          .maxAnisotropy = 0.0F,
                          .compareEnable = VK_FALSE,
                          .compareOp = VK_COMPARE_OP_ALWAYS,
                          .minLod = 0.0F,
                          .maxLod = 1.0F,
                          .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
                          .unnormalizedCoordinates = VK_FALSE,
                      })
                      .release();
}

auto Context::get_allocator_implementation() -> IAllocator & {
  assert(allocator_impl != nullptr &&
         "Allocator implementation must be set before use");
  return *allocator_impl;
}

#pragma region Destroyers

auto Context::destroy(TextureHandle handle) -> void {
  SCOPE_EXIT {
    if (const auto exp = texture_pool.destroy(handle); !exp.has_value()) {
      std::cerr << "Failed to destroy texture: "
                << std::to_underlying(exp.error()) << std::endl;
    }
    needs_update() = true;
  };

  const auto maybe_texture = texture_pool.get(handle);
  if (!maybe_texture.has_value()) {
    std::cerr << "Invalid texture handle: " << handle.index() << std::endl
              << "Generation: " << handle.generation() << std::endl;
    return;
  }

  auto *texture = maybe_texture.value();
  if (texture == nullptr) {
    return;
  }

  // Lets destroy the allocation, the image, the view, the storage view, the
  // layer/mip views.
  pre_frame_task([&allocator = get_allocator_implementation(),
                  img = texture](auto device, auto *allocation_callbacks) {
    for (const auto view = img->get_mip_layers_image_views();
         const auto &v : view) {
      if (VK_NULL_HANDLE == v) {
        continue;
      }
      vkDestroyImageView(device, v, allocation_callbacks);
    }

    vkDestroyImageView(device, img->get_image_view(), allocation_callbacks);
    if (const auto storage_view = img->get_storage_image_view())
      vkDestroyImageView(device, storage_view, allocation_callbacks);
  });

  if (!texture->owns_self()) {
    return;
  }

  pre_frame_task([&alloc = *allocator_impl, tex = texture->get_image()](
                     auto, auto) { alloc.deallocate_image(tex); });
}

auto Context::destroy(BufferHandle handle) -> void {
  TODO("Implement buffer destruction");
}

auto Context::destroy(ComputePipelineHandle handle) -> void {
  TODO("Implement compute pipeline destruction");
}

auto Context::destroy(GraphicsPipelineHandle handle) -> void {
  TODO("Implement graphics pipeline destruction");
}

auto Context::destroy(SamplerHandle handle) -> void {
  if (!handle.valid()) {
    return;
  }

  const auto maybe_sampler = get_sampler_pool().get(handle);
  if (!maybe_sampler.has_value()) {
    return;
  }

  auto sampler = *maybe_sampler.value();
  if (sampler == VK_NULL_HANDLE) {
    return;
  }

  pre_frame_task([ptr = sampler](auto device, auto *allocation_callbacks) {
    vkDestroySampler(device, ptr, allocation_callbacks);
  });

  if (auto expected = get_sampler_pool().destroy(handle);
      !expected.has_value()) {
    std::cerr << "Failed to destroy sampler: "
              << std::to_underlying(expected.error()) << std::endl;
  }
}

auto Context::destroy(ShaderModuleHandle handle) -> void {
  TODO("Implement shader module destruction");
}

#pragma endregion Destroyers

} // namespace VkBindless