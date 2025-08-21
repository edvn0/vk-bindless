#include "vk-bindless/vulkan_context.hpp"

#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/handle.hpp"
#include "vk-bindless/scope_exit.hpp"
#include "vk-bindless/swapchain.hpp"
#include "vk-bindless/texture.hpp"
#include "vk-bindless/transitions.hpp"

#include <ostream>
#include <thread>
#include <vk_mem_alloc.h>

#include <VkBootstrap.h>
#include <iostream>
#include <memory>
#include <vulkan/vulkan_core.h>

#define TODO(message)                                                          \
  do {                                                                         \
    throw std::runtime_error("TODO: " message);                                \
  } while (false)

namespace VkBindless {

namespace {
auto
format_to_vk_format(const Format format) -> VkFormat
{
    switch (format) {
    case Format::Invalid:
        return VK_FORMAT_UNDEFINED;

    case Format::R_UN8:   return VK_FORMAT_R8_UNORM;
    case Format::R_UI16:  return VK_FORMAT_R16_UINT;
    case Format::R_UI32:  return VK_FORMAT_R32_UINT;
    case Format::R_UN16:  return VK_FORMAT_R16_UNORM;
    case Format::R_F16:   return VK_FORMAT_R16_SFLOAT;
    case Format::R_F32:   return VK_FORMAT_R32_SFLOAT;

    case Format::RG_UN8:  return VK_FORMAT_R8G8_UNORM;
    case Format::RG_UI16: return VK_FORMAT_R16G16_UINT;
    case Format::RG_UI32: return VK_FORMAT_R32G32_UINT;
    case Format::RG_UN16: return VK_FORMAT_R16G16_UNORM;
    case Format::RG_F16:  return VK_FORMAT_R16G16_SFLOAT;
    case Format::RG_F32:  return VK_FORMAT_R32G32_SFLOAT;

    case Format::RGBA_UN8:   return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA_UI32:  return VK_FORMAT_R32G32B32A32_UINT;
    case Format::RGBA_F16:   return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::RGBA_F32:   return VK_FORMAT_R32G32B32A32_SFLOAT;
    case Format::RGBA_SRGB8: return VK_FORMAT_R8G8B8A8_SRGB;

    case Format::BGRA_UN8:   return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::BGRA_SRGB8: return VK_FORMAT_B8G8R8A8_SRGB;

    case Format::A2B10G10R10_UN: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Format::A2R10G10B10_UN: return VK_FORMAT_A2R10G10B10_UNORM_PACK32;

    case Format::ETC2_RGB8:   return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
    case Format::ETC2_SRGB8:  return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
    case Format::BC7_RGBA:    return VK_FORMAT_BC7_UNORM_BLOCK;

    case Format::Z_UN16:      return VK_FORMAT_D16_UNORM;
    case Format::Z_UN24:      return VK_FORMAT_X8_D24_UNORM_PACK32;
    case Format::Z_F32:       return VK_FORMAT_D32_SFLOAT;
    case Format::Z_UN24_S_UI8:return VK_FORMAT_D24_UNORM_S8_UINT;
    case Format::Z_F32_S_UI8: return VK_FORMAT_D32_SFLOAT_S8_UINT;

    case Format::YUV_NV12:    return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    case Format::YUV_420p:    return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    }
    return VK_FORMAT_UNDEFINED;
}
auto
blend_factor_to_vk_blend_factor(BlendFactor blend_factor) -> VkBlendFactor
{
    switch (blend_factor) {
    case BlendFactor::Zero:                 return VK_BLEND_FACTOR_ZERO;
    case BlendFactor::One:                  return VK_BLEND_FACTOR_ONE;
    case BlendFactor::SrcColor:             return VK_BLEND_FACTOR_SRC_COLOR;
    case BlendFactor::OneMinusSrcColor:     return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case BlendFactor::SrcAlpha:             return VK_BLEND_FACTOR_SRC_ALPHA;
    case BlendFactor::OneMinusSrcAlpha:     return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case BlendFactor::DstColor:             return VK_BLEND_FACTOR_DST_COLOR;
    case BlendFactor::OneMinusDstColor:     return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case BlendFactor::DstAlpha:             return VK_BLEND_FACTOR_DST_ALPHA;
    case BlendFactor::OneMinusDstAlpha:     return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case BlendFactor::SrcAlphaSaturated:    return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case BlendFactor::BlendColor:           return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case BlendFactor::OneMinusBlendColor:   return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case BlendFactor::BlendAlpha:           return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case BlendFactor::OneMinusBlendAlpha:   return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    case BlendFactor::Src1Color:            return VK_BLEND_FACTOR_SRC1_COLOR;
    case BlendFactor::OneMinusSrc1Color:    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case BlendFactor::Src1Alpha:            return VK_BLEND_FACTOR_SRC1_ALPHA;
    case BlendFactor::OneMinusSrc1Alpha:    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    }
    return VK_BLEND_FACTOR_ZERO;
}

auto
blend_op_to_vk_blend_op(BlendOp blend_op) -> VkBlendOp
{
  switch (blend_op) {
    case BlendOp::Add:
      return VK_BLEND_OP_ADD;
    case BlendOp::Subtract:
      return VK_BLEND_OP_SUBTRACT;
    case BlendOp::ReverseSubtract:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
    case BlendOp::Min:
      return VK_BLEND_OP_MIN;
    case BlendOp::Max:
      return VK_BLEND_OP_MAX;
  }
  return VK_BLEND_OP_ADD;
}
auto
topology_to_vk_topology(Topology topology) -> VkPrimitiveTopology
{
    switch (topology) {
    case Topology::Point:          return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case Topology::Line:           return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case Topology::LineStrip:      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case Topology::Triangle:       return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case Topology::TriangleStrip:  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case Topology::Patch:          return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    }
    return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
}

auto
polygon_mode_to_vk_polygon_mode(PolygonMode polygon_mode) -> VkPolygonMode
{
    switch (polygon_mode) {
    case PolygonMode::Fill: return VK_POLYGON_MODE_FILL;
    case PolygonMode::Line: return VK_POLYGON_MODE_LINE;
    }
    return VK_POLYGON_MODE_FILL;
}

auto
cull_mode_to_vk_cull_mode(CullMode cull_mode) -> VkCullModeFlags
{
    switch (cull_mode) {
    case CullMode::None:  return VK_CULL_MODE_NONE;
    case CullMode::Front: return VK_CULL_MODE_FRONT_BIT;
    case CullMode::Back:  return VK_CULL_MODE_BACK_BIT;
    }
    return VK_CULL_MODE_NONE;
}

auto
winding_to_vk_winding(WindingMode winding) -> VkFrontFace
{
    switch (winding) {
    case WindingMode::CCW: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
    case WindingMode::CW:  return VK_FRONT_FACE_CLOCKWISE;
    }
    return VK_FRONT_FACE_COUNTER_CLOCKWISE;
}

auto
stencil_op_to_vk_stencil_op(StencilOp stencil_op) -> VkStencilOp
{
    switch (stencil_op) {
    case StencilOp::Keep:           return VK_STENCIL_OP_KEEP;
    case StencilOp::Zero:           return VK_STENCIL_OP_ZERO;
    case StencilOp::Replace:        return VK_STENCIL_OP_REPLACE;
    case StencilOp::IncrementClamp: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case StencilOp::DecrementClamp: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case StencilOp::Invert:         return VK_STENCIL_OP_INVERT;
    case StencilOp::IncrementWrap:  return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case StencilOp::DecrementWrap:  return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    }
    return VK_STENCIL_OP_KEEP;
}

auto compare_op_to_vk_compare_op(CompareOp compare_op) -> VkCompareOp {
    switch (compare_op) {
    case CompareOp::Never:        return VK_COMPARE_OP_NEVER;
    case CompareOp::Less:         return VK_COMPARE_OP_LESS;
    case CompareOp::Equal:        return VK_COMPARE_OP_EQUAL;
    case CompareOp::LessEqual:    return VK_COMPARE_OP_LESS_OR_EQUAL;
    case CompareOp::Greater:      return VK_COMPARE_OP_GREATER;
    case CompareOp::NotEqual:     return VK_COMPARE_OP_NOT_EQUAL;
    case CompareOp::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case CompareOp::AlwaysPass:   return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_ALWAYS;
}

auto
create_timeline_semaphore(const VkDevice device,
                          const std::uint64_t initial_value) -> VkSemaphore
{
  const VkSemaphoreTypeCreateInfo semaphoreTypeCreateInfo = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    .pNext = nullptr,
    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
    .initialValue = initial_value,
  };
  const VkSemaphoreCreateInfo ci = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    .pNext = &semaphoreTypeCreateInfo,
    .flags = 0,
  };
  VkSemaphore semaphore = VK_NULL_HANDLE;
  VK_VERIFY(vkCreateSemaphore(device, &ci, nullptr, &semaphore));
  set_name_for_object(
    device, VK_OBJECT_TYPE_SEMAPHORE, semaphore, "Timeline Semaphore");
  return semaphore;
}

constexpr VkShaderStageFlags all_stages_flags =
  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
  VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
  VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
}


Context::~Context()
{
  vkDeviceWaitIdle(vkb_device.device);

  swapchain.reset();

  destroy(dummy_texture);
  destroy(dummy_sampler);

  process_callbacks();

  // Destroy all resources
  texture_pool.clear();
  sampler_pool.clear();
}

constexpr size_t QUEUE_SIZE = 1024;

template<typename Message = std::string>
struct LockFreeQueue
{
  std::array<Message, QUEUE_SIZE> buffer{};
  std::atomic<size_t> head{ 0 };
  std::atomic<size_t> tail{ 0 };

  auto push(Message&& msg)
  {
    auto t = tail.load(std::memory_order_relaxed);
    const auto next = (t + 1) % QUEUE_SIZE;
    if (next == head.load(std::memory_order_acquire)) {
      return false;
    }
    buffer[t] = std::move(msg);
    tail.store(next, std::memory_order_release);
    return true;
  }

  auto pop(Message& out)
  {
    const auto h = head.load(std::memory_order_relaxed);
    if (h == tail.load(std::memory_order_acquire)) {
      return false; // empty
    }
    out = std::move(buffer[h]);
    head.store((h + 1) % QUEUE_SIZE, std::memory_order_release);
    return true;
  }
};

static LockFreeQueue messages;

static std::jthread thread{
  [](const std::stop_token& stoken) {
    std::string msg;
    while (!stoken.stop_requested()) {
      while (messages.pop(msg)) {
        std::cout << "[VK] " << msg << "\n";
      }
      std::flush(std::cout);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  },
};

static auto
logger(VkDebugUtilsMessageSeverityFlagBitsEXT,
       VkDebugUtilsMessageTypeFlagsEXT,
       const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
       void*) -> VkBool32
{
#ifdef IS_DEBUG
  std::cout << std::string{ callbackData->pMessage } << '\n';
  return VK_FALSE;
#else
  messages.push(callbackData->pMessage);
  return VK_FALSE;
#endif
}

auto
query_vulkan_properties(VkPhysicalDevice physical_device, auto& props) -> void
{
  vkGetPhysicalDeviceProperties(physical_device, &props.base);

  props.fourteen.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
  props.fourteen.pNext = nullptr;

  props.thirteen.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
  props.thirteen.pNext = &props.fourteen;

  props.twelve.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
  props.twelve.pNext = &props.thirteen;

  props.eleven.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
  props.eleven.pNext = &props.twelve;

  VkPhysicalDeviceProperties2 device_props2{};
  device_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  device_props2.pNext = &props.eleven;

  vkGetPhysicalDeviceProperties2(physical_device, &device_props2);

  props.base = device_props2.properties;
}

auto
Context::create(std::function<VkSurfaceKHR(VkInstance)>&& surface_fn)
  -> Expected<std::unique_ptr<IContext>, ContextError>
{
#if IS_DEBUG
  constexpr auto request_validation = true;
#else
  constexpr auto request_validation = false;
#endif
  vkb::InstanceBuilder builder;
  auto inst_ret = builder.set_app_name("Bindless Vulkan")
                    .request_validation_layers(request_validation)
                    .set_debug_callback(&logger)
                    .require_api_version(1, 4, 0)
                    .build();
  if (!inst_ret) {
    return unexpected<ContextError>(
      ContextError{ "Failed to create Vulkan instance" });
  }

  vkb::Instance vkb_instance = inst_ret.value();

  VkSurfaceKHR surf = std::move(surface_fn)(vkb_instance.instance);

  bool is_headless = false;
  if (VK_NULL_HANDLE == surf) {
    std::cerr << "Headless rendering is enabled." << std::endl;
    is_headless = true;
  }

  vkb::PhysicalDeviceSelector selector{ vkb_instance, surf };
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
    return unexpected<ContextError>(
      ContextError{ "Failed to select Vulkan physical device" });
  }

  const vkb::PhysicalDevice& vkb_physical = phys_ret.value();

  VkPhysicalDeviceVulkan11Features vk11_features{};
  vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  vk11_features.shaderDrawParameters = VK_TRUE;

  VkPhysicalDeviceVulkan12Features vk12_features{};
  vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  vk12_features.pNext = &vk11_features;
  vk12_features.descriptorIndexing = VK_TRUE;
  vk12_features.timelineSemaphore = VK_TRUE;
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
  vk13_features.synchronization2 = VK_TRUE;

  VkPhysicalDeviceVulkan14Features vk14_features{};
  vk14_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
  vk14_features.pNext = &vk13_features;

  vkb::DeviceBuilder dev_builder{ vkb_physical };
  dev_builder.add_pNext(&vk14_features);

  auto dev_ret = dev_builder.build();
  if (!dev_ret) {
    return unexpected<ContextError>(
      ContextError{ "Failed to create logical device" });
  }

  const vkb::Device& vkb_device = dev_ret.value();

  auto context = std::make_unique<Context>();
  context->vkb_instance = vkb_instance;
  context->vkb_physical_device = vkb_physical;
  context->vkb_device = vkb_device;
  context->surface = surf;
  context->is_headless = is_headless;

  std::vector<VkSurfaceFormatKHR> device_formats;
  std::vector<VkFormat> device_depth_formats;
  std::vector<VkPresentModeKHR> device_present_modes;

  constexpr std::array depth_formats = {
    VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_D16_UNORM_S8_UINT,  VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D16_UNORM,
  };
  for (const auto& depth_format : depth_formats) {
    VkFormatProperties format_properties{};
    vkGetPhysicalDeviceFormatProperties(
      vkb_physical.physical_device, depth_format, &format_properties);

    if (format_properties.optimalTilingFeatures) {
      device_depth_formats.push_back(depth_format);
    }
  }

  std::uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(
    vkb_physical.physical_device, surf, &format_count, nullptr);

  if (format_count) {
    device_formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(
      vkb_physical.physical_device, surf, &format_count, device_formats.data());
  }

  std::uint32_t present_mode_count{};
  vkGetPhysicalDeviceSurfacePresentModesKHR(
    vkb_physical.physical_device, surf, &present_mode_count, nullptr);

  if (present_mode_count) {
    device_present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(vkb_physical.physical_device,
                                              surf,
                                              &present_mode_count,
                                              device_present_modes.data());
  }

  auto allocator = IAllocator::create_allocator(
    vkb_instance.instance, vkb_physical.physical_device, vkb_device.device);

  context->allocator_impl = std::move(allocator);

  context->device_surface_formats = std::move(device_formats);
  context->device_depth_formats = std::move(device_depth_formats);
  context->device_present_modes = std::move(device_present_modes);
  VkSurfaceCapabilitiesKHR device_surface_capabilities = {};
  if (surf) {
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      vkb_physical.physical_device, surf, &device_surface_capabilities);
  }
  context->device_surface_capabilities = device_surface_capabilities;

  query_vulkan_properties(vkb_physical.physical_device,
                          context->vulkan_properties);
  context->has_swapchain_maintenance_1 = false;
  context->immediate_commands = std::make_unique<ImmediateCommands>(
    vkb_device.device, context->graphics_queue_family, "Immediate Commands");

  {
    auto q = vkb_device.get_queue(vkb::QueueType::graphics);
    if (!q)
      return unexpected<ContextError>(ContextError{ "Missing graphics queue" });
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

  context->swapchain =
    !is_headless ? Unique<Swapchain>{ new Swapchain{ *context, 1920U, 1080U } }
                 : VK_NULL_HANDLE;
  context->timeline_semaphore =
    context->swapchain
      ? create_timeline_semaphore(
          vkb_device.device, context->swapchain->swapchain_image_count() - 1)
      : VK_NULL_HANDLE;

  context->create_placeholder_resources();
  context->update_resource_bindings();

  return context;
}

auto
Context::get_queue(const Queue queue) const -> Expected<VkQueue, ContextError>
{
  switch (queue) {
    using enum VkBindless::Queue;
    case Graphics:
      return graphics_queue;
    case Compute:
      return compute_queue;
    case Transfer:
      return transfer_queue;
  }
  return unexpected<ContextError>(ContextError{ "Invalid queue requested" });
}

auto
Context::get_queue_family_index(const Queue queue) const
  -> Expected<std::uint32_t, ContextError>
{
  switch (queue) {
    using enum VkBindless::Queue;
    case Graphics:
      return graphics_queue_family;
    case Compute:
      return compute_queue_family;
    case Transfer:
      return transfer_queue_family;
  }
  return unexpected<ContextError>(
    ContextError{ "Invalid queue family requested" });
}

auto
Context::get_device() const -> const VkDevice&
{
  return vkb_device.device;
}

auto
Context::get_physical_device() const -> const VkPhysicalDevice&
{
  return vkb_physical_device.physical_device;
}

auto
Context::get_instance() const -> const VkInstance&
{
  return vkb_instance.instance;
}

auto
Context::get_queue_unsafe(Queue queue) const -> const VkQueue&
{
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

auto
Context::get_queue_family_index_unsafe(Queue queue) const -> std::uint32_t
{
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

void
Context::update_resource_bindings()
{
  base::update_resource_bindings();

  if (const auto should_update = needs_update(); !should_update) [[likely]] {
    return;
  }

  constexpr auto grow_factor = 2.F;
  auto current_textures = 1U;
  auto current_samplers = 1U;

  constexpr auto grow_pool = [](const auto& pool, auto out_max) {
    while (pool.size() > out_max) {
      out_max =
        static_cast<std::uint32_t>(static_cast<float>(out_max) * grow_factor);
    }
    return out_max;
  };

  current_textures = grow_pool(texture_pool, current_textures);
  current_samplers = grow_pool(sampler_pool, current_samplers);

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
  const auto* realised_image = texture_pool.get(dummy_texture).value();
  const auto& dummy_image_view = realised_image->get_image_view();

  texture_pool.for_each_valid([v = dummy_image_view,
                               &imgs = sampled_images,
                               &storgs = storage_images](const auto& object) {
    const auto& view = object.get_image_view();
    const auto& storage_view = object.get_storage_image_view()
                                 ? object.get_storage_image_view()
                                 : object.get_image_view();

    const auto is_available =
      VK_SAMPLE_COUNT_1_BIT ==
      (object.get_sample_count() & VK_SAMPLE_COUNT_1_BIT);
    const auto is_sampled = object.is_sampled() && is_available;
    const auto is_storage = object.is_storage() && is_available;

    imgs.emplace_back(
      VK_NULL_HANDLE, is_sampled ? view : v, VK_IMAGE_LAYOUT_GENERAL);

    storgs.emplace_back(
      VK_NULL_HANDLE, is_storage ? storage_view : v, VK_IMAGE_LAYOUT_GENERAL);
  });

  std::vector<VkDescriptorImageInfo> sampler_infos;
  sampler_infos.reserve(sampler_pool.size());

  auto realised_sampler = *sampler_pool.get(dummy_sampler).value();

  sampler_pool.for_each_valid(
    [&sampler_infos, &realised_sampler](const auto& object) {
      sampler_infos.emplace_back(object ? object : realised_sampler,
                                 VK_NULL_HANDLE,
                                 VK_IMAGE_LAYOUT_UNDEFINED);
    });

  std::array<VkWriteDescriptorSet, 3> writes{};
  auto write_count = 0U;
  if (!sampled_images.empty()) {
    writes[0] = VkWriteDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = nullptr,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = static_cast<std::uint32_t>(sampled_images.size()),
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .pImageInfo = sampled_images.data(),
      .pBufferInfo = nullptr,
      .pTexelBufferView = nullptr,
    };
    write_count++;
  }
  if (!sampler_infos.empty()) {
    writes[1] = VkWriteDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = nullptr,
      .dstSet = descriptor_set,
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = static_cast<std::uint32_t>(sampler_infos.size()),
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
      .pImageInfo = sampler_infos.data(),
      .pBufferInfo = nullptr,
      .pTexelBufferView = nullptr,
    };
    write_count++;
  }
  if (!storage_images.empty()) {
    writes[2] = VkWriteDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = nullptr,
      .dstSet = descriptor_set,
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = static_cast<std::uint32_t>(storage_images.size()),
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = storage_images.data(),
      .pBufferInfo = nullptr,
      .pTexelBufferView = nullptr,
    };
    write_count++;
  }

  if (write_count != 0) {
    vkQueueWaitIdle(get_queue_unsafe(Queue::Graphics));
    vkQueueWaitIdle(get_queue_unsafe(Queue::Compute));
    vkUpdateDescriptorSets(
      vkb_device.device, write_count, writes.data(), 0, nullptr);
  }

  needs_update() = false;
}

auto
Context::get_dsl_binding(const std::uint32_t index,
                         const VkDescriptorType descriptor_type,
                         const uint32_t max_count)
  -> VkDescriptorSetLayoutBinding
{
  const auto binding = VkDescriptorSetLayoutBinding{
    .binding = index,
    .descriptorType = descriptor_type,
    .descriptorCount = max_count,
    .stageFlags = all_stages_flags,
    .pImmutableSamplers = nullptr,
  };

  return binding;
}

auto
Context::grow_descriptor_pool(std::uint32_t new_max_textures,
                              std::uint32_t new_max_samplers)
  -> Expected<void, ContextError>
{
  static auto vkGetPhysicalDeviceProperties2 =
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vkGetInstanceProcAddr(
      vkb_instance.instance, "vkGetPhysicalDeviceProperties2"));

  current_max_textures = new_max_textures;
  current_max_samplers = new_max_samplers;

  if (vkGetPhysicalDeviceProperties2 == nullptr) {
    return unexpected<ContextError>(
      ContextError{ "Failed to get vkGetPhysicalDeviceProperties2" });
  }

  VkPhysicalDeviceVulkan12Properties props12{};
  props12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;

  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &props12;

  vkGetPhysicalDeviceProperties2(vkb_physical_device, &props2);

#define VERIFY(condition, message)                                             \
  if (!(condition)) {                                                          \
    return unexpected<ContextError>(ContextError{ message });                  \
  }

  VERIFY(current_max_samplers <=
           props12.maxDescriptorSetUpdateAfterBindSamplers,
         "Maximum number of samplers exceeds device limit");
  VERIFY(current_max_textures <=
           props12.maxDescriptorSetUpdateAfterBindSampledImages,
         "Maximum number of sampled images exceeds device limit");

  if (const auto could = update_descriptor_sets(); !could.has_value()) {
    return unexpected<ContextError>(could.error());
  }

  return {};
}

auto
Context::update_descriptor_sets() -> Expected<void, ContextError>
{
  if (descriptor_pool != VK_NULL_HANDLE) {
    pre_frame_task(
      [pool = descriptor_pool](auto dev, auto* allocation_callbacks) {
        vkDestroyDescriptorPool(dev, pool, allocation_callbacks);
      });
  }
  if (descriptor_set_layout != VK_NULL_HANDLE) {
    pre_frame_task(
      [layout = descriptor_set_layout](auto dev, auto* allocation_callbacks) {
        vkDestroyDescriptorSetLayout(dev, layout, allocation_callbacks);
      });
  }

  const auto bindings = std::array{
    get_dsl_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, current_max_textures),
    get_dsl_binding(1, VK_DESCRIPTOR_TYPE_SAMPLER, current_max_samplers),
    get_dsl_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, current_max_textures),
    get_dsl_binding(
      3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, current_max_textures),
  };

  constexpr VkDescriptorSetLayoutCreateFlags flags =
    VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
    VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
    VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

  constexpr std::array<VkDescriptorSetLayoutCreateFlags, 4> binding_flags =
    std::array{ flags, flags, flags, flags };

  const VkDescriptorSetLayoutBindingFlagsCreateInfo
    binding_flags_create_info = {
      .sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .pNext = nullptr,
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

  if (vkCreateDescriptorSetLayout(
        vkb_device.device, &dsl_create_info, nullptr, &descriptor_set_layout) !=
      VK_SUCCESS) {
    return unexpected<ContextError>(
      ContextError{ "Failed to create descriptor set layout" });
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
    .pNext = nullptr,
    .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |
             VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    .maxSets = 1,
    .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
    .pPoolSizes = pool_sizes.data(),
  };
  if (vkCreateDescriptorPool(
        vkb_device.device, &pool_create_info, nullptr, &descriptor_pool) !=
      VK_SUCCESS) {
    return unexpected<ContextError>(
      ContextError{ "Failed to create descriptor pool" });
  }

  // Allocate the descriptor set
  VkDescriptorSetAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .pNext = nullptr,
    .descriptorPool = descriptor_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &descriptor_set_layout
  };
  if (vkAllocateDescriptorSets(
        vkb_device.device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
    return unexpected<ContextError>(
      ContextError{ "Failed to allocate descriptor set" });
  }

  return {};
}

auto
Context::create_placeholder_resources() -> void
{
  constexpr std::array<const std::uint8_t, 4> dummy_white_texture = {
    255,
    255,
    255,
    255,
  };
  dummy_texture =
    VkTexture::create(
      *this,
      VkTextureDescription{
        .data = std::span(dummy_white_texture),
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = { 1, 1, 1 },
        .usage_flags = TextureUsageFlags::Sampled | TextureUsageFlags::Storage,
        .debug_name = "Dummy White Texture (1x1)",
      })
      .release();
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

auto
Context::get_allocator_implementation() -> IAllocator&
{
  assert(allocator_impl != nullptr &&
         "Allocator implementation must be set before use");
  return *allocator_impl;
}

auto
Context::resize_swapchain(const std::uint32_t width, const std::uint32_t height)
  -> void
{
  if (swapchain) {
    swapchain->resize(width, height);
  }
}

auto
Context::acquire_command_buffer() -> ICommandBuffer&
{
  command_buffer = CommandBuffer(*this);
  return command_buffer;
}

auto
Context::submit(ICommandBuffer& cmd_buffer, const TextureHandle present)
  -> Expected<SubmitHandle, std::string>
{
  const auto& vk_buffer = dynamic_cast<CommandBuffer&>(cmd_buffer);

#if defined(LVK_WITH_TRACY_GPU)
  TracyVkCollect(pimpl_->tracyVkCtx_, vk_buffer.get_command_buffer());
#endif // LVK_WITH_TRACY_GPU

  if (present) {
    const auto* tex = *texture_pool.get(present);

    assert(tex->is_swapchain_image());

    Transition::swapchain_image(vk_buffer.get_command_buffer(),
                                tex->get_image(),
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  }

  const auto has_swapchain = swapchain != nullptr;
  const bool should_present = has_swapchain && present;

  if (should_present) {
    const std::uint64_t signal_value =
      swapchain->current_frame_index() + swapchain->swapchain_image_count();
    swapchain
      ->timeline_wait_values[swapchain->swapchain_current_image_index()] =
      signal_value;
    immediate_commands->signal_semaphore(timeline_semaphore, signal_value);
  }

  command_buffer.last_submit_handle =
    immediate_commands->submit(*command_buffer.wrapper);

  if (should_present) {
    const auto could =
      swapchain->present(immediate_commands->acquire_last_submit_semaphore());
    if (!could) {
      return unexpected<std::string>("Failed to present swapchain");
    }
  }

  process_callbacks();

  auto handle = command_buffer.last_submit_handle;

  command_buffer = {};

  return handle;
}

auto
Context::get_current_swapchain_texture() -> TextureHandle
{
  return swapchain->current_texture();
}


auto
Context::get_pipeline(GraphicsPipelineHandle handle, std::uint32_t viewMask) -> VkPipeline
{
  auto* rps = *get_graphics_pipeline_pool().get(handle);

  if (!rps) {
    return VK_NULL_HANDLE;
  }

  if (rps->last_descriptor_set_layout != descriptor_set_layout ||
      rps->view_mask != viewMask) {
    pre_frame_task(
      [l = rps->get_layout()](auto dev, auto callbacks) {
        vkDestroyPipelineLayout(dev, l, callbacks);
      });
    pre_frame_task([p = rps->get_pipeline()](auto dev, auto callbacks) {
      vkDestroyPipeline(dev, p, callbacks);
    });

    rps->pipeline = VK_NULL_HANDLE;
    rps->last_descriptor_set_layout = descriptor_set_layout;
    rps->view_mask = viewMask;
  }

  if (rps->pipeline != VK_NULL_HANDLE) {
    return rps->pipeline;
  }

  // build a new Vulkan pipeline

  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  const auto& desc = rps->description;

  const auto colour_attachments_count =
    rps->description.get_colour_attachments_count();

  // Not all attachments are valid. We need to create color blend attachments
  // only for active attachments
  std::array<VkPipelineColorBlendAttachmentState, max_colour_attachments>
    color_blend_attachment_states{};
  std::array<VkFormat, max_colour_attachments> color_attachment_formats {};

  for (auto i = 0U; i != colour_attachments_count; i++) {
    const auto& [format,
                 blend_enabled,
                 rgb_blend_op,
                 alpha_blend_op,
                 src_rgb_blend_factor,
                 src_alpha_blend_factor,
                 dst_rgb_blend_factor,
                 dst_alpha_blend_factor] = desc.color[i];
    assert(format != Format::Invalid);
    color_attachment_formats[i] = format_to_vk_format(format);
    if (!blend_enabled) {
      color_blend_attachment_states[i] = VkPipelineColorBlendAttachmentState{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      };
    } else {
      color_blend_attachment_states[i] = VkPipelineColorBlendAttachmentState{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor =
          blend_factor_to_vk_blend_factor(src_rgb_blend_factor),
        .dstColorBlendFactor =
          blend_factor_to_vk_blend_factor(dst_rgb_blend_factor),
        .colorBlendOp = blend_op_to_vk_blend_op(rgb_blend_op),
        .srcAlphaBlendFactor =
          blend_factor_to_vk_blend_factor(src_alpha_blend_factor),
        .dstAlphaBlendFactor =
          blend_factor_to_vk_blend_factor(dst_alpha_blend_factor),
        .alphaBlendOp = blend_op_to_vk_blend_op(alpha_blend_op),
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      };
    }
  }

  const auto* shader = *get_shader_module_pool().get(desc.shader);

  assert(shader);

  /*  if (tescModule || teseModule || desc.patchControlPoints) {
      LVK_ASSERT_MSG(tescModule && teseModule, "Both tessellation control and
    evaluation shaders should be provided"); LVK_ASSERT(desc.patchControlPoints
    > 0 && desc.patchControlPoints <=
    vkPhysicalDeviceProperties2_.properties.limits.maxTessellationPatchSize);
    }
  */
  const VkPipelineVertexInputStateCreateInfo ciVertexInputState = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = rps->binding_count,
    .pVertexBindingDescriptions =
      rps->binding_count > 0 ? rps->bindings.data() : nullptr,
    .vertexAttributeDescriptionCount = rps->attribute_count,
    .pVertexAttributeDescriptions =
      rps->attribute_count > 0 ? rps->attributes.data() : nullptr,
  };

  std::array<VkSpecializationMapEntry, SpecialisationConstantDescription::max_specialization_constants> entries{};

  static constexpr auto get_pipeline_specialisation_info = [](const SpecialisationConstantDescription& desc,
                                                    auto& spec_entries) {
      const auto num_entries = desc.get_specialisation_constants_count();
    for (auto i = 0U; i < num_entries; ++i) {
        const auto& [constant_id, offset, size] = desc.entries.at(i);
      spec_entries[i] = VkSpecializationMapEntry{
        .constantID = constant_id,
        .offset = offset,
        .size = size,
      };
    }

    return VkSpecializationInfo{
      .mapEntryCount = num_entries,
      .pMapEntries = spec_entries.data(),
      .dataSize = desc.data.size_bytes(),
      .pData = desc.data.data(),
    };
  };

  const VkSpecializationInfo si = get_pipeline_specialisation_info(
    desc.specialisation_constants, entries);

  // create pipeline layout
  {
    auto&& [size, flags] = shader->get_push_constant_info();

    static constexpr auto get_aligned_size = [](const auto s,
                                                const auto alignment) {
      return (s + alignment - 1) & ~(alignment - 1);
    };

    // duplicate for MoltenVK
    const VkDescriptorSetLayout dsls[] = { descriptor_set_layout,
                                           descriptor_set_layout,
                                           descriptor_set_layout, descriptor_set_layout };
    auto min_align = vulkan_properties.base.limits.minUniformBufferOffsetAlignment;

    const VkPushConstantRange range = {
      .stageFlags = rps->stage_flags,
      .offset = 0,
      .size = static_cast<uint32_t>(get_aligned_size(size, min_align)),
    };
    const VkPipelineLayoutCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = static_cast<std::uint32_t>(std::size(dsls)),
      .pSetLayouts = dsls,
      .pushConstantRangeCount = size ? 1u : 0u,
      .pPushConstantRanges = size ? &range : nullptr,
    };
    vkCreatePipelineLayout(get_device(), &ci, nullptr, &layout);
    set_name_for_object(
      get_device(),
      VK_OBJECT_TYPE_PIPELINE_LAYOUT,
      layout,
      std::format("Pipeline_Layout_{}",
                  !desc.debug_name.empty() ? desc.debug_name : "Unnamed"));
  }

  std::array dynamic_states = {
    VK_DYNAMIC_STATE_VIEWPORT,          VK_DYNAMIC_STATE_SCISSOR,
    VK_DYNAMIC_STATE_DEPTH_BIAS,        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
    VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,  VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE
  };

  VkPipelineDynamicStateCreateInfo ci_dynamic{};
  ci_dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  ci_dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
  ci_dynamic.pDynamicStates = dynamic_states.data();

  VkPipelineInputAssemblyStateCreateInfo ci_ia{};
  ci_ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  ci_ia.topology = topology_to_vk_topology(desc.topology);

  VkPipelineRasterizationStateCreateInfo ci_rs{};
  ci_rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  ci_rs.polygonMode = polygon_mode_to_vk_polygon_mode(desc.polygon_mode);
  ci_rs.cullMode = cull_mode_to_vk_cull_mode(desc.cull_mode);
  ci_rs.frontFace = winding_to_vk_winding(desc.winding);
  ci_rs.depthBiasEnable = VK_FALSE;
  ci_rs.lineWidth = 1.0f;

  auto getVulkanSampleCountFlags = [](const uint32_t sample_count, VkSampleCountFlags max_samples_mask) {
    if (sample_count <= 1 || VK_SAMPLE_COUNT_2_BIT > max_samples_mask) {
      return VK_SAMPLE_COUNT_1_BIT;
    }
    if (sample_count <= 2 || VK_SAMPLE_COUNT_4_BIT > max_samples_mask) {
      return VK_SAMPLE_COUNT_2_BIT;
    }
    if (sample_count <= 4 || VK_SAMPLE_COUNT_8_BIT > max_samples_mask) {
      return VK_SAMPLE_COUNT_4_BIT;
    }
    if (sample_count <= 8 || VK_SAMPLE_COUNT_16_BIT > max_samples_mask) {
      return VK_SAMPLE_COUNT_8_BIT;
    }
    if (sample_count <= 16 || VK_SAMPLE_COUNT_32_BIT > max_samples_mask) {
      return VK_SAMPLE_COUNT_16_BIT;
    }
    if (sample_count <= 32 || VK_SAMPLE_COUNT_64_BIT > max_samples_mask) {
      return VK_SAMPLE_COUNT_32_BIT;
    }
    return VK_SAMPLE_COUNT_64_BIT;
  };

  auto limits = vulkan_properties.base.limits.framebufferColorSampleCounts &
                vulkan_properties.base.limits.framebufferDepthSampleCounts;

  VkSampleCountFlagBits samples =
    getVulkanSampleCountFlags(desc.sample_count, limits);
  VkPipelineMultisampleStateCreateInfo ci_ms{};
  ci_ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ci_ms.rasterizationSamples = samples;
  ci_ms.sampleShadingEnable = desc.min_sample_shading > 0.0f ? VK_TRUE : VK_FALSE;
  ci_ms.minSampleShading = desc.min_sample_shading;

  VkStencilOpState front{};
  front.failOp = stencil_op_to_vk_stencil_op(desc.front_face_stencil.stencil_failure_operation);
  front.passOp =
    stencil_op_to_vk_stencil_op(desc.front_face_stencil.depth_stencil_pass_operation);
  front.depthFailOp =
    stencil_op_to_vk_stencil_op(desc.front_face_stencil.depth_failure_operation);
  front.compareOp =
    compare_op_to_vk_compare_op(desc.front_face_stencil.stencil_compare_op);
  front.compareMask = desc.front_face_stencil.read_mask;
  front.writeMask = desc.front_face_stencil.write_mask;
  front.reference = 0xFF;

  VkStencilOpState back{};
  back.failOp = stencil_op_to_vk_stencil_op(desc.back_face_stencil.stencil_failure_operation);
  back.passOp = stencil_op_to_vk_stencil_op(desc.back_face_stencil.depth_stencil_pass_operation);
  back.depthFailOp =
    stencil_op_to_vk_stencil_op(desc.back_face_stencil.depth_failure_operation);
  back.compareOp =
    compare_op_to_vk_compare_op(desc.back_face_stencil.stencil_compare_op);
  back.compareMask = desc.back_face_stencil.read_mask;
  back.writeMask = desc.back_face_stencil.write_mask;
  back.reference = 0xFF;

  VkPipelineDepthStencilStateCreateInfo ci_ds{};
  ci_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  ci_ds.depthTestEnable = VK_TRUE;                    // dynamic
  ci_ds.depthWriteEnable = VK_TRUE;                   // dynamic
  ci_ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; // dynamic
  ci_ds.depthBoundsTestEnable = VK_FALSE;
  ci_ds.stencilTestEnable =
    (desc.front_face_stencil.enabled || desc.back_face_stencil.enabled) ? VK_TRUE
                                                                    : VK_FALSE;
  ci_ds.front = front;
  ci_ds.back = back;

  VkPipelineViewportStateCreateInfo ci_vs{};
  ci_vs.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  ci_vs.viewportCount = 1;
  ci_vs.scissorCount = 1;

  VkPipelineColorBlendStateCreateInfo ci_cb{};
  ci_cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  ci_cb.attachmentCount = colour_attachments_count;
  ci_cb.pAttachments = color_blend_attachment_states.data();

  VkPipelineTessellationStateCreateInfo ci_ts{};
  bool has_tess = (shader->has_stage(ShaderStage::tessellation_control) &&
                   shader->has_stage(ShaderStage::tessellation_evaluation)) &&
                  desc.patch_control_points > 0;
  if (has_tess) {
    ci_ts.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    ci_ts.patchControlPoints = desc.patch_control_points;
  }

  std::vector<VkPipelineShaderStageCreateInfo> stages;
  shader->populate_stages(stages, si);

  VkPipelineRenderingCreateInfo ci_rendering{};
  ci_rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  ci_rendering.viewMask = viewMask;
  ci_rendering.colorAttachmentCount = colour_attachments_count;
  ci_rendering.pColorAttachmentFormats = color_attachment_formats.data();
  ci_rendering.depthAttachmentFormat = format_to_vk_format(desc.depth_format);
  ci_rendering.stencilAttachmentFormat = format_to_vk_format(desc.stencil_format);

  VkGraphicsPipelineCreateInfo ci_gp{};
  ci_gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  ci_gp.pNext = &ci_rendering;
  ci_gp.stageCount = static_cast<uint32_t>(stages.size());
  ci_gp.pStages = stages.data();
  ci_gp.pVertexInputState = &ciVertexInputState;
  ci_gp.pInputAssemblyState = &ci_ia;
  ci_gp.pViewportState = &ci_vs;
  ci_gp.pRasterizationState = &ci_rs;
  ci_gp.pMultisampleState = &ci_ms;
  ci_gp.pDepthStencilState = &ci_ds;
  ci_gp.pColorBlendState = &ci_cb;
  ci_gp.pDynamicState = &ci_dynamic;
  ci_gp.pTessellationState = has_tess ? &ci_ts : nullptr;
  ci_gp.layout = layout;

  if (const auto res = vkCreateGraphicsPipelines(
        get_device(), nullptr, 1, &ci_gp, nullptr, &pipeline);
      res != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  rps->pipeline = pipeline;
  rps->layout = layout;

  return pipeline;
}

#pragma region Destroyers

auto
Context::destroy(const TextureHandle handle) -> void
{
  SCOPE_EXIT
  {
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

  const auto* texture = maybe_texture.value();
  if (texture == nullptr) {
    return;
  }

  // Lets destroy the allocation, the image, the view, the storage view, the
  // layer/mip views.
  pre_frame_task([&allocator = get_allocator_implementation(),
                  img = texture](auto device, auto* allocation_callbacks) {
    for (const auto view = img->get_mip_layers_image_views();
         const auto& v : view) {
      if (VK_NULL_HANDLE == v) {
        continue;
      }
      vkDestroyImageView(device, v, allocation_callbacks);
    }

    vkDestroyImageView(device, img->get_image_view(), allocation_callbacks);
    if (const auto storage_view = img->get_storage_image_view())
      vkDestroyImageView(device, storage_view, allocation_callbacks);

    // destroy all framebuffer views
    for (auto&& view : img->get_framebuffer_views()) {
      if (view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, view, allocation_callbacks);
      }
    }
  });

  if (!texture->owns_self()) {
    return;
  }

  pre_frame_task([&alloc = *allocator_impl, tex = texture->get_image()](
                   auto, auto) { alloc.deallocate_image(tex); });
}

auto
Context::destroy(BufferHandle) -> void
{
  TODO("Implement buffer destruction");
}

auto
Context::destroy(QueryPoolHandle) -> void
{
  TODO("Implement query pool destruction");
}

auto
Context::destroy(ComputePipelineHandle) -> void
{
  TODO("Implement compute pipeline destruction");
}

auto
Context::destroy(GraphicsPipelineHandle handle) -> void
{
  if (!handle.valid()) {
    return;
  }

  const auto maybe_pipeline = get_graphics_pipeline_pool().get(handle);
  if (!maybe_pipeline.has_value()) {
    return;
  }

  auto* pipeline = maybe_pipeline.value();
  if (pipeline == nullptr) {
    return;
  }

  pre_frame_task(
    [ptr = pipeline->get_pipeline(),
     layout = pipeline->get_layout()](auto device, auto* allocation_callbacks) {
      vkDestroyPipeline(device, ptr, allocation_callbacks);
      vkDestroyPipelineLayout(device, layout, allocation_callbacks);
    });
}

auto
Context::destroy(const SamplerHandle handle) -> void
{
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

  pre_frame_task([ptr = sampler](auto device, auto* allocation_callbacks) {
    vkDestroySampler(device, ptr, allocation_callbacks);
  });

  if (auto expected = get_sampler_pool().destroy(handle);
      !expected.has_value()) {
    std::cerr << "Failed to destroy sampler: "
              << std::to_underlying(expected.error()) << std::endl;
  }
}

auto
Context::destroy(ShaderModuleHandle handle) -> void
{
  if (!handle.valid()) {
    return;
  }

  const auto maybe_shader = get_shader_module_pool().get(handle);
  if (!maybe_shader.has_value()) {
    return;
  }

  auto shader = *maybe_shader.value();

  for (const auto& module : shader.get_modules()) {
    pre_frame_task(
      [m = module.module](auto device, auto* allocation_callbacks) {
        vkDestroyShaderModule(device, m, allocation_callbacks);
      });
  }
}

#pragma endregion Destroyers

} // namespace VkBindless