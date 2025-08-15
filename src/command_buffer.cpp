#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/transitions.hpp"
#include "vk-bindless/vulkan_context.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

namespace VkBindless {

namespace {

constexpr auto sample_count_more_than_one = [](VkSampleCountFlagBits sc) {
  switch (sc) {
    case VK_SAMPLE_COUNT_1_BIT:
      return false;
    case VK_SAMPLE_COUNT_2_BIT:
    case VK_SAMPLE_COUNT_4_BIT:
    case VK_SAMPLE_COUNT_8_BIT:
    case VK_SAMPLE_COUNT_16_BIT:
    case VK_SAMPLE_COUNT_32_BIT:
    case VK_SAMPLE_COUNT_64_BIT:
      return true;
    default:
      return false;
  }
};

auto
set_clear_colour(VkClearColorValue& dst, const ClearColourValue& src) -> void
{
  static constexpr auto for_each = [](auto& dst_array, const auto& src_array) {
    for (auto i = 0U; i < std::size(src_array); ++i) {
      dst_array[i] = src_array[i];
    }
  };

  std::visit(
    [&]<typename T>(const T& value) {
      if constexpr (std::is_same_v<T, std::array<float, 4>>) {
        for_each(dst.float32, value);
      } else if constexpr (std::is_same_v<T, std::array<std::uint32_t, 4>>) {
        for_each(dst.uint32, value);
      } else if constexpr (std::is_same_v<T, std::array<std::int32_t, 4>>) {
        for_each(dst.int32, value);
      }
    },
    src);
}

static constexpr auto
resolve_mode_to_vk_resolve_mode_flag_bits(ResolveMode mode,
                                          VkResolveModeFlags supported)
  -> VkResolveModeFlagBits
{
  switch (mode) {
    case ResolveMode::None:
      return VK_RESOLVE_MODE_NONE;
    case ResolveMode::SampleZero:
      return VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    case ResolveMode::Average:
      return (supported & VK_RESOLVE_MODE_AVERAGE_BIT)
               ? VK_RESOLVE_MODE_AVERAGE_BIT
               : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    case ResolveMode::Min:
      return (supported & VK_RESOLVE_MODE_MIN_BIT)
               ? VK_RESOLVE_MODE_MIN_BIT
               : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    case ResolveMode::Max:
      return (supported & VK_RESOLVE_MODE_MAX_BIT)
               ? VK_RESOLVE_MODE_MAX_BIT
               : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
  }
  assert(false);
  return VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
}

static constexpr auto load_op_to_vk_attachment_load_op =
  [](LoadOp op) -> VkAttachmentLoadOp {
  switch (op) {
    case LoadOp::Load:
      return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::Clear:
      return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case LoadOp::DontCare:
      return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    case LoadOp::None:
      return VK_ATTACHMENT_LOAD_OP_NONE;
    default:
      return VK_ATTACHMENT_LOAD_OP_MAX_ENUM;
  }
};

static constexpr auto store_op_to_vk_attachment_store_op =
  [](StoreOp op) -> VkAttachmentStoreOp {
  switch (op) {
    case StoreOp::Store:
      return VK_ATTACHMENT_STORE_OP_STORE;
    case StoreOp::MsaaResolve:
    case StoreOp::DontCare:
      return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    case StoreOp::None:
      return VK_ATTACHMENT_STORE_OP_NONE;
    default:
      return VK_ATTACHMENT_STORE_OP_MAX_ENUM;
  }
};

} // namespace

CommandBuffer::~CommandBuffer()
{
  assert(!is_rendering);
}

CommandBuffer::CommandBuffer(IContext& ctx)
  : context(static_cast<Context*>(&ctx))
  , wrapper(&context->get_immediate_commands().acquire())
{
}

auto
CommandBuffer::cmd_begin_rendering(const RenderPass& render_pass,
                                   const Framebuffer& fb,
                                   const Dependencies& deps) -> void
{
  assert(!is_rendering);

  is_rendering = true;
  view_mask = render_pass.view_mask;

  for (std::uint32_t i = 0;
       i != Dependencies::max_dependencies && deps.textures[i];
       i++) {
    auto* image = *context->get_texture_pool().get(deps.textures[i]);
    Transition::image(wrapper->command_buffer,
                      image->get_image(),
                      VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  const std::uint32_t framebuffer_colour_attachment_count =
    fb.get_colour_attachment_count();
#if !NDEBUG
  const std::uint32_t render_pass_colour_attachment_count =
    render_pass.get_colour_attachment_count();

  assert(render_pass_colour_attachment_count ==
         framebuffer_colour_attachment_count);
#endif

  framebuffer = fb;

  for (std::uint32_t i = 0; i != framebuffer_colour_attachment_count; i++) {
    if (TextureHandle handle = fb.color[i].texture) {
      auto* texture = *context->get_texture_pool().get(handle);
      Transition::image(wrapper->command_buffer,
                        texture->get_image(),
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    if (TextureHandle handle = fb.color[i].resolve_texture) {
      auto* color_resolve_texture = *context->get_texture_pool().get(handle);
      Transition::image(wrapper->command_buffer,
                        color_resolve_texture->get_image(),
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
  }

  TextureHandle depth_texture = fb.depth_stencil.texture;
  if (depth_texture) {
    const auto* depth_image = *context->get_texture_pool().get(depth_texture);
    Transition::depth_image(wrapper->command_buffer,
                            depth_image->get_image(),
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  if (TextureHandle handle = fb.depth_stencil.resolve_texture) {
    const auto* depth_resolve_image = *context->get_texture_pool().get(handle);
    Transition::depth_image(wrapper->command_buffer,
                            depth_resolve_image->get_image(),
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  std::uint32_t mip_level = 0;
  std::uint32_t framebuffer_width = 0;
  std::uint32_t framebuffer_height = 0;

  std::array<VkRenderingAttachmentInfo, max_colour_attachments>
    colour_attachments{};

  for (std::uint32_t i = 0; i != framebuffer_colour_attachment_count; i++) {
    const auto& attachment = fb.color[i];
    assert(!attachment.texture.empty());

    auto* color_texture = *context->get_texture_pool().get(attachment.texture);
    const auto& desc_color = render_pass.color[i];
    if (mip_level && desc_color.level) {
      assert(desc_color.level == mip_level &&
             "All color attachments should have the same mip-level");
    }
    const VkExtent3D dim = color_texture->get_extent();
    if (framebuffer_width) {
      assert(dim.width == framebuffer_width &&
             "All attachments should have the same width");
    }
    if (framebuffer_height) {
      assert(dim.height == framebuffer_height &&
             "All attachments should have the same height");
    }

    mip_level = desc_color.level;
    framebuffer_width = dim.width;
    framebuffer_height = dim.height;
    samples = color_texture->get_sample_count();
    colour_attachments[i] = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = nullptr,
      .imageView = color_texture->get_or_create_framebuffer_view(
        *context, desc_color.level, desc_color.layer),
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolveMode =
        sample_count_more_than_one(samples)
          ? resolve_mode_to_vk_resolve_mode_flag_bits(
              desc_color.resolve_mode, VK_RESOLVE_MODE_FLAG_BITS_MAX_ENUM)
          : VK_RESOLVE_MODE_NONE,
      .resolveImageView = VK_NULL_HANDLE,
      .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .loadOp = load_op_to_vk_attachment_load_op(desc_color.load_op),
      .storeOp = store_op_to_vk_attachment_store_op(desc_color.store_op),
      .clearValue = { .color = { .float32 = { 0, 0, 0, 0 } } },
    };
    set_clear_colour(colour_attachments.at(i).clearValue.color,
                     desc_color.clear_colour);

    if (desc_color.store_op == StoreOp::MsaaResolve) {
      assert(samples > 1);
      assert(!attachment.resolve_texture.empty() &&
             "Framebuffer attachment should contain a resolve texture");
      auto* colour_resolve_texture =
        *context->get_texture_pool().get(attachment.resolve_texture);
      colour_attachments[i].resolveImageView =
        colour_resolve_texture->get_or_create_framebuffer_view(
          *context, desc_color.level, desc_color.layer);
      colour_attachments[i].resolveImageLayout =
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
  }

  VkRenderingAttachmentInfo depth_attachment = {};

  if (fb.depth_stencil.texture) {
    auto* depth_texture_obj =
      *context->get_texture_pool().get(fb.depth_stencil.texture);
    const auto& desc_depth = render_pass.depth;
    assert(
      desc_depth.level == mip_level &&
      "Depth attachment should have the same mip-level as color attachments");
    depth_attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = nullptr,
      .imageView = depth_texture_obj->get_or_create_framebuffer_view(
        *context, desc_depth.level, desc_depth.layer),
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_NONE,
      .resolveImageView = VK_NULL_HANDLE,
      .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .loadOp = load_op_to_vk_attachment_load_op(desc_depth.load_op),
      .storeOp = store_op_to_vk_attachment_store_op(desc_depth.store_op),
      .clearValue = { .depthStencil = { .depth = desc_depth.clear_depth,
                                        .stencil = desc_depth.clear_stencil } },
    };
    if (desc_depth.store_op == StoreOp::MsaaResolve) {
      assert(depth_texture_obj->get_sample_count() == samples);
      const auto& attachment = fb.depth_stencil;
      assert(!attachment.resolve_texture.empty() &&
             "Framebuffer depth attachment should contain a resolve texture");
      auto* depth_resolve_texture =
        *context->get_texture_pool().get(attachment.resolve_texture);
      depth_attachment.resolveImageView =
        depth_resolve_texture->get_or_create_framebuffer_view(
          *context, desc_depth.level, desc_depth.layer);
      depth_attachment.resolveImageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth_attachment.resolveMode = resolve_mode_to_vk_resolve_mode_flag_bits(
        desc_depth.resolve_mode,
        context->vulkan_properties.twelve.supportedDepthResolveModes);
    }
    const auto& dim = depth_texture_obj->get_extent();
    if (framebuffer_width) {
      assert(dim.width == framebuffer_width &&
             "All attachments should have the same width");
    }
    if (framebuffer_height) {
      assert(dim.height == framebuffer_height &&
             "All attachments should have the same height");
    }
    mip_level = desc_depth.level;
    framebuffer_width = dim.width;
    framebuffer_height = dim.height;
  }

  const std::uint32_t width = std::max(framebuffer_width >> mip_level, 1u);
  const std::uint32_t height = std::max(framebuffer_height >> mip_level, 1u);
  const Viewport viewport = {
    0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height),
    1.0f, 0.0f,
  };
  const ScissorRect scissor = { 0, 0, width, height };

  VkRenderingAttachmentInfo stencil_attachment = depth_attachment;
  const bool is_stencil_format = render_pass.stencil.load_op != LoadOp::Invalid;

  const VkRenderingInfo rendering_info = {
    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
    .pNext = nullptr,
    .flags = 0,
    .renderArea = { { static_cast<std::int32_t>(scissor.x),
                                static_cast<std::int32_t>(scissor.y), },
                    { scissor.width, scissor.height, }, },
    .layerCount = render_pass.layer_count,
    .viewMask = view_mask,
    .colorAttachmentCount = framebuffer_colour_attachment_count,
    .pColorAttachments = colour_attachments.data(),
    .pDepthAttachment = depth_texture ? &depth_attachment : nullptr,
    .pStencilAttachment = is_stencil_format ? &stencil_attachment : nullptr,
  };

  const VkViewport vp = {
    .x = viewport.x,
    .y = viewport.height - viewport.y,
    .width = viewport.width,
    .height = -viewport.height,
    .minDepth = viewport.minDepth,
    .maxDepth = viewport.maxDepth,
  };
  vkCmdSetViewport(wrapper->command_buffer, 0, 1, &vp);

  VkRect2D rect = { .offset = { static_cast<std::int32_t>(scissor.x),
                                static_cast<std::int32_t>(scissor.y), },
                    .extent = { scissor.width, scissor.height, }, };
  vkCmdSetScissor(wrapper->command_buffer, 0, 1, &rect);

  context->update_resource_bindings();

  vkCmdSetDepthCompareOp(wrapper->command_buffer, VK_COMPARE_OP_NEVER);
  vkCmdSetDepthBiasEnable(wrapper->command_buffer, VK_FALSE);

  vkCmdBeginRendering(wrapper->command_buffer, &rendering_info);
}

auto
CommandBuffer::cmd_end_rendering() -> void
{
  vkCmdEndRendering(wrapper->command_buffer);
  is_rendering = false;
  framebuffer = {};
}

} // namespace vk_bindless
