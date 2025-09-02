#pragma once

#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/holder.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace VkBindless {

struct ComputePipelineDescription
{
  ShaderModuleHandle shader;
  SpecialisationConstantDescription specialisation_constants{};
  std::string entry_point {"main"};
  std::string debug_name{};
};

class VkComputePipeline
{
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkPipelineLayout layout{ VK_NULL_HANDLE };
  VkShaderStageFlags stage_flags{ VK_SHADER_STAGE_COMPUTE_BIT };

  ComputePipelineDescription description{};

  VkDescriptorSetLayout descriptor_set_layout{ VK_NULL_HANDLE };
  VkDescriptorSetLayout last_descriptor_set_layout{ VK_NULL_HANDLE };

  std::unique_ptr<std::byte[]> specialisation_constants_storage{ nullptr };

  friend class CommandBuffer;
  friend class Context;

public:
  [[nodiscard]] auto get_layout() const -> const VkPipelineLayout&
  {
    return layout;
  }
  [[nodiscard]] auto get_stage_flags() const -> const VkShaderStageFlags&
  {
    return stage_flags;
  }
  [[nodiscard]] auto get_pipeline() const -> const VkPipeline&
  {
    return pipeline;
  }

  static auto create(IContext* context, const ComputePipelineDescription& desc)
    -> Holder<ComputePipelineHandle>;
};

struct GraphicsPipelineDescription
{
  Topology topology{ Topology::Triangle };
  VertexInput vertex_input{};
  ShaderModuleHandle shader;
  SpecialisationConstantDescription specialisation_constants{};
  std::array<ColourAttachment, max_colour_attachments> color{};
  Format depth_format = Format::Invalid;
  Format stencil_format = Format::Invalid;
  CullMode cull_mode = CullMode::None;
  WindingMode winding = WindingMode::CCW;
  PolygonMode polygon_mode = PolygonMode::Fill;
  StencilState back_face_stencil = {};
  StencilState front_face_stencil = {};
  std::uint32_t sample_count = 1u;
  std::uint32_t patch_control_points = 0;
  float min_sample_shading = 0.0f;
  std::string debug_name{};

  [[nodiscard]] auto get_colour_attachments_count() const -> std::uint32_t
  {
    const auto result = std::ranges::count_if(color, [](const ColourAttachment& attachment) {
        return attachment.format != Format::Invalid;
      });
    return static_cast<std::uint32_t>(result);
  }
};

class VkGraphicsPipeline
{
  VkPipelineLayout layout{ VK_NULL_HANDLE };
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkShaderStageFlags stage_flags{ VK_SHADER_STAGE_ALL_GRAPHICS };
  GraphicsPipelineDescription description{};
  std::uint32_t binding_count{ 0 };
  std::uint32_t attribute_count{ 0 };
  std::uint32_t view_mask{ 0 };

  std::array<VkVertexInputBindingDescription,
             VertexInput::input_bindings_max_count>
    bindings{};
  std::array<VkVertexInputAttributeDescription,
             VertexInput::vertex_attribute_max_count>
    attributes{};
  VkDescriptorSetLayout descriptor_set_layout{ VK_NULL_HANDLE };
  VkDescriptorSetLayout last_descriptor_set_layout{ VK_NULL_HANDLE };

  std::unique_ptr<std::byte[]> specialisation_constants_storage{ nullptr };

  friend class CommandBuffer;
  friend class Context;

public:
  [[nodiscard]] auto get_layout() const -> const VkPipelineLayout&
  {
    return layout;
  }
  [[nodiscard]] auto get_stage_flags() const -> VkShaderStageFlags;
  [[nodiscard]] auto get_pipeline() const -> const VkPipeline&
  {
    return pipeline;
  }

  static auto create(IContext* context, const GraphicsPipelineDescription& desc)
    -> Holder<GraphicsPipelineHandle>;
};

}