#pragma once

#include "vk-bindless/expected.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/handle.hpp"
#include "vk-bindless/holder.hpp"
#include "vk-bindless/shader_compilation.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace VkBindless {

inline auto
to_vk_stage(const ShaderStage stage) -> VkShaderStageFlagBits
{
  switch (stage) {
    case ShaderStage::vertex:
      return VK_SHADER_STAGE_VERTEX_BIT;
    case ShaderStage::fragment:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
    case ShaderStage::geometry:
      return VK_SHADER_STAGE_GEOMETRY_BIT;
    case ShaderStage::tessellation_control:
      return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case ShaderStage::tessellation_evaluation:
      return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case ShaderStage::compute:
      return VK_SHADER_STAGE_COMPUTE_BIT;
    case ShaderStage::task:
      return VK_SHADER_STAGE_TASK_BIT_NV;
    case ShaderStage::mesh:
      return VK_SHADER_STAGE_MESH_BIT_NV;
  }
  return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM; // fallback
}

class VkShader
{
  struct PushConstantInfo
  {
    std::size_t size{ 0 };
    VkShaderStageFlags stages{ 0 };
  } push_constant_info{};

  struct StageModule
  {
    ShaderStage stage;
    std::string entry_name{ "main" }; // for compute
    VkShaderModule module{ VK_NULL_HANDLE };
  };

public:
  VkShader() = default;
  VkShader(IContext*,
           std::vector<StageModule>&&,
           PushConstantInfo,
           VkShaderStageFlagBits);

  static auto create(IContext* context, const std::filesystem::path& path)
    -> Holder<ShaderModuleHandle>;

  [[nodiscard]] auto get_modules() const -> const auto& { return modules; }

  [[nodiscard]] auto has_stage(ShaderStage stage) const -> bool
  {
    return std::ranges::any_of(
      modules, [stage](const auto& m) { return m.stage == stage; });
  }
  auto populate_stages(std::vector<VkPipelineShaderStageCreateInfo>& stages,
                       const VkSpecializationInfo& info) const -> void
  {
    stages.clear();
    stages.reserve(modules.size());
    for (auto&& [stage, entry_name, module] : modules) {
      stages.push_back(VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = to_vk_stage(stage),
        .module = module,
        .pName = entry_name.c_str(),
        .pSpecializationInfo = &info,
      });
    }
  }
  [[nodiscard]] auto get_push_constant_info() const
    -> std::pair<std::size_t, VkShaderStageFlags>
  {
    return { push_constant_info.size, push_constant_info.stages };
  }
  [[nodiscard]] auto get_shader_stage_flags() const -> VkShaderStageFlagBits
  {
    return flags;
  }

private:
  Context* context{ nullptr };
  std::vector<StageModule> modules{};
  VkShaderStageFlagBits flags{ VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM };

  static auto compile(IContext* device, const std::filesystem::path& path)
    -> Expected<VkShader, std::string>;

  void move_from(VkShader&& other)
  {
    context = other.context;
    modules = std::move(other.modules);
    other.context = nullptr;
  }
};

}
