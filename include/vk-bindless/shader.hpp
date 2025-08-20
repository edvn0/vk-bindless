#pragma once

#include "vk-bindless/forward.hpp"
#include "vk-bindless/handle.hpp"
#include "vk-bindless/holder.hpp"
#include "vk-bindless/shader_compilation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace VkBindless {

class VkShader
{
public:
  struct StageModule
  {
    ShaderStage stage;
    std::string entry_name; // for compute
    VkShaderModule module{ VK_NULL_HANDLE };
  };

  VkShader() = default;
  VkShader(IContext*, std::vector<StageModule>&&);

  static auto create(IContext* context, const std::filesystem::path& path)
    -> Holder<ShaderModuleHandle>;

  auto get_modules() const -> const auto& { return modules; }

private:
  Context* context{ nullptr };
  std::vector<StageModule> modules{};

  static auto compile(IContext* device, const std::filesystem::path& path)
    -> std::expected<VkShader, std::string>;

  void move_from(VkShader&& other)
  {
    context = other.context;
    modules = std::move(other.modules);
    other.context = nullptr;
  }
};

}
