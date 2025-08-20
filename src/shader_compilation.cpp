#include "vk-bindless/shader_compilation.hpp"

namespace VkBindless {

auto
parse_shader_stage(std::string_view stage_str)
  -> std::expected<ShaderStage, ParseError>
{
  if (stage_str == "vertex")
    return ShaderStage::vertex;
  if (stage_str == "fragment")
    return ShaderStage::fragment;
  if (stage_str == "geometry")
    return ShaderStage::geometry;
  if (stage_str == "tessellation_control")
    return ShaderStage::tessellation_control;
  if (stage_str == "tessellation_evaluation")
    return ShaderStage::tessellation_evaluation;
  if (stage_str == "compute")
    return ShaderStage::compute;

  return std::unexpected(ParseError::unknown_shader_stage);
}

auto
to_string(ShaderStage stage) -> std::string
{
  switch (stage) {
    case ShaderStage::vertex:
      return "vertex";
    case ShaderStage::fragment:
      return "fragment";
    case ShaderStage::geometry:
      return "geometry";
    case ShaderStage::tessellation_control:
      return "tessellation_control";
    case ShaderStage::tessellation_evaluation:
      return "tessellation_evaluation";
    case ShaderStage::compute:
      return "compute";
  }
  return "unknown";
}

namespace ShaderUtils {
auto
find_stage(const ParsedShader& parsed,
           ShaderStage stage,
           const std::string& entry_name)
  -> std::expected<const ShaderEntry*, ParseError>
{
  std::string key = (stage == ShaderStage::compute && !entry_name.empty())
                      ? "compute_" + entry_name
                      : to_string(stage);

  auto it = parsed.stage_lookup.find(key);
  if (it == parsed.stage_lookup.end()) {
    return std::unexpected(ParseError::missing_stage_content);
  }

  return &parsed.entries[it->second];
}

auto
find_all_compute_stages(const ParsedShader& parsed)
  -> std::vector<const ShaderEntry*>
{
  std::vector<const ShaderEntry*> compute_entries;

  for (const auto& entry : parsed.entries) {
    if (entry.stage == ShaderStage::compute) {
      compute_entries.push_back(&entry);
    }
  }

  return compute_entries;
}
}

}
