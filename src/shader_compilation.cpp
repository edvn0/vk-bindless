#include "vk-bindless/shader_compilation.hpp"
#include "vk-bindless/scope_exit.hpp"
#include <glslang/Include/glslang_c_interface.h>
#include <iostream>

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

auto
compile_shader(glslang_stage_t stage,
               const std::string& source_code,
               std::vector<std::uint8_t>& output,
               const glslang_resource_t* resources)
  -> std::expected<void, std::string>
{
  const glslang_input_t input = {
    .language = GLSLANG_SOURCE_GLSL,
    .stage = stage,
    .client = GLSLANG_CLIENT_VULKAN,
    .client_version = GLSLANG_TARGET_VULKAN_1_3,
    .target_language = GLSLANG_TARGET_SPV,
    .target_language_version = GLSLANG_TARGET_SPV_1_6,
    .code = source_code.c_str(),
    .default_version = 100,
    .default_profile = GLSLANG_NO_PROFILE,
    .force_default_version_and_profile = false,
    .forward_compatible = false,
    .messages = GLSLANG_MSG_DEFAULT_BIT,
    .resource = resources,
    .callbacks = {},
    .callbacks_ctx = nullptr,
  };

  glslang_shader_t* shader = glslang_shader_create(&input);
  SCOPE_EXIT
  {
    glslang_shader_delete(shader);
  };

  if (!glslang_shader_preprocess(shader, &input)) {
    std::string error = glslang_shader_get_info_log(shader);
    auto debug_log = glslang_shader_get_info_debug_log(shader);
    std::cout << "Preprocessing error: " << error << std::endl;
    if (const std::string_view log{ debug_log }; !log.empty()) {
      std::cout << "Debug log: " << debug_log << std::endl;
    }
    return std::unexpected(error);
  }

  if (!glslang_shader_parse(shader, &input)) {
    std::string error = glslang_shader_get_info_log(shader);
    auto debug_log = glslang_shader_get_info_debug_log(shader);
    std::cout << "Parsing error: " << error << std::endl;
    if (const std::string_view log{ debug_log }; !log.empty()) {
      std::cout << "Debug log: " << debug_log << std::endl;
    }
    return std::unexpected(error);
  }

  glslang_program_t* program = glslang_program_create();
  glslang_program_add_shader(program, shader);
  SCOPE_EXIT
  {
    glslang_program_delete(program);
  };
  if (!glslang_program_link(
        program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
    std::string error = glslang_program_get_info_log(program);
    auto debug_log = glslang_program_get_info_debug_log(program);
    std::cout << "Linking error: " << error << std::endl;
    if (const std::string_view log{ debug_log }; !log.empty()) {
      std::cout << "Debug log: " << debug_log << std::endl;
    }
    return std::unexpected(error);
  }

  glslang_spv_options_t options = {
    .generate_debug_info = true,
    .strip_debug_info = false,
    .disable_optimizer = false,
    .optimize_size = true,
    .disassemble = false,
    .validate = true,
    .emit_nonsemantic_shader_debug_info = false,
    .emit_nonsemantic_shader_debug_source = false,
    .compile_only = false,
    .optimize_allow_expanded_id_bound = false,
  };
  glslang_program_SPIRV_generate_with_options(program, input.stage, &options);

  if (glslang_program_SPIRV_get_messages(program) != nullptr) {
    std::string error = glslang_program_SPIRV_get_messages(program);
    std::cout << "SPIR-V generation error: " << error << std::endl;
  }

  const auto* spirv =
    std::bit_cast<const std::uint8_t*>(glslang_program_SPIRV_get_ptr(program));
  const auto byte_count =
    glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);

  output.resize(byte_count);
  std::copy(spirv, spirv + byte_count, output.begin());
  return {};
}

auto
ShaderParser::prepend_preamble(ParsedShader& parsed) -> bool
{
  static const std::string version_line = "#version 460";
  static const std::string extension_line =
    "#extension GL_GOOGLE_include_directive : enable";

  for (auto& entry : parsed.entries) {
    auto& src = entry.source_code;

    // Ensure #version exists
    if (src.find("#version") == std::string::npos) {
      src = version_line + "\n" + extension_line + "\n" + src;
    } else {
      // Ensure the extension exists (after #version)
      if (src.find(extension_line) == std::string::npos) {
        auto pos = src.find("\n"); // end of first line (#version ...)
        if (pos != std::string::npos) {
          src.insert(pos + 1, extension_line + "\n");
        } else {
          // Degenerate case: only "#version ..." with no newline
          src += "\n" + extension_line + "\n";
        }
      }
    }
  }
  return true;
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
