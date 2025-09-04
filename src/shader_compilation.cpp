#include "vk-bindless/shader_compilation.hpp"
#include "vk-bindless/expected.hpp"
#include "vk-bindless/scope_exit.hpp"
#include <filesystem>
#include <fstream>
#include <glslang/Include/glslang_c_interface.h>
#include <iostream>

namespace VkBindless {

namespace {
// Context structure to manage include result lifetimes
struct IncludeContext
{
  std::unordered_map<void*, std::unique_ptr<glsl_include_result_t>> results;
  std::unordered_map<void*, std::string> header_names;
  std::unordered_map<void*, std::string> contents;
};

auto
read_file_to_string(const std::filesystem::path& file_path) -> std::string
{
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    return {};
  }

  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

// Callback for system includes: #include <file.glsl>
auto
include_system_callback(void* ctx, const char* header_name, const char*, size_t)
  -> glsl_include_result_t*
{
  auto* context = static_cast<IncludeContext*>(ctx);

  // Resolve system includes from assets/shaders/include/
  std::filesystem::path include_path = "assets/shaders/include";
  std::filesystem::path full_path = include_path / header_name;

  std::string content = read_file_to_string(full_path);
  if (content.empty()) {
    return nullptr; // File not found or empty
  }

  auto result = std::make_unique<glsl_include_result_t>();
  std::string header_name_str(header_name);

  void* result_ptr = result.get();
  context->header_names[result_ptr] = std::move(header_name_str);
  context->contents[result_ptr] = std::move(content);

  result->header_name = context->header_names[result_ptr].c_str();
  result->header_data = context->contents[result_ptr].c_str();
  result->header_length = context->contents[result_ptr].length();

  glsl_include_result_t* result_raw = result.release();
  context->results[result_raw] =
    std::unique_ptr<glsl_include_result_t>(result_raw);

  return result_raw;
}

auto
include_local_callback(void* ctx,
                       const char* header_name,
                       const char* includer_name,
                       size_t include_depth) -> glsl_include_result_t*
{
  // You could modify this to be relative to the includer_name if needed
  return include_system_callback(
    ctx, header_name, includer_name, include_depth);
}

// Callback to free include result
auto
free_include_result_callback(void* ctx, glsl_include_result_t* result)
{
  if (!result || !ctx) {
    return 0;
  }

  auto* context = static_cast<IncludeContext*>(ctx);

  // Clean up the stored strings and result
  context->header_names.erase(result);
  context->contents.erase(result);
  context->results.erase(result);

  return 1; // Success
}
}

auto
parse_shader_stage(std::string_view stage_str)
  -> Expected<ShaderStage, ParseError>
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
  if (stage_str == "task")
    return ShaderStage::task;
  if (stage_str == "mesh")
    return ShaderStage::mesh;

  return unexpected<ParseError>(ParseError::unknown_shader_stage);
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
    case ShaderStage::task:
      return "task";
    case ShaderStage::mesh:
      return "mesh";
  }
  return "unknown";
}

auto
compile_shader(glslang_stage_t stage,
               const std::string& source_code,
               std::vector<std::uint8_t>& output,
               const glslang_resource_t* resources)
  -> Expected<void, std::string>
{
  IncludeContext include_ctx;

  // Setup include callbacks
  static glsl_include_callbacks_t include_callbacks = {
    .include_system = include_system_callback,
    .include_local = include_local_callback,
    .free_include_result = free_include_result_callback
  };

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
    .callbacks = include_callbacks,
    .callbacks_ctx = &include_ctx, // Pass our context
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
    return unexpected<std::string>(error);
  }

  if (!glslang_shader_parse(shader, &input)) {
    std::string error = glslang_shader_get_info_log(shader);
    auto debug_log = glslang_shader_get_info_debug_log(shader);
    std::cout << "Parsing error: " << error << std::endl;
    if (const std::string_view log{ debug_log }; !log.empty()) {
      std::cout << "Debug log: " << debug_log << std::endl;
    }
    return unexpected<std::string>(error);
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
    return unexpected<std::string>(error);
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

  auto assert_does_not_have_version_or_extensions = [](const std::string& src) {
    if (src.find("#version") != std::string::npos) {
      std::cerr << "Shader source already contains a #version directive.\n";
      return false;
    }
    if (src.find("#extension") != std::string::npos) {
      std::cerr << "Shader source already contains an #extension directive.\n";
      return false;
    }
    return true;
  };

  for (auto& entry : parsed.entries) {
    if (!assert_does_not_have_version_or_extensions(entry.source_code)) {
      return false;
    }

    auto& src = entry.source_code;

    auto type = entry.stage;

    if (type == ShaderStage::task || type == ShaderStage::mesh) {
      constexpr auto append =
        R"(
      #version 460
      #extension GL_GOOGLE_include_directive : require
      #extension GL_EXT_buffer_reference : require
      #extension GL_EXT_buffer_reference_uvec2 : require
      #extension GL_EXT_debug_printf : enable
      #extension GL_EXT_nonuniform_qualifier : require
      #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
      #extension GL_EXT_mesh_shader : require

)";
      src = append + src;
    } else if (type == ShaderStage::compute || type == ShaderStage::vertex ||
               type == ShaderStage::tessellation_control ||
               type == ShaderStage::tessellation_evaluation) {
      constexpr auto append =
        R"(
      #version 460
      #extension GL_GOOGLE_include_directive : require
      #extension GL_EXT_buffer_reference : require
      #extension GL_EXT_buffer_reference_uvec2 : require
      #extension GL_EXT_debug_printf : enable
      #extension GL_EXT_nonuniform_qualifier : require
      #extension GL_EXT_samplerless_texture_functions : require
      #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
)";
      src = append + src;
    } else if (type == ShaderStage::fragment) {
      constexpr auto append =
        R"(
      #version 460
      #extension GL_GOOGLE_include_directive : require
      #extension GL_EXT_buffer_reference_uvec2 : require
      #extension GL_EXT_debug_printf : enable
      #extension GL_EXT_nonuniform_qualifier : require
      #extension GL_EXT_samplerless_texture_functions : require
      #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

      layout (set = 0, binding = 0) uniform texture2D textures_2d[];
      layout (set = 1, binding = 0) uniform texture3D textures_3d[];
      layout (set = 2, binding = 0) uniform textureCube texture_cubes[];
      layout (set = 3, binding = 0) uniform texture2D textures_2d_shadow[];
      layout (set = 0, binding = 1) uniform sampler samplers[];
      layout (set = 3, binding = 1) uniform samplerShadow shadow_samplers[];

      layout (set = 0, binding = 3) uniform sampler2D sampler_yuv[];

      vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {
        return texture(nonuniformEXT(sampler2D(textures_2d[textureid], samplers[samplerid])), uv);
      }
      vec4 textureBindless2DLod(uint textureid, uint samplerid, vec2 uv, float lod) {
        return textureLod(nonuniformEXT(sampler2D(textures_2d[textureid], samplers[samplerid])), uv, lod);
      }
      float textureBindless2DShadow(uint textureid, uint samplerid, vec3 uvw) {
        return texture(nonuniformEXT(sampler2DShadow(textures_2d_shadow[textureid], shadow_samplers[samplerid])), uvw);
      }
      ivec2 textureBindlessSize2D(uint textureid) {
        return textureSize(nonuniformEXT(textures_2d[textureid]), 0);
      }
      vec4 textureBindlessCube(uint textureid, uint samplerid, vec3 uvw) {
        return texture(nonuniformEXT(samplerCube(texture_cubes[textureid], samplers[samplerid])), uvw);
      }
      vec4 textureBindlessCubeLod(uint textureid, uint samplerid, vec3 uvw, float lod) {
        return textureLod(nonuniformEXT(samplerCube(texture_cubes[textureid], samplers[samplerid])), uvw, lod);
      }
      int textureBindlessQueryLevels2D(uint textureid) {
        return textureQueryLevels(nonuniformEXT(textures_2d[textureid]));
      }
      int textureBindlessQueryLevelsCube(uint textureid) {
        return textureQueryLevels(nonuniformEXT(texture_cubes[textureid]));
      }
)";

      src = append + src;
    }
  }
  return true;
}

namespace ShaderUtils {
auto
find_stage(const ParsedShader& parsed,
           ShaderStage stage,
           const std::string& entry_name)
  -> Expected<const ShaderEntry*, ParseError>
{
  std::string key = (stage == ShaderStage::compute && !entry_name.empty())
                      ? "compute_" + entry_name
                      : to_string(stage);

  auto it = parsed.stage_lookup.find(key);
  if (it == parsed.stage_lookup.end()) {
    return unexpected<ParseError>(ParseError::missing_stage_content);
  }

  return &parsed.entries[it->second];
}

auto
find_all_compute_stages(const ParsedShader& parsed)
  -> std::vector<const ShaderEntry*>
{
  std::vector<const ShaderEntry*> compute_entries;

  for (const ShaderEntry& entry : parsed.entries) {
    if (entry.stage == ShaderStage::compute) {
      compute_entries.push_back(&entry);
    }
  }

  return compute_entries;
}
}
}
