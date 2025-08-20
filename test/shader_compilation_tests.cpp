#include "doctest/doctest.h"

#include "vk-bindless/shader_compilation.hpp"

using namespace VkBindless;

TEST_CASE("Basic vertex and fragment shader parsing")
{
  std::string shader_source = R"(
#pragma stage : vertex
layout(location = 0) out vec3 frag_color;
void main() {
    gl_Position = vec4(0.0);
}

#pragma stage : fragment
layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec4 out_color;
void main() {
    out_color = vec4(1.0);
}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE(result.has_value());
  REQUIRE(result->entries.size() == 2);

  // Check vertex shader
  auto vertex_stage = ShaderUtils::find_stage(*result, ShaderStage::vertex);
  REQUIRE(vertex_stage.has_value());
  REQUIRE((*vertex_stage)->stage == ShaderStage::vertex);
  REQUIRE((*vertex_stage)->entry_name.empty());
  REQUIRE((*vertex_stage)->source_code.find("gl_Position") !=
          std::string::npos);

  // Check fragment shader
  auto fragment_stage = ShaderUtils::find_stage(*result, ShaderStage::fragment);
  REQUIRE(fragment_stage.has_value());
  REQUIRE((*fragment_stage)->stage == ShaderStage::fragment);
  REQUIRE((*fragment_stage)->entry_name.empty());
  REQUIRE((*fragment_stage)->source_code.find("out_color") !=
          std::string::npos);
}

TEST_CASE("Compute shader with named entry points")
{
  std::string shader_source = R"(
#pragma stage : compute("main_kernel")
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main_kernel() {
    // Main compute work
}

#pragma stage : compute("secondary_kernel")  
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void secondary_kernel() {
    // Secondary compute work
}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE(result.has_value());
  REQUIRE(result->entries.size() == 2);

  // Check main kernel
  auto main_kernel =
    ShaderUtils::find_stage(*result, ShaderStage::compute, "main_kernel");
  REQUIRE(main_kernel.has_value());
  REQUIRE((*main_kernel)->stage == ShaderStage::compute);
  REQUIRE((*main_kernel)->entry_name == "main_kernel");
  REQUIRE((*main_kernel)->source_code.find("Main compute work") !=
          std::string::npos);

  // Check secondary kernel
  auto secondary_kernel =
    ShaderUtils::find_stage(*result, ShaderStage::compute, "secondary_kernel");
  REQUIRE(secondary_kernel.has_value());
  REQUIRE((*secondary_kernel)->stage == ShaderStage::compute);
  REQUIRE((*secondary_kernel)->entry_name == "secondary_kernel");
  REQUIRE((*secondary_kernel)->source_code.find("Secondary compute work") !=
          std::string::npos);

  // Check that we can find all compute stages
  auto all_compute = ShaderUtils::find_all_compute_stages(*result);
  REQUIRE(all_compute.size() == 2);
}

TEST_CASE("Anonymous compute shader")
{
  std::string shader_source = R"(
#pragma stage : compute
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    // Anonymous compute shader
}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE(result.has_value());
  REQUIRE(result->entries.size() == 1);

  auto compute_stage = ShaderUtils::find_stage(*result, ShaderStage::compute);
  REQUIRE(compute_stage.has_value());
  REQUIRE((*compute_stage)->stage == ShaderStage::compute);
  REQUIRE((*compute_stage)->entry_name.empty());
}

TEST_CASE("All shader stage types")
{
  std::string shader_source = R"(
#pragma stage : vertex
void main() { /* vertex */ }

#pragma stage : fragment
void main() { /* fragment */ }

#pragma stage : geometry
void main() { /* geometry */ }

#pragma stage : tessellation_control
void main() { /* tess control */ }

#pragma stage : tessellation_evaluation
void main() { /* tess eval */ }

#pragma stage : compute("kernel")
void kernel() { /* compute */ }
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE(result.has_value());
  REQUIRE(result->entries.size() == 6);

  REQUIRE(ShaderUtils::find_stage(*result, ShaderStage::vertex).has_value());
  REQUIRE(ShaderUtils::find_stage(*result, ShaderStage::fragment).has_value());
  REQUIRE(ShaderUtils::find_stage(*result, ShaderStage::geometry).has_value());
  REQUIRE(ShaderUtils::find_stage(*result, ShaderStage::tessellation_control)
            .has_value());
  REQUIRE(ShaderUtils::find_stage(*result, ShaderStage::tessellation_evaluation)
            .has_value());
  REQUIRE(ShaderUtils::find_stage(*result, ShaderStage::compute, "kernel")
            .has_value());
}

TEST_CASE("Error: Invalid pragma syntax")
{
  std::string shader_source = R"(
#pragma stage vertex
void main() {}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::invalid_pragma_syntax);
}

TEST_CASE("Error: Unknown shader stage")
{
  std::string shader_source = R"(
#pragma stage : unknown_stage
void main() {}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::unknown_shader_stage);
}

TEST_CASE("Error: Duplicate stage entry")
{
  std::string shader_source = R"(
#pragma stage : vertex
void main() { /* first vertex */ }

#pragma stage : vertex
void main() { /* second vertex */ }
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::duplicate_stage_entry);
}

TEST_CASE("Error: Duplicate compute entry with same name")
{
  std::string shader_source = R"(
#pragma stage : compute("main")
void main() { /* first main */ }

#pragma stage : compute("main")
void main() { /* second main */ }
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::duplicate_stage_entry);
}

TEST_CASE("Error: Invalid compute entry name syntax")
{
  std::string shader_source = R"(
#pragma stage : compute(main)
void main() {}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::invalid_compute_entry_name);
}

TEST_CASE("Error: Missing closing quote in compute entry name")
{
  std::string shader_source = R"(
#pragma stage : compute("main
void main() {}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::invalid_compute_entry_name);
}

TEST_CASE("Error: Empty shader source")
{
  std::string shader_source = "";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::missing_stage_content);
}

TEST_CASE("Error: No pragma stages found")
{
  std::string shader_source = R"(
void main() {
    // Some shader code without pragma
}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error() == ParseError::missing_stage_content);
}

TEST_CASE("Whitespace handling and line numbers")
{
  std::string shader_source = R"(
    #pragma stage : vertex    
   
layout(location = 0) out vec3 color;
void main() {
    color = vec3(1.0);
}
   # pragma stage : fragment  
void main() {
    gl_FragColor = vec4(1.0);
}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE(result.has_value());
  REQUIRE(result->entries.size() == 2);

  auto vertex_stage = ShaderUtils::find_stage(*result, ShaderStage::vertex);
  REQUIRE(vertex_stage.has_value());
  REQUIRE((*vertex_stage)->line_number == 2);

  auto fragment_stage = ShaderUtils::find_stage(*result, ShaderStage::fragment);
  REQUIRE(fragment_stage.has_value());
  REQUIRE((*fragment_stage)->line_number == 8);
}

TEST_CASE("Mixed compute entries (named and anonymous)")
{
  std::string shader_source = R"(
#pragma stage : compute
void main() {
    // Anonymous compute
}

#pragma stage : compute("named_kernel")
void named_kernel() {
    // Named compute
}
)";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE(result.has_value());
  REQUIRE(result->entries.size() == 2);

  // Should be able to find anonymous compute
  auto anonymous_compute =
    ShaderUtils::find_stage(*result, ShaderStage::compute);
  REQUIRE(anonymous_compute.has_value());
  REQUIRE((*anonymous_compute)->entry_name.empty());

  // Should be able to find named compute
  auto named_compute =
    ShaderUtils::find_stage(*result, ShaderStage::compute, "named_kernel");
  REQUIRE(named_compute.has_value());
  REQUIRE((*named_compute)->entry_name == "named_kernel");

  // Should find both in all_compute
  auto all_compute = ShaderUtils::find_all_compute_stages(*result);
  REQUIRE(all_compute.size() == 2);
}

TEST_CASE("Error message utility function")
{
  REQUIRE(std::string(ShaderUtils::error_to_string(
            ParseError::invalid_pragma_syntax)) == "Invalid pragma syntax");
  REQUIRE(std::string(ShaderUtils::error_to_string(
            ParseError::unknown_shader_stage)) == "Unknown shader stage");
  REQUIRE(std::string(ShaderUtils::error_to_string(
            ParseError::duplicate_stage_entry)) == "Duplicate stage entry");
  REQUIRE(std::string(ShaderUtils::error_to_string(
            ParseError::missing_stage_content)) == "Missing stage content");
  REQUIRE(std::string(ShaderUtils::error_to_string(
            ParseError::invalid_compute_entry_name)) ==
          "Invalid compute entry name");
}

TEST_CASE("Real-world shader example from provided document")
{
  // Using the actual shader from your document
  std::string shader_source = R"(#pragma stage : vertex

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_tangent;
layout(location = 3) out vec3 frag_bitangent;
layout(location = 4) out vec2 frag_uv;

// Static cube vertices (8 vertices)
const vec3 cube_vertices[8] = vec3[](vec3(-0.5, -0.5, -0.5), // 0
                                     vec3(0.5, -0.5, -0.5),  // 1
                                     vec3(0.5, 0.5, -0.5),   // 2
                                     vec3(-0.5, 0.5, -0.5),  // 3
                                     vec3(-0.5, -0.5, 0.5),  // 4
                                     vec3(0.5, -0.5, 0.5),   // 5
                                     vec3(0.5, 0.5, 0.5),    // 6
                                     vec3(-0.5, 0.5, 0.5)    // 7
);

layout(push_constant) uniform PushConstants
{
  mat4 mvp_matrix;
}
pc;

void
main()
{
  // Get the vertex index from the current triangle
  uint vertex_index = cube_indices[gl_VertexIndex];
  vec3 position = cube_vertices[vertex_index];
  gl_Position = pc.mvp_matrix * vec4(position, 1.0);
}

#pragma stage : fragment

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_tangent;
layout(location = 3) in vec3 frag_bitangent;
layout(location = 4) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

void
main()
{
  mat3 tbn = mat3(
    normalize(frag_tangent), normalize(frag_bitangent), normalize(frag_normal));

  vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
  float ndotl = max(dot(frag_normal, light_dir), 0.0);

  vec3 final_color = frag_color * (0.3 + 0.7 * ndotl);

  out_color = vec4(final_color, 1.0);
})";

  auto result = ShaderParser::parse(shader_source);
  REQUIRE(result.has_value());
  REQUIRE(result->entries.size() == 2);

  auto vertex_stage = ShaderUtils::find_stage(*result, ShaderStage::vertex);
  REQUIRE(vertex_stage.has_value());
  REQUIRE((*vertex_stage)->source_code.find("cube_vertices") !=
          std::string::npos);
  REQUIRE((*vertex_stage)->source_code.find("gl_Position") !=
          std::string::npos);

  auto fragment_stage = ShaderUtils::find_stage(*result, ShaderStage::fragment);
  REQUIRE(fragment_stage.has_value());
  REQUIRE((*fragment_stage)->source_code.find("out_color") !=
          std::string::npos);
  REQUIRE((*fragment_stage)->source_code.find("normalize") !=
          std::string::npos);
}