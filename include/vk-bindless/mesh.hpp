#pragma once

#include "vk-bindless/expected.hpp"
#include "vk-bindless/handle.hpp"
#include "vk-bindless/holder.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct aiMesh;
struct aiScene;

namespace VkBindless {

struct Vertex
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 texcoord;
};

struct ShadowVertex
{
  glm::vec3 position;
};

struct LodInfo
{
  std::uint32_t index_offset;
  std::uint32_t index_count;
  float target_error;
};

struct MeshData
{
  std::vector<Vertex> vertices;
  std::vector<ShadowVertex> shadow_vertices;
  std::vector<std::uint32_t> indices;
  std::vector<LodInfo> lod_levels;
  std::uint32_t shadow_index_offset;
  std::uint32_t shadow_index_count;
};

class LodGenerator
{
private:
  // PIMPL idiom to hide dependencies
  struct Impl;
  std::unique_ptr<Impl> pimpl;

public:
  LodGenerator();
  ~LodGenerator();

  // Move-only semantics
  LodGenerator(const LodGenerator&) = delete;
  LodGenerator& operator=(const LodGenerator&) = delete;
  LodGenerator(LodGenerator&&) noexcept;
  LodGenerator& operator=(LodGenerator&&) noexcept;

  // Public interface - no assimp/meshopt types exposed
  auto process_mesh_from_data(std::span<const glm::vec3> positions,
                              std::span<const glm::vec3> normals,
                              std::span<const glm::vec2> texcoords,
                              std::span<const std::uint32_t> indices)
    -> MeshData;

  // Load from file - completely hides assimp
  auto load_model(std::string_view filename) -> std::vector<MeshData>;

  // Get the final buffers for GPU upload
  auto get_global_index_buffer() const -> std::span<const std::uint32_t>;
  auto get_global_shadow_index_buffer() const -> std::span<const std::uint32_t>;

  // Clear buffers (call between different models)
  auto clear_buffers() -> void;

  // Get buffer sizes for allocation
  auto get_index_buffer_size_bytes() const -> std::size_t;
  auto get_shadow_index_buffer_size_bytes() const -> std::size_t;

  // Configuration
  struct LodConfig
  {
    std::vector<float> target_errors = { 0.01f, 0.05f, 0.1f, 0.2f };
    float shadow_error_threshold = 0.2f;
    float shadow_reduction_factor = 4.0f;
    float overdraw_threshold = 1.05f;
  };

  auto set_lod_config(const LodConfig& config) -> void;

private:
  // Internal methods that work with assimp (hidden in implementation)
  auto process_assimp_mesh(const aiMesh* ai_mesh) -> MeshData;
};

class Mesh
{
  Holder<BufferHandle> vertex_buffer;
  Holder<BufferHandle> shadow_vertex_buffer;
  Holder<BufferHandle> lod_index_buffer;
  Holder<BufferHandle> lod_shadow_index_buffer;

  MeshData mesh_data{};

public:
  [[nodiscard]] auto get_vertex_buffer() const { return *vertex_buffer; }
  [[nodiscard]] auto get_shadow_vertex_buffer() const
  {
    return *shadow_vertex_buffer;
  }
  [[nodiscard]] auto get_index_buffer() const { return *lod_index_buffer; }
  [[nodiscard]] auto get_shadow_index_buffer() const
  {
    return *lod_shadow_index_buffer;
  }
  [[nodiscard]] auto get_mesh_data() const -> const auto& { return mesh_data; }
  [[nodiscard]] auto get_index_binding_data(std::size_t lod_index) const
  {
    auto& data = mesh_data.lod_levels.at(lod_index);
    return std::make_pair(data.index_count, data.index_offset);
  }

  static auto create(IContext&, std::string_view)
    -> Expected<Mesh, std::string>;
};

}