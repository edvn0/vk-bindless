#include "vk-bindless/mesh.hpp"
#include "vk-bindless/buffer.hpp"
#include "vk-bindless/expected.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>
#include <meshoptimizer.h>
#include <stdexcept>

namespace VkBindless {

struct LodGenerator::Impl
{
  std::vector<uint32_t> global_index_buffer;
  std::vector<uint32_t> global_shadow_index_buffer;
  LodConfig config;

  // Internal methods using assimp/meshopt types
  void extract_vertices_from_assimp(const aiMesh* ai_mesh,
                                    std::vector<Vertex>& vertices,
                                    std::vector<ShadowVertex>& shadow_vertices);

  void extract_indices_from_assimp(const aiMesh* ai_mesh,
                                   std::vector<uint32_t>& indices);

  void generate_lod_chain(std::span<const Vertex> vertices,
                          std::span<const uint32_t> original_indices,
                          MeshData& mesh);

  void optimize_lod_indices(std::span<uint32_t> indices,
                            std::span<const Vertex> vertices);

  void generate_shadow_lods(std::span<const ShadowVertex> shadow_vertices,
                            std::span<const uint32_t> original_indices,
                            MeshData& mesh);

  MeshData create_mesh_from_raw_data(std::span<const glm::vec3> positions,
                                     std::span<const glm::vec3> normals,
                                     std::span<const glm::vec2> texcoords,
                                     std::span<const uint32_t> indices);
};

// Constructor/Destructor
LodGenerator::LodGenerator()
  : pimpl(std::make_unique<Impl>())
{
}
LodGenerator::~LodGenerator() = default;

// Move semantics
LodGenerator::LodGenerator(LodGenerator&&) noexcept = default;
LodGenerator&
LodGenerator::operator=(LodGenerator&&) noexcept = default;

// Public interface implementation
auto
LodGenerator::process_mesh_from_data(std::span<const glm::vec3> positions,
                                     std::span<const glm::vec3> normals,
                                     std::span<const glm::vec2> texcoords,
                                     std::span<const std::uint32_t> indices)
  -> MeshData
{

  return pimpl->create_mesh_from_raw_data(
    positions, normals, texcoords, indices);
}

auto
LodGenerator::load_model(const std::string_view filename)
  -> std::vector<MeshData>
{
  Assimp::Importer importer;
  const aiScene* scene =
    importer.ReadFile(std::string{ filename },
                      aiProcess_Triangulate | aiProcess_GenNormals |
                        aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes);

  if (!scene || !scene->mRootNode ||
      scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
    throw std::runtime_error("Failed to load model.");
  }

  std::vector<MeshData> meshes;
  meshes.reserve(scene->mNumMeshes);

  for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
    meshes.push_back(process_assimp_mesh(scene->mMeshes[i]));
  }

  return meshes;
}

auto
LodGenerator::process_assimp_mesh(const aiMesh* ai_mesh) -> MeshData
{
  MeshData mesh;

  // Extract vertices from assimp mesh
  pimpl->extract_vertices_from_assimp(
    ai_mesh, mesh.vertices, mesh.shadow_vertices);

  // Extract original indices
  std::vector<uint32_t> original_indices;
  pimpl->extract_indices_from_assimp(ai_mesh, original_indices);

  // Generate LODs
  pimpl->generate_lod_chain(mesh.vertices, original_indices, mesh);

  // Generate shadow indices
  pimpl->generate_shadow_lods(mesh.shadow_vertices, original_indices, mesh);

  return mesh;
}

auto
LodGenerator::get_global_index_buffer() const -> std::span<const std::uint32_t>
{
  return pimpl->global_index_buffer;
}

auto
LodGenerator::get_global_shadow_index_buffer() const
  -> std::span<const std::uint32_t>
{
  return pimpl->global_shadow_index_buffer;
}

auto
LodGenerator::clear_buffers() -> void
{
  pimpl->global_index_buffer.clear();
  pimpl->global_shadow_index_buffer.clear();
}

auto
LodGenerator::get_index_buffer_size_bytes() const -> std::size_t
{
  return pimpl->global_index_buffer.size() * sizeof(uint32_t);
}

auto
LodGenerator::get_shadow_index_buffer_size_bytes() const -> std::size_t
{
  return pimpl->global_shadow_index_buffer.size() * sizeof(uint32_t);
}

auto
LodGenerator::set_lod_config(const LodConfig& config) -> void
{
  pimpl->config = config;
}

// Implementation methods (would be in .cpp file)
void
LodGenerator::Impl::extract_vertices_from_assimp(
  const aiMesh* ai_mesh,
  std::vector<Vertex>& vertices,
  std::vector<ShadowVertex>& shadow_vertices)
{
  auto positions =
    std::span<const aiVector3D>(ai_mesh->mVertices, ai_mesh->mNumVertices);
  auto normals =
    ai_mesh->HasNormals()
      ? std::span<const aiVector3D>(ai_mesh->mNormals, ai_mesh->mNumVertices)
      : std::span<const aiVector3D>();
  auto texcoords = ai_mesh->HasTextureCoords(0)
                     ? std::span<const aiVector3D>(ai_mesh->mTextureCoords[0],
                                                   ai_mesh->mNumVertices)
                     : std::span<const aiVector3D>();

  vertices.reserve(ai_mesh->mNumVertices);
  shadow_vertices.reserve(ai_mesh->mNumVertices);

  for (size_t i = 0; i < positions.size(); ++i) {
    Vertex v;
    ShadowVertex sv;

    // Position
    v.position = glm::vec3(positions[i].x, positions[i].y, positions[i].z);
    sv.position = v.position;

    // Normal
    if (!normals.empty()) {
      v.normal = glm::vec3(normals[i].x, normals[i].y, normals[i].z);
    } else {
      v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // Texture coordinates
    if (!texcoords.empty()) {
      v.texcoord = glm::vec2(texcoords[i].x, texcoords[i].y);
    } else {
      v.texcoord = glm::vec2(0.0f);
    }

    vertices.push_back(v);
    shadow_vertices.push_back(sv);
  }
}

void
LodGenerator::Impl::extract_indices_from_assimp(const aiMesh* ai_mesh,
                                                std::vector<uint32_t>& indices)
{
  indices.reserve(ai_mesh->mNumFaces * 3);

  for (unsigned int i = 0; i < ai_mesh->mNumFaces; i++) {
    const aiFace& face = ai_mesh->mFaces[i];
    if (face.mNumIndices == 3) {
      indices.push_back(face.mIndices[0]);
      indices.push_back(face.mIndices[1]);
      indices.push_back(face.mIndices[2]);
    }
  }
}

MeshData
LodGenerator::Impl::create_mesh_from_raw_data(
  std::span<const glm::vec3> positions,
  std::span<const glm::vec3> normals,
  std::span<const glm::vec2> texcoords,
  std::span<const std::uint32_t> indices)
{

  MeshData mesh;

  // Combine raw data into Vertex structures
  size_t vertex_count = positions.size();
  mesh.vertices.reserve(vertex_count);
  mesh.shadow_vertices.reserve(vertex_count);

  for (size_t i = 0; i < vertex_count; i++) {
    Vertex v;
    v.position = positions[i];
    v.normal = (i < normals.size()) ? normals[i] : glm::vec3(0.0f, 1.0f, 0.0f);
    v.texcoord = (i < texcoords.size()) ? texcoords[i] : glm::vec2(0.0f);

    ShadowVertex sv;
    sv.position = v.position;

    mesh.vertices.push_back(v);
    mesh.shadow_vertices.push_back(sv);
  }

  // Generate LODs
  generate_lod_chain(mesh.vertices, indices, mesh);
  generate_shadow_lods(mesh.shadow_vertices, indices, mesh);

  return mesh;
}

void
LodGenerator::Impl::generate_lod_chain(
  std::span<const Vertex> vertices,
  std::span<const uint32_t> original_indices,
  MeshData& mesh)
{

  std::vector<uint32_t> current_indices(original_indices.begin(),
                                        original_indices.end());

  // LOD 0 - Original mesh
  auto lod0_offset = static_cast<std::uint32_t>(global_index_buffer.size());
  global_index_buffer.insert(
    global_index_buffer.end(), current_indices.begin(), current_indices.end());

  mesh.lod_levels.push_back(
    { lod0_offset, static_cast<uint32_t>(current_indices.size()), 0.0f });

  // Generate additional LOD levels
  for (size_t lod = 0; lod < config.target_errors.size(); lod++) {
    size_t target_index_count = current_indices.size() / 2;
    if (target_index_count < 3)
      break;

    std::vector<uint32_t> lod_indices(current_indices.size());

    size_t result_count =
      meshopt_simplify(lod_indices.data(),
                       current_indices.data(),
                       current_indices.size(),
                       reinterpret_cast<const float*>(vertices.data()),
                       vertices.size(),
                       sizeof(Vertex),
                       target_index_count,
                       config.target_errors[lod]);
    if (result_count == 0 || result_count >= current_indices.size()) {
      break;
    }

    lod_indices.resize(result_count);
    optimize_lod_indices(lod_indices, vertices);

    auto offset = static_cast<std::uint32_t>(global_index_buffer.size());
    global_index_buffer.insert(
      global_index_buffer.end(), lod_indices.begin(), lod_indices.end());

    mesh.lod_levels.push_back({ offset,
                                static_cast<uint32_t>(result_count),
                                config.target_errors[lod] });

    current_indices = std::move(lod_indices);
  }
}

void
LodGenerator::Impl::optimize_lod_indices(std::span<uint32_t> indices,
                                         std::span<const Vertex> vertices)
{
  meshopt_optimizeVertexCache(
    indices.data(), indices.data(), indices.size(), vertices.size());

  meshopt_optimizeOverdraw(indices.data(),
                           indices.data(),
                           indices.size(),
                           reinterpret_cast<const float*>(vertices.data()),
                           vertices.size(),
                           sizeof(Vertex),
                           config.overdraw_threshold);
}

void
LodGenerator::Impl::generate_shadow_lods(
  std::span<const ShadowVertex> shadow_vertices,
  std::span<const uint32_t> original_indices,
  MeshData& mesh)
{

  size_t target_count = original_indices.size() /
                        static_cast<size_t>(config.shadow_reduction_factor);
  target_count = std::max(target_count, size_t(3));

  std::vector<uint32_t> shadow_indices(original_indices.size());

  auto result_count =
    meshopt_simplify(shadow_indices.data(),
                     original_indices.data(),
                     original_indices.size(),
                     reinterpret_cast<const float*>(shadow_vertices.data()),
                     shadow_vertices.size(),
                     sizeof(ShadowVertex),
                     target_count,
                     config.shadow_error_threshold);

  if (result_count > 0) {
    shadow_indices.resize(result_count);

    meshopt_optimizeVertexCache(shadow_indices.data(),
                                shadow_indices.data(),
                                result_count,
                                shadow_vertices.size());

    mesh.shadow_index_offset =
      static_cast<std::uint32_t>(global_shadow_index_buffer.size());
    mesh.shadow_index_count = static_cast<std::uint32_t>(result_count);

    global_shadow_index_buffer.insert(global_shadow_index_buffer.end(),
                                      shadow_indices.begin(),
                                      shadow_indices.end());
  } else {
    mesh.shadow_index_offset =
      static_cast<std::uint32_t>(global_shadow_index_buffer.size());
    mesh.shadow_index_count =
      static_cast<std::uint32_t>(original_indices.size());

    global_shadow_index_buffer.insert(global_shadow_index_buffer.end(),
                                      original_indices.begin(),
                                      original_indices.end());
  }
}

static constexpr auto
in(const auto& lhs, auto&&... rhs)
{
  return ((lhs == rhs) || ...);
}

auto
Mesh::create(IContext& context, const std::string_view path_view)
  -> Expected<Mesh, std::string>
{
  auto path = std::filesystem::path{ path_view };

  if (!std::filesystem::exists(path))
    return unexpected<std::string>{ "File does not exist." };
  if (!in(path.extension(), ".obj", ".gltf", ".glb"))
    return unexpected<std::string>{ "Not obj or gltf." };

  LodGenerator generator;
  auto meshes = generator.load_model(path.string());

  if (meshes.empty())
    return unexpected<std::string>{ "No meshes found." };
  if (meshes.size() != 1)
    return unexpected<std::string>{
      "Only single-mesh assets supported for now."
    };

  LodGenerator::LodConfig lod_config{};
  lod_config.target_errors = { 0.005f, 0.02f, 0.08f, 0.15f, 0.3f };
  lod_config.shadow_reduction_factor = 6.0f;
  generator.set_lod_config(lod_config);

  auto index_buffer_data = generator.get_global_index_buffer();
  auto shadow_index_buffer_data = generator.get_global_shadow_index_buffer();

  auto& loaded_mesh = meshes.at(0);

  Mesh mesh{};

  mesh.mesh_data = loaded_mesh;

  // Vertex buffer
  mesh.vertex_buffer = VkDataBuffer::create(
    context,
    BufferDescription{
      .data = std::as_bytes(std::span(loaded_mesh.vertices)),
      .size = loaded_mesh.vertices.size() * sizeof(Vertex),
      .storage = StorageType::DeviceLocal,
      .usage = BufferUsageFlags::VertexBuffer,
      .debug_name = std::format("{}_VB", path.filename().string()),
    });

  // Shadow vertex buffer
  mesh.shadow_vertex_buffer = VkDataBuffer::create(
    context,
    BufferDescription{
      .data = std::as_bytes(std::span(loaded_mesh.shadow_vertices)),
      .size = loaded_mesh.shadow_vertices.size() * sizeof(ShadowVertex),
      .storage = StorageType::DeviceLocal,
      .usage = BufferUsageFlags::VertexBuffer,
      .debug_name = std::format("{}_SVB", path.filename().string()),
    });

  // LOD index buffer
  mesh.lod_index_buffer = VkDataBuffer::create(
    context,
    BufferDescription{
      .data = std::as_bytes(index_buffer_data),
      .size = index_buffer_data.size() * sizeof(std::uint32_t),
      .storage = StorageType::DeviceLocal,
      .usage = BufferUsageFlags::IndexBuffer,
      .debug_name = std::format("{}_IB", path.filename().string()),
    });

  // Shadow LOD index buffer
  mesh.lod_shadow_index_buffer = VkDataBuffer::create(
    context,
    BufferDescription{
      .data = std::as_bytes(shadow_index_buffer_data),
      .size = shadow_index_buffer_data.size() * sizeof(std::uint32_t),
      .storage = StorageType::DeviceLocal,
      .usage = BufferUsageFlags::IndexBuffer,
      .debug_name = std::format("{}_SIB", path.filename().string()),
    });

  return mesh;
}

}