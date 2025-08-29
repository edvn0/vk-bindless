#include "vk-bindless/mesh.hpp"
#include "vk-bindless/buffer.hpp"
#include "vk-bindless/expected.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>
#include <iostream>
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
  const aiScene* scene = importer.ReadFile(
    std::string{ filename },
    aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
      aiProcess_GenNormals | aiProcess_CalcTangentSpace |
      aiProcess_ImproveCacheLocality | aiProcess_OptimizeMeshes |
      aiProcess_PreTransformVertices);

  if (!scene || !scene->mRootNode ||
      scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
    throw std::runtime_error("Failed to load model.");
  }

  auto meta_data = scene->mMetaData;
  auto keys = std::span{ meta_data->mKeys, meta_data->mNumProperties };
  auto values = std::span{ meta_data->mValues, meta_data->mNumProperties };
  for (std::size_t i = 0; i < keys.size(); ++i) {
    auto to_string = [](const aiString& str) {
      return std::string_view{ str.C_Str(), str.length };
    };
    std::cout << "  " << to_string(keys[i]) << " : " << values[i].mType << "\n";
  }

  std::vector<MeshData> meshes;
  meshes.reserve(scene->mNumMeshes);

  for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
    meshes.push_back(process_assimp_mesh(scene, scene->mMeshes[i]));
  }

  return meshes;
}

auto
LodGenerator::process_assimp_mesh(const aiScene* scene, const aiMesh* ai_mesh)
  -> MeshData
{
  MeshData mesh;

  // Extract vertices from assimp mesh
  pimpl->extract_vertices_from_assimp(
    ai_mesh, mesh.vertices, mesh.shadow_vertices);

  // Extract original indices
  std::vector<uint32_t> original_indices;
  pimpl->extract_indices_from_assimp(ai_mesh, original_indices);

  mesh.indices = original_indices;

  // Generate LODs
  pimpl->generate_lod_chain(mesh.vertices, original_indices, mesh);

  // Generate shadow indices
  pimpl->generate_shadow_lods(mesh.shadow_vertices, original_indices, mesh);

  if (ai_mesh->mMaterialIndex >= 0) {
    if (mesh.material.size() < ai_mesh->mMaterialIndex) {
      mesh.material.resize(ai_mesh->mMaterialIndex + 1);
    }

    aiMaterial* ai_mat = scene->mMaterials[ai_mesh->mMaterialIndex];

    constexpr auto base_dir = "assets/meshes";

    aiString tex_path;
    if (ai_mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_path) == AI_SUCCESS) {
      // Convert Assimp path to your engine’s texture loader
      std::filesystem::path full_path =
        std::filesystem::path{ base_dir } / tex_path.C_Str();

      // TODO: replace with your actual VkTexture::create wrapper
      auto tex_handle = VkTexture::create(
        context,
        VkTextureDescription{
          .data = load_file_binary(full_path), // You’ll need a file loader here
          .format = vk_format_to_format(VK_FORMAT_R8G8B8A8_UNORM),
          .usage_flags = TextureUsageFlags::Sampled,
          .debug_name = full_path.filename().string() });

      mesh.material.at(ai_mesh->mMaterialIndex).albedo_texture =
        tex_handle.index(); // store bindless handle
    } else {
      mesh.material.at(ai_mesh->mMaterialIndex).albedo_texture =
        Material::white_texture; // fallback
    }
  }

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

// Implementation methods
void
LodGenerator::Impl::extract_vertices_from_assimp(
  const aiMesh* ai_mesh,
  std::vector<Vertex>& vertices,
  std::vector<ShadowVertex>& shadow_vertices)
{
  const auto positions =
    std::span<const aiVector3D>(ai_mesh->mVertices, ai_mesh->mNumVertices);
  const auto normals =
    ai_mesh->HasNormals()
      ? std::span<const aiVector3D>(ai_mesh->mNormals, ai_mesh->mNumVertices)
      : std::span<const aiVector3D>();
  const auto texcoords =
    ai_mesh->HasTextureCoords(0)
      ? std::span<const aiVector3D>(ai_mesh->mTextureCoords[0],
                                    ai_mesh->mNumVertices)
      : std::span<const aiVector3D>();

  vertices.reserve(ai_mesh->mNumVertices);

  for (std::size_t i = 0; i < positions.size(); ++i) {
    Vertex v{};

    // Position
    v.position = glm::vec3(positions[i].x, positions[i].y, positions[i].z);

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
  }

  shadow_vertices = vertices |
                    std::views::transform([](const Vertex& v) -> ShadowVertex {
                      return { v.position };
                    }) |
                    std::ranges::to<std::vector<ShadowVertex>>();
}

void
LodGenerator::Impl::extract_indices_from_assimp(const aiMesh* ai_mesh,
                                                std::vector<uint32_t>& indices)
{
  indices.reserve(ai_mesh->mNumFaces * 3);

  for (unsigned int i = 0; i < ai_mesh->mNumFaces; i++) {
    if (const aiFace& face = ai_mesh->mFaces[i]; face.mNumIndices == 3) {
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

  mesh.indices.assign(indices.begin(), indices.end());

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
  const auto lod0_offset =
    static_cast<std::uint32_t>(global_index_buffer.size());
  global_index_buffer.insert(
    global_index_buffer.end(), current_indices.begin(), current_indices.end());

  mesh.lod_levels.push_back(
    { lod0_offset, static_cast<uint32_t>(current_indices.size()), 0.0f });

  // Ensure we have target errors configured
  if (config.target_errors.empty()) {
    config.target_errors = { 0.01f, 0.05f, 0.1f, 0.2f };
  }

  // Generate additional LOD levels
  for (const auto& target_error : config.target_errors) {
    auto target_index_count = static_cast<std::size_t>(
      static_cast<float>(current_indices.size()) * 0.7f);
    if (target_index_count < 12) // Ensure at least 4 triangles minimum
      break;

    // Ensure target count is multiple of 3 for triangles
    target_index_count = (target_index_count / 3) * 3;

    std::vector<uint32_t> lod_indices(current_indices.size());

    size_t result_count =
      meshopt_simplify(lod_indices.data(),
                       current_indices.data(),
                       current_indices.size(),
                       reinterpret_cast<const float*>(vertices.data()),
                       vertices.size(),
                       sizeof(Vertex),
                       target_index_count,
                       target_error);

    // Better validation of result
    if (result_count == 0 || result_count >= current_indices.size() ||
        result_count < 3) {
      break;
    }

    // Ensure result is multiple of 3
    result_count = (result_count / 3) * 3;
    if (result_count < 3) {
      break;
    }

    lod_indices.resize(result_count);
    optimize_lod_indices(lod_indices, vertices);

    const auto offset = static_cast<std::uint32_t>(global_index_buffer.size());
    global_index_buffer.insert(
      global_index_buffer.end(), lod_indices.begin(), lod_indices.end());

    mesh.lod_levels.emplace_back(
      offset, static_cast<uint32_t>(result_count), target_error);

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
  std::vector<uint32_t> current_indices(original_indices.begin(),
                                        original_indices.end());

  // Shadow LOD 0 - Original mesh (for shadows)
  const auto lod0_offset =
    static_cast<std::uint32_t>(global_shadow_index_buffer.size());
  global_shadow_index_buffer.insert(global_shadow_index_buffer.end(),
                                    current_indices.begin(),
                                    current_indices.end());

  mesh.shadow_lod_levels.push_back(
    { lod0_offset, static_cast<uint32_t>(current_indices.size()), 0.0f });

  // Ensure we have shadow target errors configured
  if (config.shadow_target_errors.empty()) {
    config.shadow_target_errors = { 0.1f, 0.3f, 0.6f };
  }

  // Generate additional shadow LOD levels (more aggressive simplification than
  // main geometry)
  for (const auto& target_error : config.shadow_target_errors) {
    auto target_index_count =
      static_cast<std::size_t>(static_cast<float>(current_indices.size()) /
                               config.shadow_reduction_factor);

    if (target_index_count < 12) // Ensure at least 4 triangles minimum
      break;

    // Ensure target count is multiple of 3 for triangles
    target_index_count = (target_index_count / 3) * 3;

    std::vector<uint32_t> shadow_indices(current_indices.size());

    auto result_count =
      meshopt_simplify(shadow_indices.data(),
                       current_indices.data(),
                       current_indices.size(),
                       reinterpret_cast<const float*>(shadow_vertices.data()),
                       shadow_vertices.size(),
                       sizeof(ShadowVertex),
                       target_index_count,
                       target_error);

    // Validate result
    if (result_count == 0 || result_count >= current_indices.size() ||
        result_count < 3) {
      break;
    }

    // Ensure result is multiple of 3
    result_count = (result_count / 3) * 3;
    if (result_count < 3) {
      break;
    }

    shadow_indices.resize(result_count);

    // Optimize shadow indices (simpler than main geometry - only vertex cache)
    meshopt_optimizeVertexCache(shadow_indices.data(),
                                shadow_indices.data(),
                                result_count,
                                shadow_vertices.size());

    const auto offset =
      static_cast<std::uint32_t>(global_shadow_index_buffer.size());
    global_shadow_index_buffer.insert(global_shadow_index_buffer.end(),
                                      shadow_indices.begin(),
                                      shadow_indices.end());

    mesh.shadow_lod_levels.emplace_back(
      offset, static_cast<uint32_t>(result_count), target_error);

    current_indices = std::move(shadow_indices);
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

  using namespace std::string_view_literals;
  if (!std::filesystem::exists(path))
    return unexpected<std::string>{ "File does not exist." };
  if (!in(path.extension(), ".obj"sv, ".gltf"sv, ".glb"sv))
    return unexpected<std::string>{ "Not obj or gltf." };

  // Configure LOD settings BEFORE loading
  LodGenerator generator;
  LodGenerator::LodConfig lod_config{};
  lod_config.target_errors = { 0.005f, 0.02f, 0.08f, 0.15f,
                               0.3f,   0.5f,  0.7f,  0.9f };
  lod_config.shadow_target_errors = {
    0.1f, 0.3f, 0.6f, 0.9f
  }; // More aggressive for shadows
  lod_config.shadow_reduction_factor = 6.0f;
  lod_config.overdraw_threshold = 1.05f;
  lod_config.shadow_error_threshold = 0.2f;
  generator.set_lod_config(lod_config);

  auto meshes = generator.load_model(path.string());
  if (meshes.empty())
    return unexpected<std::string>{ "No meshes found." };
  if (meshes.size() != 1)
    return unexpected<std::string>{
      "Only single-mesh assets supported for now."
    };

  auto index_buffer_data = generator.get_global_index_buffer();
  auto shadow_index_buffer_data = generator.get_global_shadow_index_buffer();

  auto& loaded_mesh = meshes.at(0);

  Mesh mesh{};

  mesh.mesh_data = loaded_mesh;

  if (loaded_mesh.vertices.empty()) {
    return unexpected<std::string>{ "No vertices in mesh." };
  }
  if (index_buffer_data.empty()) {
    return unexpected<std::string>{ "No indices generated." };
  }
  if (loaded_mesh.lod_levels.empty()) {
    return unexpected<std::string>{ "No LOD levels generated." };
  }
  if (loaded_mesh.shadow_lod_levels.empty()) {
    return unexpected<std::string>{ "No shadow LOD levels generated." };
  }

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