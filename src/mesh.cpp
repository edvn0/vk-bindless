#include "vk-bindless/mesh.hpp"
#include "assimp/Importer.hpp"
#include "assimp/material.h"
#include "assimp/vector3.h"
#include "glm/gtc/packing.hpp"
#include "glm/packing.hpp"
#include "meshoptimizer.h"

#include "vk-bindless/buffer.hpp"
#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/common.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/material.hpp"
#include "vk-bindless/texture.hpp"

#include <bit>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <fstream>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <execution>
#include <future>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <ktx.h>
#include <semaphore>
#include <stb_image.h>
#include <type_traits>
#include <vk-bindless/scope_exit.hpp>

namespace VkBindless {

namespace {

struct TextureCache
{
  std::vector<ProcessedTexture> textures;
  std::unordered_map<std::string, std::int32_t> texture_path_to_index;
  std::mutex textures_mutex;

  std::vector<ProcessedTexture> opacity_textures;
  std::unordered_map<std::string, std::int32_t> opacity_path_to_index;
  std::mutex opacity_mutex;
};

class TextureProcessor
{
public:
  static auto process_assimp_texture(
    const aiTexture* ai_texture,
    const std::string& debug_name,
    bool generate_mipmaps = true,
    float quality = 1.0f // 0.0 = fastest, 1.0 = highest quality
    ) -> std::expected<ProcessedTexture, std::string>
  {
    assert(quality > 0.0F && quality <= 1.0F);

    if (!ai_texture) {
      return std::unexpected("Null texture provided");
    }

    std::vector<std::uint8_t> rgba_data;
    int width, height, channels;

    if (ai_texture->mHeight == 0) {
      std::uint8_t* decompressed = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(ai_texture->pcData),
        ai_texture->mWidth,
        &width,
        &height,
        &channels,
        4);

      if (!decompressed) {
        return std::unexpected("STB failed to decompress texture: " +
                               std::string(stbi_failure_reason()));
      }

      const size_t data_size = width * height * 4;
      rgba_data.resize(data_size);
      std::memcpy(rgba_data.data(), decompressed, data_size);
      stbi_image_free(decompressed);

    } else {
      width = ai_texture->mWidth;
      height = ai_texture->mHeight;
      channels = 4;

      rgba_data.reserve(width * height * 4);

      for (int i = 0; i < width * height; ++i) {
        const aiTexel& texel = ai_texture->pcData[i];
        rgba_data.push_back(texel.r);
        rgba_data.push_back(texel.g);
        rgba_data.push_back(texel.b);
        rgba_data.push_back(texel.a);
      }
    }

    return create_bc7_ktx_texture(
      rgba_data, width, height, debug_name, generate_mipmaps, quality);
  }

  static auto process_external_texture(
    const std::filesystem::path& texture_path,
    const std::string& debug_name,
    bool generate_mipmaps = true,
    float quality = 1.0f) -> std::expected<ProcessedTexture, std::string>
  {

    int width, height, channels;
    std::uint8_t* data = stbi_load(texture_path.string().c_str(),
                                   &width,
                                   &height,
                                   &channels,
                                   4); // Force RGBA

    if (!data) {
      const auto formatted = std::format("Failed to load texture {}: {}",
                                         texture_path.string(),
                                         stbi_failure_reason());
      return std::unexpected(formatted);
    }

    std::vector<std::uint8_t> rgba_data(data, data + (width * height * 4));
    stbi_image_free(data);

    return create_bc7_ktx_texture(
      rgba_data, width, height, debug_name, generate_mipmaps, quality);
  }

  static auto save_ktx_to_file(const ProcessedTexture& texture,
                               const std::filesystem::path& output_path) -> bool
  {
    return ktxTexture_WriteToNamedFile(ktxTexture(texture.ktx_texture.get()),
                                       output_path.string().c_str()) ==
           KTX_SUCCESS;
  }

  static auto get_gpu_upload_info(const ProcessedTexture& texture)
    -> std::span<const std::uint8_t>
  {

    auto* data = ktxTexture_GetData(ktxTexture(texture.ktx_texture.get()));
    std::size_t size =
      ktxTexture_GetDataSize(ktxTexture(texture.ktx_texture.get()));
    return std::span{ data, size };
  }

private:
  static auto create_bc7_ktx_texture(const std::vector<std::uint8_t>& rgba_data,
                                     int width,
                                     int height,
                                     const std::string& debug_name,
                                     bool generate_mipmaps,
                                     float)
    -> std::expected<ProcessedTexture, std::string>
  {
    const auto mip_levels =
      generate_mipmaps ? calculate_mip_levels(width, height) : 1u;

    ktxTextureCreateInfo create_info{};
    create_info.vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
    create_info.baseWidth = static_cast<ktx_uint32_t>(width);
    create_info.baseHeight = static_cast<ktx_uint32_t>(height);
    create_info.baseDepth = 1;
    create_info.numDimensions = 2;
    create_info.numLevels = mip_levels;
    create_info.numLayers = 1;
    create_info.numFaces = 1;
    create_info.isArray = KTX_FALSE;
    create_info.generateMipmaps = KTX_FALSE;

    ktxTexture2* texture = nullptr;
    if (auto rc = ktxTexture2_Create(
          &create_info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
        rc != KTX_SUCCESS) {
      return std::unexpected("ktxTexture2_Create failed: " +
                             std::to_string(rc));
    }
    std::unique_ptr<ktxTexture2, ProcessedTexture::Destructor> texture_ptr(
      texture, ProcessedTexture::Destructor{});

    int w = width;
    int h = height;
    for (auto i = 0U; i != mip_levels; ++i) {
      size_t offset = 0;
      ktxTexture_GetImageOffset(ktxTexture(texture), i, 0, 0, &offset);
      stbir_resize_uint8_linear((const unsigned char*)rgba_data.data(),
                                width,
                                height,
                                0,
                                ktxTexture_GetData(ktxTexture(texture)) +
                                  offset,
                                w,
                                h,
                                0,
                                STBIR_RGBA);
      h = h > 1 ? h >> 1 : 1;
      w = w > 1 ? w >> 1 : 1;
    }

    if (auto rc = ktxTexture_SetImageFromMemory(
          ktxTexture(texture), 0, 0, 0, rgba_data.data(), rgba_data.size());
        rc != KTX_SUCCESS) {
      return std::unexpected("ktxTexture_SetImageFromMemory failed: " +
                             std::to_string(rc));
    }

    ktxTexture2_CompressBasis(texture, 255);
    ktxTexture2_TranscodeBasis(texture, KTX_TTF_BC7_RGBA, 0);

    // ensure container reports BC7
    const auto final_vk_format = static_cast<VkFormat>(texture->vkFormat);
    if (final_vk_format != VK_FORMAT_BC7_UNORM_BLOCK &&
        final_vk_format != VK_FORMAT_BC7_SRGB_BLOCK) {
      return std::unexpected("Final vkFormat is not BC7 after transcode.");
    }

    return ProcessedTexture{
      .ktx_texture = std::move(texture_ptr),
      .debug_name = debug_name,
      .width = static_cast<std::uint32_t>(width),
      .height = static_cast<std::uint32_t>(height),
      .mip_levels = mip_levels,
    };
  }

  static auto calculate_mip_levels(int width, int height) -> std::uint32_t
  {
    return static_cast<std::uint32_t>(
      std::floor(std::log2(std::max(width, height))) + 1);
  }
};

template<typename T>
auto
add_unique_texture(std::vector<T>& textures,
                   std::unordered_map<std::string, std::int32_t>& path_map,
                   const std::string& texture_path,
                   auto& mutex,
                   const aiScene* scene) -> std::int32_t
{
  if (texture_path.empty()) {
    return -1;
  }

  auto it = path_map.find(texture_path);
  if (it != path_map.end()) {
    return it->second;
  }

  const aiTexture* ai_texture = nullptr;
  if (texture_path.starts_with("*")) {
    try {
      int texture_index = std::stoi(texture_path.substr(1));
      if (texture_index >= 0 &&
          texture_index < static_cast<int>(scene->mNumTextures)) {
        ai_texture = scene->mTextures[texture_index];
      }
    } catch (const std::exception&) {
      return -1;
    }
  } else {
    std::filesystem::path full_path = texture_path;
    if (!std::filesystem::exists(full_path)) {
      return -1;
    }

    auto processed_result =
      TextureProcessor::process_external_texture(full_path,
                                                 texture_path,
                                                 true, // generate_mipmaps
                                                 1.0f  // quality
      );

    if (!processed_result) {
      return -1;
    }

    std::lock_guard lock(mutex);
    const auto index = static_cast<std::uint32_t>(textures.size());
    textures.emplace_back(std::move(processed_result.value()));
    path_map[texture_path] = index;
    return index;
  }

  if (ai_texture) {
    auto processed_result = TextureProcessor::process_assimp_texture(
      ai_texture, texture_path, true, 1.0F);

    if (!processed_result) {
      return -1;
    }

    std::lock_guard lock(mutex);
    const auto index = static_cast<std::uint32_t>(textures.size());
    textures.emplace_back(std::move(processed_result.value()));
    path_map[texture_path] = index;
    return index;
  }

  return -1;
}

}

namespace {

template<typename T>
auto
read_into(std::istream& stream, T& output) -> bool
{
  static_assert(std::is_trivially_copyable_v<T>,
                "read_into requires trivially copyable T");
  return static_cast<bool>(
    stream.read(std::bit_cast<char*>(&output), sizeof(T)));
}

template<typename T>
bool
read_into(std::istream& stream, std::span<T> out)
{
  static_assert(std::is_trivially_copyable_v<T>);
  return static_cast<bool>(
    stream.read(reinterpret_cast<char*>(out.data()), out.size_bytes()));
}

auto
read_file(const std::filesystem::path& path) -> std::optional<std::ifstream>
{
  std::ifstream stream(path, std::ios::binary | std::ios::in);
  if (!stream)
    return std::nullopt;

  stream.seekg(0, std::ios::end);
  const auto size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  if (size <= 0)
    return std::nullopt;

  return stream;
}

template<typename T>
  requires std::is_trivially_copyable_v<T>
auto
write_to(std::ostream& output, const T& type) -> bool
{
  return static_cast<bool>(
    output.write(reinterpret_cast<const char*>(&type), sizeof(type)));
}

template<typename T>
  requires std::is_trivially_copyable_v<T>
auto
write_to(std::ostream& output, const std::span<T>& out) -> bool
{
  return static_cast<bool>(
    output.write(reinterpret_cast<char*>(out.data()), out.size_bytes()));
}

template<typename T>
  requires std::is_trivially_copyable_v<T>
auto
write_to(std::vector<std::uint8_t>& output, const T& data) -> bool
{
  constexpr auto Size = sizeof(T);
  const auto pos = output.size();
  output.resize(pos + Size);
  std::memcpy(output.data() + pos, reinterpret_cast<const char*>(&data), Size);
  return true;
}

auto
process_lods(const std::vector<std::uint32_t>& source_indices,
             const std::vector<float>& source_vertices,
             std::vector<std::vector<std::uint32_t>>& output_lods) -> void
{
  if (source_indices.empty() || source_vertices.empty()) {
    return;
  }

  const size_t vertex_count = source_vertices.size() / 3;
  const size_t index_count = source_indices.size();

  output_lods.clear();

  std::vector<std::uint32_t> lod0_indices(index_count);
  meshopt_optimizeVertexCache(
    lod0_indices.data(), source_indices.data(), index_count, vertex_count);
  output_lods.push_back(std::move(lod0_indices));

  std::vector<std::uint32_t> current_indices = source_indices;
  const float lod_reduction_rates[] = { 0.75f, 0.5f, 0.25f, 0.1f };
  const float target_errors[] = { 0.01f, 0.05f, 0.1f, 0.2f };

  for (size_t lod_index = 0; lod_index < 4; ++lod_index) {
    const size_t target_index_count = static_cast<size_t>(
      source_indices.size() * lod_reduction_rates[lod_index]);
    const float target_error = target_errors[lod_index];

    if (target_index_count < 6) {
      break;
    }

    // Simplify mesh
    std::vector<std::uint32_t> simplified_indices(current_indices.size());
    const size_t result_count = meshopt_simplify(simplified_indices.data(),
                                                 current_indices.data(),
                                                 current_indices.size(),
                                                 source_vertices.data(),
                                                 vertex_count,
                                                 sizeof(float) * 3, // stride
                                                 target_index_count,
                                                 target_error);

    if (result_count == 0 || result_count >= current_indices.size()) {
      break; // No further simplification possible
    }

    // Resize to actual result count
    simplified_indices.resize(result_count);

    // Optimize vertex cache for this LOD
    std::vector<std::uint32_t> optimized_indices(result_count);
    meshopt_optimizeVertexCache(optimized_indices.data(),
                                simplified_indices.data(),
                                result_count,
                                vertex_count);

    output_lods.push_back(std::move(optimized_indices));
    current_indices = simplified_indices;
  }

  // Ensure we have at least one LOD
  if (output_lods.empty()) {
    output_lods.push_back(source_indices);
  }
}

auto
convert_assimp_mesh_to_mesh(const aiMesh& mesh,
                            MeshData& output,
                            std::uint32_t& index_offset,
                            std::uint32_t& vertex_offset) -> Mesh
{
  const auto has_tex_coords = mesh.HasTextureCoords(0);
  const auto has_tangent_space = mesh.HasTangentsAndBitangents();
  const auto has_normals = mesh.HasNormals();
  std::vector<float> source_vertices;
  std::vector<std::uint32_t> source_indices;
  auto& vertices = output.vertex_data;
  std::vector<std::vector<std::uint32_t>> out_lods;
  auto tex_span = has_tex_coords
                    ? std::span{ mesh.mTextureCoords[0], mesh.mNumVertices }
                    : std::span<aiVector3D>{};
  auto normal_span = has_normals ? std::span{ mesh.mNormals, mesh.mNumVertices }
                                 : std::span<aiVector3D>{};
  auto tangent_span = has_tangent_space
                        ? std::span{ mesh.mTangents, mesh.mNumVertices }
                        : std::span<aiVector3D>{};
  auto bitangent_span = has_tangent_space
                          ? std::span{ mesh.mBitangents, mesh.mNumVertices }
                          : std::span<aiVector3D>{};
  auto position_span = std::span{ mesh.mVertices, mesh.mNumVertices };

  for (auto i = 0U; i < position_span.size(); i++) {
    const aiVector3D& v = position_span[i];
    const aiVector3D& n = normal_span[i];
    const aiVector2D t = tex_span.empty()
                           ? aiVector2D{ 0, 0 }
                           : aiVector2D{ tex_span[i].x, tex_span[i].y };

    source_vertices.push_back(v.x);
    source_vertices.push_back(v.y);
    source_vertices.push_back(v.z);

    const auto tangent =
      tangent_span.empty()
        ? aiVector3D{ 0, 0, 0 }
        : aiVector3D{ tangent_span[i].x, tangent_span[i].y, tangent_span[i].z };

    auto handedness = [](auto invalid,
                         const auto& bitangent,
                         const auto& tangent,
                         const auto& normal) -> float {
      if (invalid) {
        return 1.0f; // Default to right-handed
      }

      // Convert to glm vectors for computation
      const glm::vec3 normal_vec{ normal.x, normal.y, normal.z };
      const glm::vec3 tangent_vec{ tangent.x, tangent.y, tangent.z };
      const glm::vec3 bitangent_vec{ bitangent.x, bitangent.y, bitangent.z };

      // Compute cross product and dot product using glm
      const glm::vec3 cross_product = glm::cross(normal_vec, tangent_vec);
      const float dot = glm::dot(cross_product, bitangent_vec);

      return dot < 0.0f ? -1.0f : 1.0f;
    }(!has_tangent_space || !has_normals,
                                             bitangent_span[i],
                                             tangent_span[i],
                                             normal_span[i]);

    // Serialization should be:
    // Float3
    // Int_2_10_10_10_REV
    // HalfFloat2
    // Int_2_10_10_10_REV
    write_to(vertices, v);
    write_to(vertices,
             glm::packSnorm3x10_1x2(glm::vec4{ n.x, n.y, n.z, 0.0f }));
    write_to(vertices, glm::packHalf2x16(glm::vec2{ t.x, t.y }));
    write_to(vertices,
             glm::packSnorm3x10_1x2(
               glm::vec4{ tangent.x, tangent.y, tangent.z, handedness }));
  }

  VertexInput static_opaque_geometry_vertex_input = VertexInput::create({
    VertexFormat::Float3,             // position
    VertexFormat::Int_2_10_10_10_REV, // normal+roughness
    VertexFormat::HalfFloat2,         // uvs
    VertexFormat::Int_2_10_10_10_REV, // tangent+handedness
  });
  output.vertex_streams = static_opaque_geometry_vertex_input;

  for (unsigned int i = 0; i != mesh.mNumFaces; i++) {
    if (mesh.mFaces[i].mNumIndices != 3)
      continue;
    for (unsigned j = 0; j != mesh.mFaces[i].mNumIndices; j++)
      source_indices.push_back(mesh.mFaces[i].mIndices[j]);
  }

  process_lods(source_indices, source_vertices, out_lods);

  Mesh result{
    .index_offset = index_offset,
    .vertex_offset = vertex_offset,
    .vertex_count = mesh.mNumVertices,
    .material_id = mesh.mMaterialIndex,
  };
  std::uint32_t num_indices = 0;
  for (auto lod_choice = 0ULL; lod_choice < out_lods.size(); lod_choice++) {
    for (auto lod = 0ULL; lod < out_lods[lod_choice].size(); lod++) {
      output.index_data.push_back(out_lods[lod_choice][lod]);
    }
    result.lod_offset[lod_choice] = num_indices;
    num_indices += static_cast<std::uint32_t>(out_lods[lod_choice].size());
  }
  result.lod_offset[out_lods.size()] = num_indices;
  result.lod_count = static_cast<std::uint32_t>(out_lods.size());

  index_offset += num_indices;
  vertex_offset += mesh.mNumVertices;

  return result;
}

auto
convert_assimp_material_to_material(const aiMaterial& material,
                                    TextureCache& texture_cache,
                                    const aiScene* scene) -> Material
{
  Material output{};

  aiColor4D color;
  if (aiGetMaterialColor(&material, AI_MATKEY_COLOR_EMISSIVE, &color) ==
      AI_SUCCESS) {
    output.emissive_factor = { color.r, color.g, color.b, color.a };
    output.emissive_factor.w = glm::clamp(output.emissive_factor.w, 0.0F, 1.0F);
  }

  if (aiGetMaterialColor(&material, AI_MATKEY_COLOR_DIFFUSE, &color)) {
    output.albedo_factor = { color.r, color.g, color.b, color.a };
    output.albedo_factor.w = glm::clamp(output.albedo_factor.w, 0.0F, 1.0F);
  }

  aiString path;
  aiTextureMapping mapping;
  std::uint32_t uv_index = 0;
  float blend = 1.0f;
  aiTextureOp texture_op = aiTextureOp_Add;
  std::array<aiTextureMapMode, 2> texture_map_mode = { aiTextureMapMode_Wrap,
                                                       aiTextureMapMode_Wrap };
  std::uint32_t texture_flags = 0;

  if (aiGetMaterialTexture(&material,
                           aiTextureType_EMISSIVE,
                           0,
                           &path,
                           &mapping,
                           &uv_index,
                           &blend,
                           &texture_op,
                           texture_map_mode.data(),
                           &texture_flags) == AI_SUCCESS) {
    output.emissive_texture_index =
      add_unique_texture(texture_cache.textures,
                         texture_cache.texture_path_to_index,
                         path.C_Str(),
                         texture_cache.textures_mutex,

                         scene);
  }

  if (aiGetMaterialTexture(&material,
                           aiTextureType_DIFFUSE,
                           0,
                           &path,
                           &mapping,
                           &uv_index,
                           &blend,
                           &texture_op,
                           texture_map_mode.data(),
                           &texture_flags) == AI_SUCCESS) {
    output.albedo_texture_index =
      add_unique_texture(texture_cache.textures,
                         texture_cache.texture_path_to_index,
                         path.C_Str(),
                         texture_cache.textures_mutex,

                         scene);
  }

  if (aiGetMaterialTexture(&material,
                           aiTextureType_NORMALS,
                           0,
                           &path,
                           &mapping,
                           &uv_index,
                           &blend,
                           &texture_op,
                           texture_map_mode.data(),
                           &texture_flags) == AI_SUCCESS) {
    output.normal_texture_index =
      add_unique_texture(texture_cache.textures,
                         texture_cache.texture_path_to_index,
                         path.C_Str(),
                         texture_cache.textures_mutex,

                         scene);
  }

  if ((output.normal_texture_index == 0) &&
      (aiGetMaterialTexture(&material,
                            aiTextureType_HEIGHT,
                            0,
                            &path,
                            &mapping,
                            &uv_index,
                            &blend,
                            &texture_op,
                            texture_map_mode.data(),
                            &texture_flags) == AI_SUCCESS)) {
    output.normal_texture_index =
      add_unique_texture(texture_cache.textures,
                         texture_cache.texture_path_to_index,
                         path.C_Str(),
                         texture_cache.textures_mutex,
                         scene);
  }

  return output;
}

auto
recalculate_bounding_boxes(MeshData& output) -> void
{
  const auto stride = output.vertex_streams.compute_vertex_size();

  output.aabbs.clear();
  output.aabbs.reserve(output.meshes.size());

  for (const Mesh& mesh : output.meshes) {
    const auto num_indices = mesh.get_lod_indices_count(0U);

    BoundingBox box;

    for (uint32_t i = 0; i != num_indices; i++) {
      const uint32_t vtxOffset =
        output.index_data[mesh.index_offset + i] + mesh.vertex_offset;
      const float* vf =
        std::bit_cast<const float*>(&output.vertex_data.at(vtxOffset * stride));

      box.expand(glm::make_vec3(vf));
    }

    output.aabbs.emplace_back(std::move(box));
  }
}
}

auto
MeshFile::preload_mesh(const std::filesystem::path& path,
                       const std::filesystem::path& cache_directory) -> bool
{
  if (std::filesystem::is_regular_file(cache_directory / path.filename()))
    return true;

  const std::uint32_t flags =
    aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
    aiProcess_GenSmoothNormals | aiProcess_LimitBoneWeights |
    aiProcess_SplitLargeMeshes | aiProcess_ImproveCacheLocality |
    aiProcess_RemoveRedundantMaterials | aiProcess_FindDegenerates |
    aiProcess_FindInvalidData | aiProcess_GenUVCoords | aiProcess_FlipUVs |
    aiProcess_FlipWindingOrder | aiProcess_CalcTangentSpace |
    aiProcess_GlobalScale;

  Assimp::Importer importer{};
  const aiScene* scene{ nullptr };
  if (scene = importer.ReadFile(path.string().c_str(), flags);
      nullptr == scene) {
    return false;
  }

  MeshFile mesh_file{};
  auto& mesh_data = mesh_file.mesh_data;
  auto& header = mesh_file.header;

  mesh_data.meshes.reserve(scene->mNumMeshes);
  mesh_data.aabbs.reserve(scene->mNumMeshes);

  std::uint32_t index_offset{ 0 };
  std::uint32_t vertex_offset{ 0 };
  for (auto i = 0U; i < scene->mNumMeshes; i++) {
    const auto& ai_mesh = *scene->mMeshes[i];
    mesh_data.meshes.push_back(convert_assimp_mesh_to_mesh(
      ai_mesh, mesh_data, index_offset, vertex_offset));
  }

  auto texture_cache_dir = cache_directory / "textures";
  std::error_code ec;
  if (!std::filesystem::exists(texture_cache_dir)) {
    std::filesystem::create_directories(texture_cache_dir, ec);
  }

  TextureCache texture_cache;
  std::vector<std::future<Material>> futures;
  futures.reserve(scene->mNumMaterials);

  for (auto i = 0U; i < scene->mNumMaterials; i++) {
    const auto& ai_material = *scene->mMaterials[i];
    futures.push_back(std::async(std::launch::async, [&]() {
      return convert_assimp_material_to_material(
        ai_material, texture_cache, scene);
    }));
  }

  for (auto& fut : futures) {
    mesh_data.materials.push_back(fut.get());
  }

  recalculate_bounding_boxes(mesh_data);

  header.mesh_count = static_cast<std::uint32_t>(mesh_data.meshes.size());
  header.index_data_size = std::span(mesh_data.index_data).size_bytes();
  header.vertex_data_size = std::span(mesh_data.vertex_data).size_bytes();
  mesh_data.textures = std::move(texture_cache.textures);
  mesh_data.opacity_textures = std::move(texture_cache.opacity_textures);

  if (!std::filesystem::is_directory(cache_directory)) {
    std::filesystem::create_directory(cache_directory, ec);
    if (ec.value() != 0) {
      return false;
    }
  }

  std::ofstream output_file{ cache_directory / path.filename(),
                             std::ios::out | std::ios::binary };
  if (!output_file) {
    return false;
  }
#define WRITE_MAYBE(x)                                                         \
  if (!write_to(output_file, (x)))                                             \
    return false;

  WRITE_MAYBE(header);
  WRITE_MAYBE(mesh_data.vertex_streams)
  WRITE_MAYBE(std::span{ mesh_data.meshes });
  WRITE_MAYBE(std::span{ mesh_data.aabbs });
  WRITE_MAYBE(std::span{ mesh_data.index_data });
  WRITE_MAYBE(std::span{ mesh_data.vertex_data });

  const auto materials = std::span(mesh_data.materials);
  WRITE_MAYBE(materials.size());
  WRITE_MAYBE(materials.size_bytes());
  WRITE_MAYBE(std::span{ materials });

  const auto textures = std::span(mesh_data.textures);
  WRITE_MAYBE(textures.size());

  for (const auto& tex : textures) {
    std::uint8_t has_tex = tex.ktx_texture ? 1 : 0;
    WRITE_MAYBE(has_tex);

    std::uint64_t name_len = static_cast<std::uint64_t>(tex.debug_name.size());
    WRITE_MAYBE(name_len);
    if (name_len > 0) {
      output_file.write(tex.debug_name.data(),
                        static_cast<std::streamsize>(name_len));
    }

    if (!has_tex) {
      continue; // nothing more for this entry
    }

    WRITE_MAYBE(tex.width);
    WRITE_MAYBE(tex.height);
    WRITE_MAYBE(tex.mip_levels);

    ktx_uint8_t* ktx_buffer = nullptr;
    ktx_size_t ktx_size = 0;
    KTX_error_code wc = ktxTexture_WriteToMemory(
      ktxTexture(tex.ktx_texture.get()), &ktx_buffer, &ktx_size);

    if (wc != KTX_SUCCESS) {

      return false;
    }

    // write container size then container bytes
    WRITE_MAYBE(static_cast<std::uint64_t>(ktx_size));
    output_file.write(reinterpret_cast<const char*>(ktx_buffer),
                      static_cast<std::streamsize>(ktx_size));
  }
  return true;
}

auto
MeshFile::create(IContext&, const std::filesystem::path& path)
  -> std::expected<MeshFile, std::string>
{
  MeshFile mesh_file{};
  auto chosen = std::filesystem::is_regular_file(path)
                  ? path
                  : path.parent_path().parent_path() / path.filename();

  auto file = std::move(*read_file(path));
  if (!file) {
    return std::unexpected("Could not open file.");
  }

  if (!read_into(file, mesh_file.header)) {
    return std::unexpected("Could not write into header");
  }

  if (mesh_file.header.magic_bytes != MeshFileHeader::magic_header) {
    return std::unexpected("Invalid mesh file. Maybe you're trying to decode "
                           "a GLTF(etc) mesh?");
  }

  if (!read_into(file, mesh_file.mesh_data.vertex_streams)) {
    return std::unexpected("Could not write vertex streams");
  }
  mesh_file.mesh_data.meshes.resize(mesh_file.header.mesh_count);
  mesh_file.mesh_data.aabbs.resize(mesh_file.header.mesh_count);
  if (!read_into(file, std::span{ mesh_file.mesh_data.meshes })) {
    return std::unexpected("Could not write meshes");
  }
  if (!read_into(file, std::span{ mesh_file.mesh_data.aabbs })) {
    return std::unexpected("Could not write meshes");
  }
  mesh_file.mesh_data.index_data.resize(mesh_file.header.index_data_size /
                                        sizeof(std::uint32_t));
  mesh_file.mesh_data.vertex_data.resize(mesh_file.header.vertex_data_size);
  if (!read_into(file, std::span{ mesh_file.mesh_data.index_data })) {
    return std::unexpected("Could not read index data");
  }
  if (!read_into(file, std::span{ mesh_file.mesh_data.vertex_data })) {
    return std::unexpected("Could not read vertex data");
  }

  std::size_t num_materials{ 0 };
  std::size_t materials_size{ 0 };
  if (!read_into(file, num_materials)) {
    return std::unexpected("Could not read material count");
  }
  if (!read_into(file, materials_size)) {
    return std::unexpected("Could not read material count");
  }
  mesh_file.mesh_data.materials.resize(num_materials);
  if (!read_into(file, std::span{ mesh_file.mesh_data.materials })) {
    return std::unexpected("Could not read materials data");
  }

  std::size_t num_textures = 0;
  if (!read_into(file, num_textures)) {
    return std::unexpected("Could not read texture count");
  }

  mesh_file.mesh_data.textures.resize(num_textures);

  for (auto& tex : mesh_file.mesh_data.textures) {
    std::uint8_t has_tex = 0;
    if (!read_into(file, has_tex))
      return std::unexpected("Could not read texture presence flag");

    std::uint64_t name_len = 0;
    if (!read_into(file, name_len))
      return std::unexpected("Could not read texture name length");

    tex.debug_name.clear();
    if (name_len > 0) {
      tex.debug_name.resize(name_len);
      file.read(tex.debug_name.data(), static_cast<std::streamsize>(name_len));
      if (!file)
        return std::unexpected("Failed to read texture debug_name bytes");
    }

    if (!has_tex) {
      tex.width = tex.height = tex.mip_levels = 0;
      tex.ktx_texture.reset(nullptr);
      continue;
    }

    if (!read_into(file, tex.width))
      return std::unexpected("Texture width fail");
    if (!read_into(file, tex.height))
      return std::unexpected("Texture height fail");
    if (!read_into(file, tex.mip_levels))
      return std::unexpected("Texture mips fail");

    std::uint64_t data_size = 0;
    if (!read_into(file, data_size))
      return std::unexpected("Texture data size fail");

    if (data_size == 0) {
      // unexpected but handle gracefully
      tex.ktx_texture.reset(nullptr);
      continue;
    }

    std::vector<std::uint8_t> buffer(data_size);
    file.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(data_size));
    if (!file)
      return std::unexpected("Could not read texture binary data");

    ktxTexture2* new_tex = nullptr;

    if (auto rc =
          ktxTexture2_CreateFromMemory(buffer.data(),
                                       buffer.size(),
                                       KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
                                       &new_tex);
        rc != KTX_SUCCESS) {
      return std::unexpected("Failed to create KTX texture from file data");
    }

    tex.ktx_texture.reset(new_tex);
  }
  return mesh_file;
}

auto
VkMesh::get_material_buffer_handle(const IContext& ctx) const -> std::uint64_t
{
  return ctx.get_device_address(*materials);
}

auto
VkMesh::get_material_remap_buffer_handle(const IContext& ctx) const
  -> std::uint64_t
{
  return ctx.get_device_address(*material_remap_buffer);
}

auto
VkMesh::draw(ICommandBuffer& cmd,
             const MeshFile& file,
             const std::span<const std::byte> pc) -> void
{
  cmd.cmd_bind_index_buffer(*index_buffer, IndexFormat::UI32, 0);
  cmd.cmd_bind_vertex_buffer(0, *vertex_buffer, 0);
  cmd.cmd_bind_graphics_pipeline(*pipeline);
  cmd.cmd_bind_depth_state({
    .compare_operation = CompareOp::Greater,
    .is_depth_write_enabled = true,
  });
  cmd.cmd_push_constants(pc);
  cmd.cmd_draw_indexed_indirect(indirect_buffer->get_buffer(),
                                sizeof(uint32_t),
                                file.get_header().mesh_count,
                                0);
}

VkMesh::VkMesh(IContext& context, const MeshFile& mesh_file)
  : index_count(static_cast<std::uint32_t>(
      mesh_file.get_header().index_data_size / sizeof(std::uint32_t)))
{
  const auto& data = mesh_file.get_data();
  const auto& header = mesh_file.get_header();

  index_buffer =
    VkDataBuffer::create(context,
                         {
                           .data = VkBindless::as_bytes(data.index_data),
                           .storage = StorageType::DeviceLocal,
                           .usage = BufferUsageFlags::IndexBuffer,
                           .debug_name = "Mesh IB",
                         });
  vertex_buffer =
    VkDataBuffer::create(context,
                         {
                           .data = VkBindless::as_bytes(data.vertex_data),
                           .storage = StorageType::DeviceLocal,
                           .usage = BufferUsageFlags::VertexBuffer,
                           .debug_name = "Mesh IB",
                         });
  std::vector<std::uint8_t> draw_commands;
  const uint32_t num_commands = header.mesh_count;
  draw_commands.resize(sizeof(VkDrawIndexedIndirectCommand) * num_commands +
                       sizeof(uint32_t));
  std::memcpy(draw_commands.data(), &num_commands, sizeof(num_commands));

  indirect_buffer = std::make_unique<IndirectBuffer>(context, num_commands);
  auto command_array = indirect_buffer->as_span();
  for (auto i = 0U; i < num_commands; i++) {
    const auto& mesh = data.meshes.at(i);
    command_array[i] = VkDrawIndexedIndirectCommand{
      .indexCount = mesh.get_lod_indices_count(0U),
      .instanceCount = 1,
      .firstIndex = mesh.index_offset,
      .vertexOffset = static_cast<int32_t>(mesh.vertex_offset),
      .firstInstance = i,
    };
  }
  indirect_buffer->upload();

  std::vector<uint32_t> material_remap(num_commands);
  for (uint32_t i = 0; i < num_commands; ++i) {
    material_remap[i] = data.meshes[i].material_id;
  }
  material_remap_buffer =
    VkDataBuffer::create(context,
                         {
                           .data = VkBindless::as_bytes(material_remap),
                           .storage = StorageType::DeviceLocal,
                           .usage = BufferUsageFlags::StorageBuffer,
                           .debug_name = "Material Remap Buffer",
                         });

  shader = *VkShader::create(&context, "assets/shaders/opaque_geometry.shader");
  pipeline = VkGraphicsPipeline::create(
    &context,
    GraphicsPipelineDescription{ .vertex_input = data.vertex_streams,
                                 .shader = *shader,
                                 .color = { 
                                  ColourAttachment{
                                      .format = Format::RG_F16, //UVs
                                    },
                                  ColourAttachment{
                                      .format = Format::RGBA_F16, // Normal roughness
                                    },
                                    ColourAttachment{
                                      .format = Format::RGBA_UI16, // texture indices (albedo, normal, roughness, metallic)
                                    },
                                  },
                                  .depth_format = Format::Z_F32,
                                 .cull_mode = CullMode::Back,
                                 .debug_name = "Mesh Pipeline" });
  context.on_shader_changed("assets/shaders/opaque_geometry.shader", *pipeline);

  const auto materials_span = std::span{ mesh_file.get_data().materials };

  std::vector<GPUMaterial> copy;
  {
    const auto& texture_data = mesh_file.get_data().textures;
    const auto& opacity_texture_data = mesh_file.get_data().opacity_textures;

    // Create a mapping from texture indices to VkTexture handles
    std::vector<TextureHandle> texture_handles;
    std::vector<TextureHandle> opacity_texture_handles;

    texture_handles.reserve(texture_data.size());
    opacity_texture_handles.reserve(opacity_texture_data.size());

    // Upload regular textures
    for (const auto& processed_texture : texture_data) {
      if (processed_texture.ktx_texture) {
        auto ptr = processed_texture.ktx_texture.get();
        VkTextureDescription tex_desc{
          .fully_specified_data = ptr,
          .format = Format::BC7_RGBA,
          .extent = { processed_texture.width, processed_texture.height, 1 },
          .usage_flags =
            TextureUsageFlags::Sampled | TextureUsageFlags::TransferDestination,
          .mip_levels = processed_texture.mip_levels,
          .debug_name = processed_texture.debug_name
        };

        auto texture_handle = VkTexture::create(context, tex_desc).release();
        texture_handles.push_back(texture_handle);
      } else {
        // Push invalid handle for missing textures
        texture_handles.push_back(TextureHandle{});
      }
    }

    // Upload opacity textures
    /*for (const auto& processed_texture : opacity_texture_data) {
      if (processed_texture.ktx_texture) {
        auto upload_data =
          TextureProcessor::get_gpu_upload_info(processed_texture);

        TextureDescription tex_desc{ .width = processed_texture.width,
                                     .height = processed_texture.height,
                                     .mip_levels = processed_texture.mip_levels,
                                     .format = Format::BC7_UNORM,
                                     .usage = TextureUsageFlags::Sampled,
                                     .debug_name =
                                       processed_texture.debug_name };

        auto texture_handle = context.create_texture(tex_desc, upload_data);
        opacity_texture_handles.push_back(texture_handle);
      } else {
        opacity_texture_handles.push_back(TextureHandle{});
      }
    }*/

    static constexpr auto to_zero = [](const std::int32_t in) -> std::uint32_t {
      return in > 0 ? in : 0U;
    };
    static constexpr auto mapper = [](const Material& mat) -> GPUMaterial {
      return {
        .albedo_factor = mat.albedo_factor,
        .emissive_factor = mat.emissive_factor,
        .metallic_factor = mat.metallic_factor,
        .roughness_factor = mat.roughness_factor,
        .normal_scale = mat.normal_scale,
        .ao_strength = mat.ao_strength,
        .albedo_texture = to_zero(mat.albedo_texture_index),
        .normal_texture = to_zero(mat.normal_texture_index),
        .roughness_texture = to_zero(mat.roughness_texture_index),
        .metallic_texture = to_zero(mat.metallic_texture_index),
        .ao_texture = to_zero(mat.ao_texture_index),
        .emissive_texture = to_zero(mat.emissive_texture_index),
        .tbd_texture = to_zero(mat.tbd_texture),
        .flags = std::to_underlying(mat.flags),
      };
    };

    // Now assign the texture handles to materials
    for (auto& read_material : std::span(mesh_file.get_data().materials)) {
      copy.push_back(mapper(read_material));
      auto& material = copy.back();

      // Assign textures based on the indices stored in the material
      if (read_material.emissive_texture_index >= 0) {
        material.emissive_texture =
          texture_handles[material.emissive_texture].index();
      }

      if (read_material.albedo_texture_index >= 0) {
        material.albedo_texture =
          texture_handles[material.albedo_texture].index();
      }

      if (read_material.normal_texture_index >= 0) {
        material.normal_texture =
          texture_handles[material.normal_texture].index();
      }
      /*
      if (material.opacity_texture >= 0) {
        material.opacity_texture =
          opacity_texture_handles[material.opacity_texture].index();
      }
      */
    }
  }

  materials = VkDataBuffer::create(context,
                                   {
                                     .data = VkBindless::as_bytes(copy),
                                     .storage = StorageType::DeviceLocal,
                                     .usage = BufferUsageFlags::StorageBuffer,
                                     .debug_name = "Mesh SSBO",
                                   });
}
}