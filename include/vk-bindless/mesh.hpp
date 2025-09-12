#pragma once

#include "vk-bindless/buffer.hpp"
#include "vk-bindless/common.hpp"
#include "vk-bindless/container.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/handle.hpp"
#include "vk-bindless/holder.hpp"
#include "vk-bindless/material.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <ktx.h>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace VkBindless {

constexpr std::uint32_t max_lods{ 8 };

using IndexType = std::uint32_t;

struct Mesh final
{
  std::uint32_t lod_count{ 1 };
  std::uint32_t index_offset{ 0 };
  std::uint32_t vertex_offset{ 0 };
  std::uint32_t vertex_count{ 0 };
  std::uint32_t material_id{ 0 };

  std::array<std::uint32_t, max_lods + 1> lod_offset{};

  auto get_lod_indices_count(std::unsigned_integral auto lod_index) const
  {
    return lod_index < lod_count
             ? lod_offset.at(lod_index + 1) - lod_offset.at(lod_index)
             : 0;
  }
};

enum class LoadedTextureType : std::uint8_t
{
  Emissive,
  Diffuse,
  Normals,
  Height,
  Opacity,
};

struct ProcessedTexture
{
  struct Destructor
  {
    auto operator()(ktxTexture2* ptr) const -> void
    {
      ktxTexture_Destroy(ktxTexture(ptr));
    }
  };
  std::unique_ptr<ktxTexture2, Destructor> ktx_texture{ nullptr, Destructor{} };
  std::string debug_name;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t mip_levels;
};

struct MeshData final
{
  VertexInput vertex_streams{};

  std::vector<IndexType> index_data{};
  std::vector<std::uint8_t> vertex_data{};

  std::vector<Mesh> meshes{};
  std::vector<BoundingBox> aabbs{};
  std::vector<Material> materials{};
  std::vector<ProcessedTexture> textures;
  std::vector<ProcessedTexture> opacity_textures;
};

struct MeshFileHeader
{
  static constexpr auto magic_header = 0x46696E65U;

  std::uint32_t magic_bytes = magic_header; // 'Fine' in ASCII.
  std::uint32_t mesh_count{ 0 };
  std::size_t index_data_size{ 0 };
  std::size_t vertex_data_size{ 0 };
};

class MeshFile
{
  MeshFileHeader header;
  MeshData mesh_data;

public:
  [[nodiscard]] auto get_header() const -> const auto& { return header; }
  [[nodiscard]] auto get_data() const -> const auto& { return mesh_data; }
  [[nodiscard]] auto get_data() -> auto& { return mesh_data; }

  static auto create(IContext&, const std::filesystem::path&)
    -> std::expected<MeshFile, std::string>;
  static auto preload_mesh(const std::filesystem::path&,
                           const std::filesystem::path& cache_directory = {
                             "assets/.mesh_cache" }) -> bool;
};

class VkMesh final
{
  using BufferHolder = Holder<BufferHandle>;
  BufferHolder index_buffer;
  BufferHolder vertex_buffer;
  BufferHolder material_remap_buffer;
  std::unique_ptr<IndirectBuffer> indirect_buffer;
  BufferHolder materials;
  Holder<ShaderModuleHandle> shader;
  Holder<GraphicsPipelineHandle> pipeline;

  std::uint32_t index_count{ 0 };

public:
  VkMesh(IContext&, const MeshFile&);
  auto draw(ICommandBuffer&, const MeshFile&, std::span<const std::byte>)
    -> void;
  auto get_material_buffer_handle(const IContext&) const -> std::uint64_t;
  auto get_material_remap_buffer_handle(const IContext&) const -> std::uint64_t;
};

}