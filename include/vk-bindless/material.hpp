#pragma once

#include "vk-bindless/common.hpp"

#include <cstdint>
#include <glm/glm.hpp>

namespace VkBindless {

enum class MaterialFlags : std::uint32_t
{
  None = 0,
  CastShadow = 0x1,
  ReceiveShadow = 0x2,
  Transparent = 0x4,
};
MAKE_BIT_FIELD(MaterialFlags);

struct Material
{
  glm::vec4 albedo_factor{ 0.0F };
  glm::vec4 emissive_factor{ 0.0F };
  float metallic_factor{ 1.0F };
  float roughness_factor{ 1.0F };
  float normal_scale{ 0.0F };
  float ao_strength{ 0.0F };

  // These are all int32 CPU side to indicate missing texture -> which should
  // map to the white texture 0 on GPU side.
  std::int32_t albedo_texture_index{ -1 };
  std::int32_t normal_texture_index{ -1 };
  std::int32_t roughness_texture_index{ -1 };
  std::int32_t metallic_texture_index{ -1 };
  std::int32_t ao_texture_index{ -1 };
  std::int32_t emissive_texture_index{ -1 };
  std::int32_t tbd_texture{ -1 };

  MaterialFlags flags =
    MaterialFlags::CastShadow | MaterialFlags::ReceiveShadow;
};
static_assert(sizeof(Material) % 16 == 0);

struct GPUMaterial
{
  glm::vec4 albedo_factor{ 0.0F };
  glm::vec4 emissive_factor{ 0.0F };
  float metallic_factor{ 1.0F };
  float roughness_factor{ 1.0F };
  float normal_scale{ 0.0F };
  float ao_strength{ 0.0F };

  // These are all int32 CPU side to indicate missing texture -> which should
  // map to the white texture 0 on GPU side.
  std::uint32_t albedo_texture{ 0 };
  std::uint32_t normal_texture{ 0 };
  std::uint32_t roughness_texture{ 0 };
  std::uint32_t metallic_texture{ 0 };
  std::uint32_t ao_texture{ 0 };
  std::uint32_t emissive_texture{ 0 };
  std::uint32_t tbd_texture{ 0 };

  std::uint32_t flags{ 0 };
};

}