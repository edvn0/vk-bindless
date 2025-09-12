#pragma stage : vertex

#include <packing.glsl>
#include <ubo.glsl>

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec2 texcoord;
layout(location = 3) in vec4 signed_tangent;

layout(location = 0) out flat uint out_instance_draw_id;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out mat3 frag_tbn;

layout(constant_id = 0) const bool uses_ssbo_transforms = true;

layout(std430, buffer_reference) readonly buffer MaterialRemapSSBO
{
  uint remap[];
};
layout(std430, buffer_reference) readonly buffer SSBO
{
  mat4 transforms[];
};
struct PBRMaterial
{
  vec4 albedo_factor;
  vec4 emissive_factor;
  float metallic_factor;
  float roughness_factor;
  float normal_scale;
  float ao_strength;

  uint albedo_texture_index;
  uint normal_texture_index;
  uint roughness_texture_index;
  uint metallic_texture_index;
  uint ao_texture_index;
  uint emissive_texture_index;

  uint flags;
};
layout(std430, buffer_reference) readonly buffer MaterialSSBO
{
  PBRMaterial materials[];
};

layout(push_constant) uniform PushConstants
{
  mat4 model_transform;
  UBO ubo;
  MaterialSSBO material_ssbo;
  MaterialRemapSSBO remap_ssbo;
  uint sampler_index;
  uint material_index;
};

precise invariant gl_Position;

void
main()
{
  mat4 transform = model_transform;

  mat3 normal_matrix = mat3(transpose(inverse(transform)));

  // Transform to world space
  vec3 world_normal = normalize(normal_matrix * normal.xyz);
  vec3 world_tangent = normalize(normal_matrix * signed_tangent.xyz);

  // Re-orthogonalize tangent with respect to normal (Gram-Schmidt process)
  world_tangent =
    normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);

  // Calculate bitangent using handedness
  vec3 world_bitangent = cross(world_normal, world_tangent) * signed_tangent.w;

  // Construct TBN matrix
  frag_tbn = mat3(world_tangent, world_bitangent, world_normal);

  frag_uv = texcoord;

  vec4 world_pos = transform * vec4(position, 1.0);
  gl_Position = ubo.proj * ubo.view * world_pos;
  out_instance_draw_id = gl_DrawID;
}

#pragma stage : fragment
#include <ubo.glsl>

layout(location = 0) flat in uint in_draw_id;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in mat3 frag_tbn;

layout(location = 0) out vec2 out_uvs;
layout(location = 1) out vec4 out_normal_roughness;
layout(location = 2) out uvec4 out_texture_indices;

layout(std430, buffer_reference) readonly buffer MaterialRemapSSBO
{
  uint remap[];
};
layout(std430, buffer_reference) readonly buffer SSBO
{
  mat4 transforms[];
};
struct PBRMaterial
{
  vec4 albedo_factor;
  vec4 emissive_factor;
  float metallic_factor;
  float roughness_factor;
  float normal_scale;
  float ao_strength;

  uint albedo_texture_index;
  uint normal_texture_index;
  uint roughness_texture_index;
  uint metallic_texture_index;
  uint ao_texture_index;
  uint emissive_texture_index;

  uint flags;
};
layout(std430, buffer_reference) readonly buffer MaterialSSBO
{
  PBRMaterial materials[];
};

layout(push_constant) uniform PushConstants
{
  mat4 model_transform;
  UBO ubo;
  MaterialSSBO material_ssbo;
  MaterialRemapSSBO remap_ssbo;
  uint sampler_index;
  uint material_index;
};

void
main()
{
  out_uvs = vec2(frag_uv);

  vec3 final_normal;

  uint mat_index = remap_ssbo.remap[in_draw_id];
  PBRMaterial material = material_ssbo.materials[mat_index];

  uint normal_texture = material.normal_texture_index;

  if (normal_texture != 0) {
    // Sample normal map
    vec3 normal_map = textureBindless2D(normal_texture, 0, frag_uv).xyz;

    // Transform from tangent space to world space using TBN matrix
    final_normal = normalize(frag_tbn * normal_map);
  } else {
    // Use vertex normal if no normal map
    final_normal = normalize(frag_tbn[2]); // Z component of TBN is the normal
  }
  uint roughness_texture = material.roughness_texture_index;
  float roughness = material.roughness_factor; // Default roughness
  if (roughness_texture != 0) {
    roughness = textureBindless2D(roughness_texture, 1, frag_uv).r;
  }

  out_normal_roughness = vec4(final_normal, roughness);
  out_texture_indices = uvec4(material.albedo_texture_index,
                              normal_texture,
                              roughness_texture,
                              material.metallic_texture_index);
}