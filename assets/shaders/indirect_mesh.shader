#pragma stage : vertex

#include <packing.glsl>
#include <ubo.glsl>

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 signed_tangent;

layout(location = 0) out flat uint out_instance_index;
layout(location = 1) out vec2 uv;
layout(location = 2) out mat3 frag_tbn;

layout(constant_id = 0) const bool uses_ssbo_transforms = true;

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

  uint _padding[2];
};

layout(std430, buffer_reference) readonly buffer MaterialSSBO
{
  PBRMaterial materials[];
};

layout(push_constant) uniform PushConstants
{
  UBO ubo;
  SSBO ssbo;
  MaterialSSBO material_ssbo;
  mat4 model_transform;
  uint sampler_index;
  uint material_index;
};

precise invariant gl_Position;

void
main()
{
  mat4 transform =
    uses_ssbo_transforms ? ssbo.transforms[gl_InstanceIndex] : model_transform;

  mat3 normal_matrix = mat3(transpose(inverse(transform)));

  // Transform to world space
  vec3 world_normal = normalize(normal_matrix * in_normal.xyz);
  vec3 world_tangent = normalize(normal_matrix * signed_tangent.xyz);

  // Re-orthogonalize tangent with respect to normal (Gram-Schmidt process)
  world_tangent =
    normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);

  // Calculate bitangent using handedness
  vec3 world_bitangent = cross(world_normal, world_tangent) * signed_tangent.w;

  // Construct TBN matrix
  frag_tbn = mat3(world_tangent, world_bitangent, world_normal);

  uv = in_texcoord;

  vec4 world_pos = transform * vec4(in_position, 1.0);
  gl_Position = ubo.proj * ubo.view * world_pos;
  out_instance_index = gl_InstanceIndex;
}

// Geometry shader removed - not needed for basic rendering

#pragma stage : fragment
#include <ubo.glsl>

layout(location = 0) in vec2 uvs;
layout(location = 1) in vec3 barycoords;
layout(location = 2) in mat3 tbn_matrix;
layout(location = 5) flat in uint instance_index;

layout(location = 0) out vec4 out_albedo_metallic;
layout(location = 1) out vec4 out_normal_roughness;
layout(location = 2) out vec4 out_emissive_ao;
layout(location = 3) out vec4 out_wireframe;

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

  uint _padding[2];
};

layout(std430, buffer_reference) readonly buffer MaterialSSBO
{
  PBRMaterial materials[];
};

layout(push_constant) uniform PushConstants
{
  UBO ubo;
  SSBO ssbo;
  MaterialSSBO material_ssbo;
  mat4 model_transform;
  uint sampler_index;
  uint material_index;
};

float
edge_factor(float thickness)
{
  vec3 a3 = smoothstep(vec3(0.0), fwidth(barycoords) * thickness, barycoords);
  return min(min(a3.x, a3.y), a3.z);
}

void
main()
{
  PBRMaterial material = material_ssbo.materials[material_index];

  // Sample albedo
  vec4 albedo = material.albedo_factor;
  if (material.albedo_texture_index != 0) {
    albedo *=
      textureBindless2D(material.albedo_texture_index, sampler_index, uvs);
  }

  // Sample and calculate normal
  vec3 final_normal;
  if (material.normal_texture_index != 0) {
    vec3 normal_map =
      textureBindless2D(material.normal_texture_index, sampler_index, uvs).xyz;
    normal_map = normal_map * 2.0 - 1.0;
    normal_map.xy *= material.normal_scale;
    final_normal = normalize(tbn_matrix * normal_map);
  } else {
    final_normal = normalize(tbn_matrix[2]); // Use vertex normal
  }

  // Sample metallic
  float metallic = material.metallic_factor;
  if (material.metallic_texture_index != 0) {
    metallic *=
      textureBindless2D(material.metallic_texture_index, sampler_index, uvs).r;
  }

  // Sample roughness
  float roughness = material.roughness_factor;
  if (material.roughness_texture_index != 0) {
    roughness *=
      textureBindless2D(material.roughness_texture_index, sampler_index, uvs).r;
  }

  // Sample AO
  float ao = 1.0;
  if (material.ao_texture_index != 0) {
    ao = textureBindless2D(material.ao_texture_index, sampler_index, uvs).r;
    ao = mix(1.0, ao, material.ao_strength);
  }

  // Sample emissive
  vec3 emissive = material.emissive_factor.rgb;
  if (material.emissive_texture_index != 0) {
    emissive *=
      textureBindless2D(material.emissive_texture_index, sampler_index, uvs)
        .rgb;
  }

  // Output to GBuffer
  out_albedo_metallic = vec4(albedo.rgb, metallic);
  out_normal_roughness = vec4(final_normal * 0.5 + 0.5, roughness);
  out_emissive_ao = vec4(emissive, ao);

  // Wireframe overlay (your original wireframe effect)
  float n_dot_l =
    clamp(dot(final_normal, normalize(vec3(-1, 1, -1))), 0.5, 1.0);
  vec4 lit_color = vec4(1.0, 1.0, 1.0, 1.0) * n_dot_l;
  out_wireframe = mix(vec4(0.1), lit_color, edge_factor(1.0));
}