#pragma stage : vertex

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec3 frag_world_pos;

#include <ubo.glsl>

layout(std430, buffer_reference) readonly buffer SSBO
{
  mat4 transforms[];
};
layout(push_constant) uniform PushConstants
{
  UBO ubo;
  SSBO ssbo;
};

precise invariant gl_Position;

void
main()
{
  mat3 normal_matrix =
    mat3(transpose(inverse(ssbo.transforms[gl_InstanceIndex])));
  frag_normal = normalize(normal_matrix * normal);
  frag_uv = texcoord;
  vec4 world_pos = ssbo.transforms[gl_InstanceIndex] * vec4(position, 1.0);
  frag_world_pos = world_pos.xyz;
  gl_Position = ubo.proj * ubo.view * world_pos;
}

#pragma stage : fragment

#include <ubo.glsl>

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in vec3 frag_world_pos;

layout(location = 0) out uint out_material_index;
layout(location = 1) out vec2 out_uvs;
layout(location = 2) out vec4 out_normal_roughness;

layout(std430, buffer_reference) readonly buffer SSBO
{
  mat4 transforms[];
};
layout(push_constant) uniform PushConstants
{
  UBO ubo;
  SSBO ssbo;
};

void
main()
{
  vec3 n = normalize(frag_normal);

  vec3 enc_normal = normalize(n) * 0.5 + 0.5;

  out_material_index = uint(ubo.texture);
  out_uvs = frag_uv;
  out_normal_roughness = vec4(enc_normal, 1.0);
}
