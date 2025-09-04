#pragma stage : vertex

layout(location = 0) out vec2 out_uvs;

void
main()
{
  out_uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(out_uvs * 2.0f + -1.0f, 0.0f, 1.0f);
  out_uvs.y = 1.0F - out_uvs.y;
}

#pragma stage : fragment

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

#include <ubo.glsl>

layout(push_constant) uniform LightingPC
{
  uint
    g_material_tex; // index of R8_UINT texture (stores material id per pixel)
  uint g_uvs_tex;   // uvs, f16 RG
  uint g_normal_rough_tex; // index of RGBA16F texture
  uint g_depth_tex;        // index of D32F texture
  uint sampler_idx;        // sampler for GBuffer lookups
  UBO ubo;
};

vec3
reconstruct_world_pos(vec2 uv, float depth, mat4 invProj, mat4 invView)
{
  vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
  vec4 view_pos = invProj * ndc;
  view_pos /= view_pos.w;
  vec4 world_pos = invView * view_pos;
  return world_pos.xyz;
}

void
main()
{
  // --- Fetch GBuffer values ---
  uint material_index =
    floatBitsToUint(textureBindless2D(g_material_tex, sampler_idx, frag_uv).r);
  vec4 normal_rough =
    textureBindless2D(g_normal_rough_tex, sampler_idx, frag_uv);
  float depth = textureBindless2D(g_depth_tex, sampler_idx, frag_uv).r;

  // Decode normal
  vec3 normal = normalize(normal_rough.xyz * 2.0 - 1.0);
  float roughness = normal_rough.a;

  // Reconstruct world position from depth
  mat4 invProj = inverse(ubo.proj);
  mat4 invView = inverse(ubo.view);
  vec3 world_pos = reconstruct_world_pos(frag_uv, depth, invProj, invView);

  // Fetch albedo from material index
  vec2 texture_uvs = textureBindless2D(g_uvs_tex, sampler_idx, frag_uv).rg;
  vec4 albedo = textureBindless2D(material_index, sampler_idx, texture_uvs);

  // --- Lighting ---
  vec3 L = normalize(vec3(ubo.light_direction));
  vec3 V = normalize(ubo.camera_position.xyz - world_pos);
  vec3 H = normalize(L + V);

  float NdotL = max(dot(normal, L), 0.0);
  float NdotH = max(dot(normal, H), 0.0);

  vec3 diffuse = albedo.rgb * NdotL;
  float spec = pow(NdotH, mix(1.0, 128.0, 1.0 - roughness));
  vec3 specular = spec * vec3(1.0);

  out_color = vec4(diffuse + specular, 1.0);
}