#pragma stage : vertex
layout(location = 0) out vec2 out_uvs;

void
main()
{
  // Fullscreen triangle UVs
  out_uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(out_uvs * 2.0 - 1.0, 0.0, 1.0);
  out_uvs.y = 1.0 - out_uvs.y;
}

#pragma stage : fragment
layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

#include <math_constants.glsl>
#include <ubo.glsl>

layout(push_constant) uniform LightingPC
{
  uint g_uvs_tex;             // f16 RG
  uint g_normal_rough_tex;    // RGBA16F
  uint g_texture_indices_tex; // RGBAUI16
  uint g_depth_tex;           // D32F
  uint sampler_idx;           // sampler for GBuffer
  UBO ubo;
};

vec3
fresnel_schlick(float cosTheta, vec3 F0)
{
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float
distribution_ggx(vec3 N, vec3 H, float roughness)
{
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  return a2 / (3.14159265 * denom * denom + 1e-5);
}

float
geometry_schlick_ggx(float NdotV, float roughness)
{
  float r = roughness + 1.0;
  float k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k + 1e-5);
}

float
geometry_smith(vec3 N, vec3 V, vec3 L, float roughness)
{
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx1 = geometry_schlick_ggx(NdotV, roughness);
  float ggx2 = geometry_schlick_ggx(NdotL, roughness);
  return ggx1 * ggx2;
}

vec3
reconstruct_world_position(float reverse_z, vec2 uv)
{
  vec2 ndc_xy = uv * 2.0 - 1.0;
  vec4 clip_pos = vec4(ndc_xy, reverse_z, 1.0);
  mat4 inv_proj = inverse(ubo.proj);
  mat4 inv_view = inverse(ubo.view);
  mat4 inverse_view_proj_matrix = inv_view * inv_proj;
  vec4 world_pos = inverse_view_proj_matrix * clip_pos;
  return world_pos.xyz / world_pos.w;
}

void
main()
{
  // --- Fetch GBuffer values ---
  vec4 texture_indices =
    textureBindless2D(g_texture_indices_tex, sampler_idx, frag_uv);
  uint albedo_index = floatBitsToUint(texture_indices.r);
  uint metallic_index = floatBitsToUint(texture_indices.a);

  vec4 normal_rough =
    textureBindless2D(g_normal_rough_tex, sampler_idx, frag_uv);
  vec2 texture_uvs = textureBindless2D(g_uvs_tex, sampler_idx, frag_uv).rg;
  vec3 world_pos = reconstruct_world_position(
    textureBindless2D(g_depth_tex, sampler_idx, frag_uv).r, frag_uv);

  vec3 normal = normalize(normal_rough.xyz);
  float roughness = clamp(normal_rough.a, 0.05, 1.0); // avoid zero roughness

  // Sample albedo texture
  vec4 albedo = vec4(1.0); // default white
  if (albedo_index != 0) {
    albedo = textureBindless2D(albedo_index, sampler_idx, texture_uvs);
  }

  // Sample metallic texture
  float metallic = 0.0; // default non-metallic
  if (metallic_index != 0) {
    metallic = textureBindless2D(metallic_index, sampler_idx, texture_uvs).r;
  }

  // --- Lighting ---
  vec3 L = normalize(vec3(ubo.light_direction)); // directional light
  vec3 V = normalize(ubo.camera_position.xyz - world_pos);
  vec3 H = normalize(V + L);

  float NdotL = max(dot(normal, L), 0.0);
  float NdotV = max(dot(normal, V), 0.0);

  vec3 F0 = vec3(0.04);               // default dielectric F0
  F0 = mix(F0, albedo.rgb, metallic); // blend with metallic

  vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
  float D = distribution_ggx(normal, H, roughness);
  float G = geometry_smith(normal, V, L, roughness);

  vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-5);

  vec3 kD = vec3(1.0) - F; // energy conservation
  kD *= (1.0 - metallic);  // metallic surfaces have no diffuse

  vec3 diffuse = kD * albedo.rgb / PI;

  vec3 color = (diffuse + specular) * NdotL;

  vec3 ambient = vec3(0.03) * albedo.rgb;
  color += ambient;

  out_color = vec4(color, albedo.a);
}