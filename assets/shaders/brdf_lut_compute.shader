#pragma stage : compute

#include <math_constants.glsl>
#include <pbr_utils.glsl>

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(constant_id = 0) const uint NUM_SAMPLES = 1024u;
layout(std430, buffer_reference) readonly buffer Data
{
  float16_t floats[];
};
layout(push_constant) uniform constants
{
  uint BRDF_W;
  uint BRDF_H;
  Data data;
};

vec3
BRDF(float NoV, float roughness)
{
  const vec3 N = vec3(0.0, 0.0, 1.0);
  vec3 V = vec3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
  vec3 LUT = vec3(0.0);
  for (uint i = 0u; i < NUM_SAMPLES; i++) {
    vec2 Xi = hammersley2d(i, NUM_SAMPLES);
    vec3 H = importanceSample_GGX(Xi, roughness, N);
    vec3 L = 2.0 * dot(V, H) * H - V;
    float dotNL = max(dot(N, L), 0.0);
    float dotNV = max(dot(N, V), 0.0);
    float dotVH = max(dot(V, H), 0.0);
    float dotNH = max(dot(H, N), 0.0);
    if (dotNL > 0.0) {
      float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
      float G_Vis = (G * dotVH) / (dotNH * dotNV);
      float Fc = pow(1.0 - dotVH, 5.0);
      LUT.rg += vec2((1.0 - Fc) * G_Vis, Fc * G_Vis);
    }
  }
  for (uint i = 0u; i < NUM_SAMPLES; i++) {
    vec2 Xi = hammersley2d(i, NUM_SAMPLES);
    vec3 H = importanceSample_Charlie(Xi, roughness, N);
    vec3 L = 2.0 * dot(V, H) * H - V;
    float dotNL = max(dot(N, L), 0.0);
    float dotNV = max(dot(N, V), 0.0);
    float dotVH = max(dot(V, H), 0.0);
    float dotNH = max(dot(H, N), 0.0);
    if (dotNL > 0.0) {
      float sheenDistribution = D_Charlie(roughness, dotNH);
      float sheenVisibility = V_Ashikhmin(dotNL, dotNV);
      LUT.b += sheenVisibility * sheenDistribution * dotNL * dotVH;
    }
  }
  return LUT / float(NUM_SAMPLES);
}

void
main()
{
  vec2 uv;
  uv.x = (float(gl_GlobalInvocationID.x) + 0.5) / float(BRDF_W);
  uv.y = (float(gl_GlobalInvocationID.y) + 0.5) / float(BRDF_H);
  vec3 v = BRDF(uv.x, uv.y);
  uint offset = gl_GlobalInvocationID.y * BRDF_W + gl_GlobalInvocationID.x;
  data.floats[offset * 4 + 0] = float16_t(v.x);
  data.floats[offset * 4 + 1] = float16_t(v.y);
  data.floats[offset * 4 + 2] = float16_t(v.z);
}