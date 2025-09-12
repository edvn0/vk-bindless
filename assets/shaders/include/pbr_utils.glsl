#ifndef PBR_UTILS_GLSL
#define PBR_UTILS_GLSL

#include <math_constants.glsl>

vec2
hammersley2d(uint i, uint N)
{
  uint bits = (i << 16u) | (i >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  float rdi = float(bits) * 2.3283064365386963e-10;
  return vec2(float(i) / float(N), rdi);
}

float
random(vec2 co)
{
  float a = 12.9898;
  float b = 78.233;
  float c = 43758.5453;
  float dt = dot(co.xy, vec2(a, b));
  float sn = mod(dt, 3.14);
  return fract(sin(sn) * c);
}

vec3
importanceSample_GGX(vec2 Xi, float roughness, vec3 normal)
{
  float alpha = roughness * roughness;
  float phi = 2.0 * PI * Xi.x + random(normal.xz) * 0.1;
  float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (alpha * alpha - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
  vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangentX = normalize(cross(up, normal));
  vec3 tangentY = normalize(cross(normal, tangentX));
  return normalize(tangentX * H.x + tangentY * H.y + normal * H.z);
}
float
G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
  float k = (roughness * roughness) / 2.0;
  float GL = dotNL / (dotNL * (1.0 - k) + k);
  float GV = dotNV / (dotNV * (1.0 - k) + k);
  return GL * GV;
}

float
V_Ashikhmin(float NdotL, float NdotV)
{
  return clamp(1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV)), 0.0, 1.0);
}
float
D_Charlie(float sheenRoughness, float NdotH)
{
  sheenRoughness = max(sheenRoughness, 0.000001);
  float invR = 1.0 / sheenRoughness;
  float cos2h = NdotH * NdotH;
  float sin2h = 1.0 - cos2h;
  return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
}

vec3
importanceSample_Charlie(vec2 Xi, float roughness, vec3 normal)
{
  float alpha = roughness * roughness;
  float phi = 2.0 * PI * Xi.x;
  float sinTheta = pow(Xi.y, alpha / (2.0 * alpha + 1.0));
  float cosTheta = sqrt(1.0 - sinTheta * sinTheta);
  vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
  vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangentX = normalize(cross(up, normal));
  vec3 tangentY = normalize(cross(normal, tangentX));
  return normalize(tangentX * H.x + tangentY * H.y + normal * H.z);
}

#endif