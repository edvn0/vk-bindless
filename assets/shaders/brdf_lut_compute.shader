#pragma stage : compute

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (constant_id = 0) const uint NUM_SAMPLES = 1024u;
layout (std430, buffer_reference) readonly buffer Data {
  float16_t floats[];
};
layout (push_constant) uniform constants {
  uint BRDF_W;
  uint BRDF_H;
  Data data;
  uint padding[16];
};
const float PI = 3.1415926536;

vec2 hammersley2d(uint i, uint N) {
  uint bits = (i << 16u) | (i >> 16u);
  bits = ((bits & 0x55555555u)<<1u)|((bits & 0xAAAAAAAAu)>>1u);
  bits = ((bits & 0x33333333u)<<2u)|((bits & 0xCCCCCCCCu)>>2u);
  bits = ((bits & 0x0F0F0F0Fu)<<4u)|((bits & 0xF0F0F0F0u)>>4u);
  bits = ((bits & 0x00FF00FFu)<<8u)|((bits & 0xFF00FF00u)>>8u);
  float rdi = float(bits) * 2.3283064365386963e-10;
  return vec2(float(i) / float(N), rdi);
}

float random(vec2 co) {
  float a  = 12.9898;
  float b  = 78.233;
  float c  = 43758.5453;
  float dt = dot( co.xy ,vec2(a,b) );
  float sn = mod(dt, 3.14);
  return fract(sin(sn) * c);
}

vec3 importanceSample_GGX(vec2 Xi, float roughness, vec3 normal)
{
  float alpha = roughness * roughness;
  float phi = 2.0 * PI * Xi.x + random(normal.xz) * 0.1;
  float cosTheta  = sqrt((1.0 - Xi.y) / (1.0 + (alpha * alpha - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    vec3 up = abs(normal.z) < 0.999 ?
      vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangentX = normalize(cross(up, normal));
    vec3 tangentY = normalize(cross(normal, tangentX));
    return normalize(tangentX * H.x +
                     tangentY * H.y +
                       normal * H.z);
  }
  float G_SchlicksmithGGX(
    float dotNL, float dotNV, float roughness)
  {
    float k  = (roughness * roughness) / 2.0;
    float GL = dotNL / (dotNL * (1.0 - k) + k);
    float GV = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
  }

  float V_Ashikhmin(float NdotL, float NdotV) {
    return clamp(
      1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV)), 0.0, 1.0);
  }
  float D_Charlie(float sheenRoughness, float NdotH) {
    sheenRoughness = max(sheenRoughness, 0.000001);
    float invR = 1.0 / sheenRoughness;
    float cos2h = NdotH * NdotH;
    float sin2h = 1.0 - cos2h;
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
  }

  vec3 importanceSample_Charlie(
    vec2 Xi, float roughness, vec3 normal)
  {
    float alpha = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float sinTheta = pow(Xi.y, alpha / (2.0 *   alpha + 1.0));
    float cosTheta = sqrt(1.0 - sinTheta * sinTheta);
    vec3 H  = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    vec3 up = abs(normal.z) < 0.999 ?
      vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangentX = normalize(cross(up, normal));
    vec3 tangentY = normalize(cross(normal, tangentX));
    return normalize(tangentX * H.x +
                     tangentY * H.y +
                       normal * H.z);
  }

vec3 BRDF(float NoV, float roughness) {
  const vec3 N = vec3(0.0, 0.0, 1.0);
  vec3 V   = vec3(sqrt(1.0 - NoV*  NoV), 0.0, NoV);
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
          float sheenVisibility   = V_Ashikhmin(dotNL, dotNV);
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
    uint offset = gl_GlobalInvocationID.y * BRDF_W +
                  gl_GlobalInvocationID.x;
    data.floats[offset * 4 + 0] = float16_t(v.x);
    data.floats[offset * 4 + 1] = float16_t(v.y);
    data.floats[offset * 4 + 2] = float16_t(v.z);
}