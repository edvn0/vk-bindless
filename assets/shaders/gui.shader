#pragma stage : vertex

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;
layout(location = 2) out flat uint out_texture_id;

struct Vertex
{
  float x, y;
  float u, v;
  uint rgba;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer
{
  Vertex vertices[];
};

layout(push_constant) uniform PushConstants
{
  vec4 LRTB;
  VertexBuffer vb;
  uint texture_id;
}
pc;
void
main()
{
  float L = pc.LRTB.x;
  float R = pc.LRTB.y;
  float T = pc.LRTB.z;
  float B = pc.LRTB.w;

  mat4 proj = mat4(2.0 / (R - L),
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   2.0 / (T - B),
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   -1.0,
                   0.0,
                   (R + L) / (L - R),
                   (T + B) / (B - T),
                   0.0,
                   1.0);

  Vertex v = pc.vb.vertices[gl_VertexIndex];
  out_color = unpackUnorm4x8(v.rgba);

  out_uv = vec2(v.u, v.v);
  out_texture_id = pc.texture_id;
  gl_Position = proj * vec4(v.x, v.y, 0, 1);
}

#pragma stage : fragment

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in flat uint in_texture_id;
layout(location = 0) out vec4 out_color;

layout(constant_id = 0) const bool is_non_linear_colour_space = false;

void
main()
{
  vec4 sampled = in_color * textureBindless2D(in_texture_id, 0, in_uv);
  out_color = is_non_linear_colour_space
                ? vec4(pow(sampled.rgb, vec3(2.2)), sampled.a)
                : sampled;
}