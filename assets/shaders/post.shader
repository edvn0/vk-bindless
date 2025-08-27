#pragma stage : vertex

layout(location = 0) out flat uint index;
layout(location = 1) out vec2 out_uvs;

layout(push_constant) uniform PushConst
{
  uint textureIndex;
}
pc;

void
main()
{
  index = pc.textureIndex;
  out_uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(out_uvs * 2.0f + -1.0f, 0.0f, 1.0f);
  out_uvs.y = 1.0F - out_uvs.y;
}

#pragma stage : fragment
layout(location = 0) in flat uint index;
layout(location = 1) in vec2 v_uvs;

layout(location = 0) out vec4 out_colour;

vec3
ACESFilm(vec3 x)
{
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3
GammaCorrect(vec3 color)
{
  return pow(color, vec3(1.0 / 2.2));
}

void
main()
{
  vec3 color = textureBindless2D(index, 0, v_uvs).rgb;
  color = ACESFilm(color);
  color = GammaCorrect(color);
  out_colour = vec4(color, 1.0);
}