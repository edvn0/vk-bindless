#pragma stage : vertex

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;

layout(std430, buffer_reference) readonly buffer UBO
{
  mat4 model;
  mat4 view;
  mat4 proj;
  vec4 camera_position;
  vec4 light_direction;
  uint texture;
  uint cube_texture;
};
layout(push_constant) uniform PushConstants
{
  UBO pc;
};

precise invariant gl_Position;

void
main()
{
  mat3 normal_matrix = mat3(transpose(inverse(pc.model)));
  frag_normal = normalize(normal_matrix * normal);

  frag_uv = texcoord;

  gl_Position = pc.proj * pc.view * pc.model * vec4(position, 1.0);
}

#pragma stage : fragment

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;

layout(location = 0) out vec4 frag_colour;

layout(std430, buffer_reference) readonly buffer UBO
{
  mat4 model;
  mat4 view;
  mat4 proj;
  vec4 camera_position;
  vec4 light_direction;
  uint texture;
  uint cube_texture;
};
layout(push_constant) uniform PushConstants
{
  UBO pc;
};

void
main()
{
  float ndotl = max(dot(frag_normal, vec3(pc.light_direction)), 0.0);

  vec4 texture_color = textureBindless2D(pc.texture, 0, frag_uv);
  vec4 diffuse = texture_color;
  frag_colour = vec4(ndotl * diffuse.rgb, texture_color.a);
}