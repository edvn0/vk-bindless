#pragma stage : vertex

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec3 frag_world_pos;

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

void main()
{
  mat3 normal_matrix = mat3(transpose(inverse(pc.model)));
  frag_normal = normalize(normal_matrix * normal);
  frag_uv = texcoord;
  vec4 world_pos = pc.model * vec4(position, 1.0);
  frag_world_pos = world_pos.xyz;
  gl_Position = pc.proj * pc.view * world_pos;
}

#pragma stage : fragment

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in vec3 frag_world_pos;

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

void main()
{
  vec3 n = normalize(frag_normal);
  vec3 l = normalize(vec3(pc.light_direction));
  vec3 v = normalize(pc.camera_position.xyz - frag_world_pos);
  vec3 h = normalize(l + v);

  float ndotl = max(dot(n, l), 0.0);
  float ndoth = max(dot(n, h), 0.0);

  vec4 tex_color = textureBindless2D(pc.texture, 0, frag_uv);
  vec3 ambient = 0.05 * tex_color.rgb;
  vec3 diffuse = ndotl * tex_color.rgb;
  float spec = pow(ndoth, 32.0);
  vec3 specular = spec * vec3(1.0);

  frag_colour = vec4(ambient + diffuse + specular, tex_color.a);
}
