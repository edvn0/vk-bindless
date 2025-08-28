#pragma stage : vertex

layout(location = 0) in vec3 position;

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
  gl_Position = pc.proj * pc.view * pc.model * vec4(position, 1.0);
}

#pragma stage : fragment

void
main()
{
}