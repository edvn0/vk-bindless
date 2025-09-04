#ifndef UBO_GLSL
#define UBO_GLSL

layout(std430, buffer_reference) readonly buffer UBO
{
  mat4 view;
  mat4 proj;
  vec4 camera_position;
  vec4 light_direction;
  uint texture;
  uint cube_texture;
  uint padding[2];
};

#endif