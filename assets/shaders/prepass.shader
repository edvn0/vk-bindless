#pragma stage : vertex

#include <ubo.glsl>

layout(location = 0) in vec3 position;

layout(std430, buffer_reference) readonly buffer SSBO
{
  mat4 transforms[];
};
layout(push_constant) uniform PushConstants
{
  UBO ubo;
  SSBO ssbo;
};

precise invariant gl_Position;

void
main()
{
  gl_Position = ubo.proj * ubo.view * ssbo.transforms[gl_InstanceIndex] *
                vec4(position, 1.0);
}

#pragma stage : fragment

void
main()
{
}