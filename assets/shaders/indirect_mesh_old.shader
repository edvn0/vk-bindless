#pragma stage : vertex

layout(push_constant) uniform PerFrameData
{
  mat4 MVP;
};
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 signed_tangent;
layout(location = 0) out vec2 uv;
layout(location = 1) out vec3 normal;
void
main()
{
  gl_Position = MVP * vec4(in_position, 1.0);
  uv = in_texcoord;
  normal = vec3(in_normal);
};

#pragma stage : geometry

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec2 uv[];
layout(location = 1) in vec3 normal[];
layout(location = 0) out vec2 uvs;
layout(location = 1) out vec3 barycoords;
layout(location = 2) out vec3 normals;
void
main()
{
  const vec3 bc[3] =
    vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
  for (int i = 0; i < 3; i++) {
    gl_Position = gl_in[i].gl_Position;
    uvs = uv[i];
    barycoords = bc[i];
    normals = normal[i];
    EmitVertex();
  }
  EndPrimitive();
}

#pragma stage : fragment

layout(location = 0) in vec2 uvs;
layout(location = 1) in vec3 barycoords;
layout(location = 2) in vec3 normal;
layout(location = 0) out vec4 out_FragColor;
float
edgeFactor(float thickness)
{
  vec3 a3 = smoothstep(vec3(0.0), fwidth(barycoords) * thickness, barycoords);
  return min(min(a3.x, a3.y), a3.z);
}
void
main()
{
  float NdotL =
    clamp(dot(normalize(normal), normalize(vec3(-1, 1, -1))), 0.5, 1.0);
  vec4 color = vec4(1.0, 1.0, 1.0, 1.0) * NdotL;
  out_FragColor = mix(vec4(0.1), color, edgeFactor(1.0));
};