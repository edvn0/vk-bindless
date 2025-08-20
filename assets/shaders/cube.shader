#pragma stage : vertex

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_tangent;
layout(location = 3) out vec3 frag_bitangent;
layout(location = 4) out vec2 frag_uv;

// Static cube vertices (8 vertices)
const vec3 cube_vertices[8] = vec3[](vec3(-0.5, -0.5, -0.5), // 0
                                     vec3(0.5, -0.5, -0.5),  // 1
                                     vec3(0.5, 0.5, -0.5),   // 2
                                     vec3(-0.5, 0.5, -0.5),  // 3
                                     vec3(-0.5, -0.5, 0.5),  // 4
                                     vec3(0.5, -0.5, 0.5),   // 5
                                     vec3(0.5, 0.5, 0.5),    // 6
                                     vec3(-0.5, 0.5, 0.5)    // 7
);

// Static cube indices (36 indices for 12 triangles, 6 faces)
const uint cube_indices[36] = uint[](
  // Front face
  0,
  1,
  2,
  2,
  3,
  0,
  // Back face
  4,
  6,
  5,
  6,
  4,
  7,
  // Left face
  4,
  0,
  3,
  3,
  7,
  4,
  // Right face
  1,
  5,
  6,
  6,
  2,
  1,
  // Bottom face
  4,
  5,
  1,
  1,
  0,
  4,
  // Top face
  3,
  2,
  6,
  6,
  7,
  3);

// Static cube normals (per face, 6 faces)
const vec3 face_normals[6] = vec3[](vec3(0.0, 0.0, -1.0), // Front
                                    vec3(0.0, 0.0, 1.0),  // Back
                                    vec3(-1.0, 0.0, 0.0), // Left
                                    vec3(1.0, 0.0, 0.0),  // Right
                                    vec3(0.0, -1.0, 0.0), // Bottom
                                    vec3(0.0, 1.0, 0.0)   // Top
);

// Static cube tangents (per face)
const vec3 face_tangents[6] = vec3[](vec3(1.0, 0.0, 0.0),  // Front
                                     vec3(-1.0, 0.0, 0.0), // Back
                                     vec3(0.0, 0.0, 1.0),  // Left
                                     vec3(0.0, 0.0, -1.0), // Right
                                     vec3(1.0, 0.0, 0.0),  // Bottom
                                     vec3(1.0, 0.0, 0.0)   // Top
);

// Static cube bitangents (per face)
const vec3 face_bitangents[6] = vec3[](vec3(0.0, 1.0, 0.0), // Front
                                       vec3(0.0, 1.0, 0.0), // Back
                                       vec3(0.0, 1.0, 0.0), // Left
                                       vec3(0.0, 1.0, 0.0), // Right
                                       vec3(0.0, 0.0, 1.0), // Bottom
                                       vec3(0.0, 0.0, -1.0) // Top
);

// Static UV coordinates (per face corner)
const vec2 face_uvs[4] = vec2[](vec2(0.0, 0.0), // Bottom-left
                                vec2(1.0, 0.0), // Bottom-right
                                vec2(1.0, 1.0), // Top-right
                                vec2(0.0, 1.0)  // Top-left
);

// Map each triangle vertex to its face and UV index
const uint face_mapping[36] = uint[](
  // Front face (face 0)
  0,
  0,
  0,
  0,
  0,
  0,
  // Back face (face 1)
  1,
  1,
  1,
  1,
  1,
  1,
  // Left face (face 2)
  2,
  2,
  2,
  2,
  2,
  2,
  // Right face (face 3)
  3,
  3,
  3,
  3,
  3,
  3,
  // Bottom face (face 4)
  4,
  4,
  4,
  4,
  4,
  4,
  // Top face (face 5)
  5,
  5,
  5,
  5,
  5,
  5);

// Map each triangle vertex to its UV corner
const uint uv_mapping[36] = uint[](
  // Front face: 0,1,2 -> 0,1,2  2,3,0 -> 2,3,0
  0,
  1,
  2,
  2,
  3,
  0,
  // Back face: 4,6,5 -> 0,2,1  6,4,7 -> 2,0,3
  0,
  2,
  1,
  2,
  0,
  3,
  // Left face: 4,0,3 -> 0,1,2  3,7,4 -> 2,3,0
  0,
  1,
  2,
  2,
  3,
  0,
  // Right face: 1,5,6 -> 0,1,2  6,2,1 -> 2,3,0
  0,
  1,
  2,
  2,
  3,
  0,
  // Bottom face: 4,5,1 -> 0,1,2  1,0,4 -> 2,3,0
  0,
  1,
  2,
  2,
  3,
  0,
  // Top face: 3,2,6 -> 0,1,2  6,7,3 -> 2,3,0
  0,
  1,
  2,
  2,
  3,
  0);

layout(push_constant) uniform PushConstants
{
  mat4 mvp_matrix;
}
pc;

void
main()
{
  // Get the vertex index from the current triangle
  uint vertex_index = cube_indices[gl_VertexIndex];

  // Get face and UV mapping for TBN calculation
  uint face_index = face_mapping[gl_VertexIndex];
  uint uv_index = uv_mapping[gl_VertexIndex];

  // Get position
  vec3 position = cube_vertices[vertex_index];

  // Get TBN vectors for this face
  vec3 normal = face_normals[face_index];
  vec3 tangent = face_tangents[face_index];
  vec3 bitangent = face_bitangents[face_index];

  // Transform TBN to world space (assuming model matrix is part of MVP)
  mat3 normal_matrix = mat3(transpose(inverse(pc.mvp_matrix)));
  frag_normal = normalize(normal_matrix * normal);
  frag_tangent = normalize(normal_matrix * tangent);
  frag_bitangent = normalize(normal_matrix * bitangent);

  frag_uv = face_uvs[uv_index];

  const vec3 face_colors[6] = vec3[](vec3(1.0, 0.0, 0.0), // Front - Red
                                     vec3(0.0, 1.0, 0.0), // Back - Green
                                     vec3(0.0, 0.0, 1.0), // Left - Blue
                                     vec3(1.0, 1.0, 0.0), // Right - Yellow
                                     vec3(1.0, 0.0, 1.0), // Bottom - Magenta
                                     vec3(0.0, 1.0, 1.0)  // Top - Cyan
  );
  frag_color = face_colors[face_index];

  gl_Position = pc.mvp_matrix * vec4(position, 1.0);
}

#pragma stage : fragment

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_tangent;
layout(location = 3) in vec3 frag_bitangent;
layout(location = 4) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

void
main()
{
  mat3 tbn = mat3(
    normalize(frag_tangent), normalize(frag_bitangent), normalize(frag_normal));

  vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
  float ndotl = max(dot(frag_normal, light_dir), 0.0);

  vec3 final_color = frag_color * (0.3 + 0.7 * ndotl);

  out_color = vec4(final_color, 1.0);
}