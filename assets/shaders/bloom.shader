#pragma stage : compute("extract_bright")
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform ExtractPC
{
  uint input_texture_idx;
  uint output_texture_idx;
  uint sampler_idx;
  float bloom_threshold;
}
pc;

void
extract_bright()
{
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 image_size = textureSize(
    sampler2D(textures[pc.input_texture_idx], samplers[pc.sampler_idx]), 0);

  if (coord.x >= image_size.x || coord.y >= image_size.y)
    return;

  vec3 color = texelFetch(sampler2D(textures[pc.input_texture_idx],
                                    samplers[pc.sampler_idx]),
                          coord,
                          0)
                 .rgb;

  // Extract bright pixels above threshold
  float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722)); // Luminance
  vec3 bloom_color = brightness > pc.bloom_threshold ? color : vec3(0.0);

  imageStore(images2D[pc.output_texture_idx], coord, vec4(bloom_color, 1.0));
}

#pragma stage : compute("blur_horizontal")
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform BlurPC
{
  uint input_texture_idx;
  uint output_texture_idx;
  uint sampler_idx;
  uint blur_radius;
}
blur_pc;

void
blur_horizontal()
{
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 image_size = textureSize(sampler2D(textures[blur_pc.input_texture_idx],
                                           samplers[blur_pc.sampler_idx]),
                                 0);

  if (coord.x >= image_size.x || coord.y >= image_size.y)
    return;

  vec3 result = vec3(0.0);
  float total_weight = 0.0;

  // Gaussian weights for blur_radius=5: [0.06, 0.24, 0.40, 0.24, 0.06]
  float weights[5] = float[](0.06, 0.24, 0.40, 0.24, 0.06);

  for (int i = -int(blur_pc.blur_radius); i <= int(blur_pc.blur_radius); i++) {
    ivec2 sample_coord = coord + ivec2(i, 0);
    sample_coord.x = clamp(sample_coord.x, 0, image_size.x - 1);

    vec3 sample_color =
      texelFetch(sampler2D(textures[blur_pc.input_texture_idx],
                           samplers[blur_pc.sampler_idx]),
                 sample_coord,
                 0)
        .rgb;
    float weight = weights[abs(i)];

    result += sample_color * weight;
    total_weight += weight;
  }

  imageStore(images2D[blur_pc.output_texture_idx],
             coord,
             vec4(result / total_weight, 1.0));
}

#pragma stage : compute("blur_vertical")
layout(local_size_x = 1, local_size_y = 64, local_size_z = 1) in;

void
blur_vertical()
{
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 image_size = textureSize(sampler2D(textures[blur_pc.input_texture_idx],
                                           samplers[blur_pc.sampler_idx]),
                                 0);

  if (coord.x >= image_size.x || coord.y >= image_size.y)
    return;

  vec3 result = vec3(0.0);
  float total_weight = 0.0;

  float weights[5] = float[](0.06, 0.24, 0.40, 0.24, 0.06);

  for (int i = -int(blur_pc.blur_radius); i <= int(blur_pc.blur_radius); i++) {
    ivec2 sample_coord = coord + ivec2(0, i);
    sample_coord.y = clamp(sample_coord.y, 0, image_size.y - 1);

    vec3 sample_color =
      texelFetch(sampler2D(textures[blur_pc.input_texture_idx],
                           samplers[blur_pc.sampler_idx]),
                 sample_coord,
                 0)
        .rgb;
    float weight = weights[abs(i)];

    result += sample_color * weight;
    total_weight += weight;
  }

  imageStore(images2D[blur_pc.output_texture_idx],
             coord,
             vec4(result / total_weight, 1.0));
}