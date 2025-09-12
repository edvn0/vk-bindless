#ifndef PACKING_GLSL
#define PACKING_GLSL

vec4
unpackSnorm3x10_1x2_manual(uint packed)
{
  int ix = int(packed & 0x3FF);
  int iy = int((packed >> 10) & 0x3FF);
  int iz = int((packed >> 20) & 0x3FF);
  int iw = int((packed >> 30) & 0x3);

  ix = (ix << 22) >> 22;
  iy = (iy << 22) >> 22;
  iz = (iz << 22) >> 22;
  iw = (iw << 30) >> 30;

  vec4 result;
  result.x = clamp(float(ix) / 511.0, -1.0, 1.0);
  result.y = clamp(float(iy) / 511.0, -1.0, 1.0);
  result.z = clamp(float(iz) / 511.0, -1.0, 1.0);
  result.w = clamp(float(iw) / 1.0, -1.0, 1.0);
  return result;
}

#endif