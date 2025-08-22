#pragma once

#include "vk-bindless/handle.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <variant>

namespace VkBindless {

static constexpr auto max_colour_attachments = 8U;

enum class IndexFormat : std::uint8_t
{
  UI16,
  UI32,
  UI8,
};

enum class Topology : std::uint8_t
{
  Point,
  Line,
  LineStrip,
  Triangle,
  TriangleStrip,
  Patch,
};

enum class CullMode : std::uint8_t
{
  None,
  Front,
  Back
};

enum class WindingMode : std::uint8_t
{
  CCW,
  CW
};

enum class ColorSpace : std::uint8_t
{
  SRGB_NONLINEAR,
  SRGB_EXTENDED_LINEAR,
  HDR10,
  BT709_LINEAR,
};

enum class TextureType : std::uint8_t
{
  Two,
  Three,
  Cube,
};

struct Dimensions
{
  std::uint32_t width = 1;
  std::uint32_t height = 1;
  std::uint32_t depth = 1;
  inline auto divide1D(std::integral auto v) const
  {
    return Dimensions{
      .width = width / v,
      .height = height,
      .depth = depth,
    };
  }
  inline auto divide2D(std::integral auto v) const
  {
    return Dimensions{
      .width = width / v,
      .height = height / v,
      .depth = depth,
    };
  }
  inline auto divide3D(std::integral auto v) const
  {
    return Dimensions{
      .width = width / v,
      .height = height / v,
      .depth = depth / v,
    };
  }
  inline auto operator<=>(const Dimensions& other) const = default;
};

enum class CompareOp : std::uint8_t
{
  Never = 0,
  Less,
  Equal,
  LessEqual,
  Greater,
  NotEqual,
  GreaterEqual,
  AlwaysPass
};

enum class StencilOp : std::uint8_t
{
  Keep = 0,
  Zero,
  Replace,
  IncrementClamp,
  DecrementClamp,
  Invert,
  IncrementWrap,
  DecrementWrap
};

enum class BlendOp : std::uint8_t
{
  Add = 0,
  Subtract,
  ReverseSubtract,
  Min,
  Max
};

enum class BlendFactor : std::uint8_t
{
  Zero = 0,
  One,
  SrcColor,
  OneMinusSrcColor,
  SrcAlpha,
  OneMinusSrcAlpha,
  DstColor,
  OneMinusDstColor,
  DstAlpha,
  OneMinusDstAlpha,
  SrcAlphaSaturated,
  BlendColor,
  OneMinusBlendColor,
  BlendAlpha,
  OneMinusBlendAlpha,
  Src1Color,
  OneMinusSrc1Color,
  Src1Alpha,
  OneMinusSrc1Alpha
};

enum class LoadOp : std::uint8_t
{
  Invalid = 0,
  DontCare,
  Load,
  Clear,
  None,
};

enum class StoreOp : std::uint8_t
{
  DontCare = 0,
  Store,
  MsaaResolve,
  None,
};

enum class ResolveMode : std::uint8_t
{
  None = 0,
  SampleZero, // always supported
  Average,
  Min,
  Max,
};

enum class PolygonMode : uint8_t
{
  Fill = 0,
  Line = 1,
};

enum class VertexFormat
{
  Invalid = 0,

  Float1,
  Float2,
  Float3,
  Float4,

  Byte1,
  Byte2,
  Byte3,
  Byte4,

  UByte1,
  UByte2,
  UByte3,
  UByte4,

  Short1,
  Short2,
  Short3,
  Short4,

  UShort1,
  UShort2,
  UShort3,
  UShort4,

  Byte2Norm,
  Byte4Norm,

  UByte2Norm,
  UByte4Norm,

  Short2Norm,
  Short4Norm,

  UShort2Norm,
  UShort4Norm,

  Int1,
  Int2,
  Int3,
  Int4,

  UInt1,
  UInt2,
  UInt3,
  UInt4,

  HalfFloat1,
  HalfFloat2,
  HalfFloat3,
  HalfFloat4,

  Int_2_10_10_10_REV,
};

enum class Format : uint8_t
{
  Invalid = 0,

  R_UN8,
  R_UI16,
  R_UI32,
  R_UN16,
  R_F16,
  R_F32,

  RG_UN8,
  RG_UI16,
  RG_UI32,
  RG_UN16,
  RG_F16,
  RG_F32,

  RGBA_UN8,
  RGBA_UI32,
  RGBA_F16,
  RGBA_F32,
  RGBA_SRGB8,

  BGRA_UN8,
  BGRA_SRGB8,

  A2B10G10R10_UN,
  A2R10G10B10_UN,

  ETC2_RGB8,
  ETC2_SRGB8,
  BC7_RGBA,

  Z_UN16,
  Z_UN24,
  Z_F32,
  Z_UN24_S_UI8,
  Z_F32_S_UI8,

  YUV_NV12,
  YUV_420p,
};

struct VertexInput final
{
  static constexpr std::uint32_t vertex_attribute_max_count = 16;
  static constexpr std::uint32_t input_bindings_max_count = 16;
  struct VertexAttribute final
  {
    std::uint32_t location = 0;
    std::uint32_t binding = 0;
    VertexFormat format = VertexFormat::Invalid;
    std::uintptr_t offset = 0;

    auto operator<=>(const VertexAttribute& other) const = default;
  };
  std::array<VertexAttribute, vertex_attribute_max_count> attributes{};
  struct VertexInputBinding final
  {
    std::uint32_t stride = 0;

    auto operator<=>(const VertexInputBinding& other) const = default;
  };
  std::array<VertexInputBinding, input_bindings_max_count> input_bindings{};

  [[nodiscard]] auto get_attributes_count() const
  {
    std::uint32_t n = 0;
    while (n < vertex_attribute_max_count &&
           attributes[n].format != VertexFormat::Invalid) {
      n++;
    }
    return n;
  }

  [[nodiscard]] auto get_input_bindings_count() const
  {
    uint32_t n = 0;
    while (n < input_bindings_max_count && input_bindings[n].stride) {
      n++;
    }
    return n;
  }

  auto operator<=>(const VertexInput& other) const = default;
};

struct ColourAttachment
{
  Format format = Format::Invalid;
  bool blend_enabled = false;
  BlendOp rgb_blend_op = BlendOp::Add;
  BlendOp alpha_blend_op = BlendOp::Add;
  BlendFactor src_rgb_blend_factor = BlendFactor::One;
  BlendFactor src_alpha_blend_factor = BlendFactor::One;
  BlendFactor dst_rgb_blend_factor = BlendFactor::Zero;
  BlendFactor dst_alpha_blend_factor = BlendFactor::Zero;
};

struct SpecialisationConstantDescription
{
  struct SpecialisationConstantEntry
  {
    std::uint32_t constant_id = 0;
    std::uint32_t offset = 0; // offset within SpecializationConstantDesc::data
    std::size_t size = 0;
  };

  static constexpr auto max_specialization_constants = 16U;

  std::array<SpecialisationConstantEntry, max_specialization_constants>
    entries{};
  std::span<const std::byte> data{};

  auto get_specialisation_constants_count() const
  {
    std::uint32_t n = 0;
    while (n < max_specialization_constants && entries[n].size) {
      n++;
    }
    return n;
  }
};

struct Dependencies
{
  static constexpr auto max_dependencies = 4U;
  std::array<TextureHandle, max_dependencies> textures = {};
  std::array<BufferHandle, max_dependencies> buffers = {};
};

using ClearColourValue = std::variant<std::array<float, 4>,
                                      std::array<std::uint32_t, 4>,
                                      std::array<std::int32_t, 4>>;

struct RenderPass final
{
  struct AttachmentDescription final
  {
    LoadOp load_op = LoadOp::Invalid;
    StoreOp store_op = StoreOp::Store;
    ResolveMode resolve_mode = ResolveMode::Average;
    std::uint8_t layer = 0;
    std::uint8_t level = 0;
    ClearColourValue clear_colour =
      std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f };
    float clear_depth = 1.0f;
    uint32_t clear_stencil = 0;
  };

  std::array<AttachmentDescription, max_colour_attachments> color{};
  AttachmentDescription depth = {
    .load_op = LoadOp::DontCare,
    .store_op = StoreOp::DontCare,
  };
  AttachmentDescription stencil = {
    .load_op = LoadOp::Invalid,
    .store_op = StoreOp::DontCare,
  };

  std::uint32_t layer_count = 1;
  std::uint32_t view_mask = 0;

  [[nodiscard]] auto get_colour_attachment_count() const
  {
    std::uint32_t n = 0;
    while (n < max_colour_attachments && color[n].load_op != LoadOp::Invalid) {
      n++;
    }
    return n;
  }
};

struct Framebuffer final
{
  struct AttachmentDescription
  {
    TextureHandle texture{};
    TextureHandle resolve_texture{};
  };

  std::array<AttachmentDescription, max_colour_attachments> color{};
  AttachmentDescription depth_stencil{};

  std::string debug_name{};

  [[nodiscard]] auto get_colour_attachment_count() const
  {
    std::uint32_t n = 0;
    while (n < max_colour_attachments && color[n].texture) {
      n++;
    }
    return n;
  }
};

struct Viewport
{
  float x = 0.0f;
  float y = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
  float minDepth = 1.0f;
  float maxDepth = 0.0f;
};

struct ScissorRect
{
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct StencilState
{
  bool enabled{ false };
  StencilOp stencil_failure_operation = StencilOp::Keep;
  StencilOp depth_failure_operation = StencilOp::Keep;
  StencilOp depth_stencil_pass_operation = StencilOp::Keep;
  CompareOp stencil_compare_op = CompareOp::AlwaysPass;
  std::uint32_t read_mask = static_cast<std::uint32_t>(~0);
  std::uint32_t write_mask = static_cast<std::uint32_t>(~0);
};

struct DepthState
{
  CompareOp compare_operation{ CompareOp::AlwaysPass };
  bool is_depth_write_enabled{ false };
};

struct TextureLayers
{
  std::uint32_t mip_level = 0;
  std::uint32_t layer = 0;
  std::uint32_t num_layers = 1;
};

struct Offset3D
{
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

#define MAKE_BIT_FIELD(E)                                                      \
  constexpr E operator|(const E lhs, const E rhs)                              \
  {                                                                            \
    const auto underlying_lhs = std::to_underlying(lhs);                       \
    const auto underlying_rhs = std::to_underlying(rhs);                       \
    return static_cast<E>(underlying_lhs | underlying_rhs);                    \
  }                                                                            \
  constexpr E operator&(const E lhs, const E rhs)                              \
  {                                                                            \
    const auto underlying_lhs = std::to_underlying(lhs);                       \
    const auto underlying_rhs = std::to_underlying(rhs);                       \
    return static_cast<E>(underlying_lhs & underlying_rhs);                    \
  }                                                                            \
  constexpr bool operator!(const E value)                                      \
  {                                                                            \
    return std::to_underlying(value) == 0;                                     \
  }                                                                            \
  constexpr bool operator==(const E lhs, const E rhs)                          \
  {                                                                            \
    return std::to_underlying(lhs) == std::to_underlying(rhs);                 \
  }                                                                            \
  constexpr bool operator!=(const E lhs, const E rhs)                          \
  {                                                                            \
    return std::to_underlying(lhs) != std::to_underlying(rhs);                 \
  }                                                                            \
  constexpr E& operator|=(E& lhs, const E rhs)                                 \
  {                                                                            \
    lhs = lhs | rhs;                                                           \
    return lhs;                                                                \
  }                                                                            \
  constexpr E& operator&=(E& lhs, const E rhs)                                 \
  {                                                                            \
    lhs = lhs & rhs;                                                           \
    return lhs;                                                                \
  }                                                                            \
  constexpr E operator~(const E value)                                         \
  {                                                                            \
    return static_cast<E>(~std::to_underlying(value));                         \
  }

}