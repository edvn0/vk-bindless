#pragma once

#include "vk-bindless/commands.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/handle.hpp"


#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>

namespace VkBindless {

static constexpr auto max_colour_attachments = 8U;

enum class IndexFormat : std::uint8_t
{
  UI8,
  UI16,
  UI32,
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
  std::span<std::byte> data{};

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

struct ICommandBuffer
{
public:
  virtual ~ICommandBuffer() = default;
  /*
    virtual auto transition_to_shader_read_only(TextureHandle surface) const
        -> void = 0;

    virtual auto
    cmd_push_debug_group_label(const char *label,
                               std::uint32_t color_rgba = 0xffffffff) const
        -> void = 0;
    virtual auto
    cmd_insert_debug_event_label(const char *label,
                                 std::uint32_t color_rgba = 0xffffffff) const
        -> void = 0;
    virtual auto cmd_pop_debug_group_label() const -> void = 0;

    virtual auto cmd_bind_compute_pipeline(ComputePipelineHandle handle)
        -> void = 0;
    virtual auto cmd_dispatch_thread_groups(const Dimensions &threadgroup_count,
                                            const Dependencies &deps = {})
        -> void = 0;
*/
  virtual auto cmd_begin_rendering(const RenderPass& render_pass,
                                   const Framebuffer& framebuffer,
                                   const Dependencies& deps) -> void = 0;
  virtual auto cmd_end_rendering() -> void = 0;

  virtual auto cmd_bind_viewport(const Viewport& viewport) -> void = 0;
  virtual auto cmd_bind_scissor_rect(const ScissorRect& rect) -> void = 0;

  virtual auto cmd_bind_graphics_pipeline(GraphicsPipelineHandle handle)
    -> void = 0;
  virtual auto cmd_bind_depth_state(const DepthState& state) -> void = 0;
  virtual auto cmd_draw(std::uint32_t vertex_count,
                        std::uint32_t instance_count,
                        std::uint32_t first_vertex,
                        std::uint32_t base_instance) -> void = 0;
  virtual auto cmd_draw_indexed(std::uint32_t index_count,
                                std::uint32_t instance_count,
                                std::uint32_t first_index,
                                std::int32_t vertex_offset,
                                std::uint32_t base_instance) -> void = 0;

  virtual auto cmd_push_constants(std::span<const std::byte>) -> void = 0;
  template<typename T>
  auto cmd_push_constants(const T& data, std::size_t offset) -> void
  {
    this->cmd_push_constants(std::span<const std::byte>{
      reinterpret_cast<const std::byte*>(&data) + offset,
      sizeof(T) - offset,
    });
  }
  /*
    virtual auto cmd_bind_vertex_buffer(std::uint32_t index,
                                      BufferHandle buffer,
                                      std::uint64_t buffer_offset = 0)
    -> void = 0;
  virtual auto cmd_bind_index_buffer(BufferHandle index_buffer,
                                     IndexFormat index_format,
                                     std::uint64_t index_buffer_offset = 0)
    -> void = 0;

          virtual auto cmd_fill_buffer(BufferHandle buffer, std::size_t
         buffer_offset, std::size_t size, std::uint32_t data)
              -> void = 0;
          virtual auto cmd_update_buffer(BufferHandle buffer, std::size_t
          buffer_offset, std::size_t size, const void *data)
              -> void = 0;
          template <typename Struct>
          auto cmd_update_buffer(BufferHandle buffer, const Struct &data,
                                 std::size_t buffer_offset = 0) -> void {
            this->cmd_update_buffer(buffer, buffer_offset, sizeof(Struct),
     &data);
          }


          virtual auto cmd_draw_indirect(BufferHandle indirect_buffer,
                                         std::size_t indirect_buffer_offset,
                                         std::uint32_t draw_count,
                                         std::uint32_t stride = 0) -> void = 0;
          virtual auto cmd_draw_indexed_indirect(BufferHandle indirect_buffer,
                                                 std::size_t
       indirect_buffer_offset, std::uint32_t draw_count, std::uint32_t stride =
     0)
       -> void = 0; virtual auto cmd_draw_indexed_indirect_count( BufferHandle
          indirect_buffer, std::size_t indirect_buffer_offset, BufferHandle
          count_buffer, std::size_t count_buffer_offset, std::uint32_t
         max_draw_count, std::uint32_t stride = 0) -> void = 0;

          virtual auto cmd_draw_mesh_tasks(const Dimensions &threadgroup_count)
              -> void = 0;
          virtual auto cmd_draw_mesh_tasks_indirect(BufferHandle
     indirect_buffer, std::size_t indirect_buffer_offset, std::uint32_t
     draw_count, std::uint32_t stride = 0)
              -> void = 0;
          virtual auto cmd_draw_mesh_tasks_indirect_count(
              BufferHandle indirect_buffer, std::size_t indirect_buffer_offset,
              BufferHandle count_buffer, std::size_t count_buffer_offset,
              std::uint32_t max_draw_count, std::uint32_t stride = 0) -> void =
     0;

          virtual auto cmd_trace_rays(std::uint32_t width, std::uint32_t height,
                                      std::uint32_t depth = 1,
                                      const Dependencies &deps = {}) -> void =
     0;

          virtual auto cmd_set_blend_color(const float color[4]) -> void = 0;
          virtual auto cmd_set_depth_bias(float constant_factor, float
       slope_factor, float clamp = 0.0f) -> void = 0; virtual auto
       cmd_set_depth_bias_enable(bool enable) -> void = 0;

          virtual auto cmd_reset_query_pool(QueryPoolHandle pool,
                                            std::uint32_t first_query,
                                            std::uint32_t query_count) -> void =
       0; virtual auto cmd_write_timestamp(QueryPoolHandle pool, std::uint32_t
         query)
              -> void = 0;

          virtual auto cmd_clear_color_image(TextureHandle tex,
                                             const ClearColourValue &value,
                                             const TextureLayers &layers = {})
              -> void = 0;
          virtual auto cmd_copy_image(TextureHandle src, TextureHandle dst,
                                      const Dimensions &extent,
                                      const Offset3D &src_offset = {},
                                      const Offset3D &dst_offset = {},
                                      const TextureLayers &src_layers = {},
                                      const TextureLayers &dst_layers = {}) ->
       void = 0; virtual auto cmd_generate_mipmap(TextureHandle handle) -> void
     = 0;
          */
};

class CommandBuffer final : public ICommandBuffer
{
  static constexpr Dependencies empty_deps{};

public:
  CommandBuffer() = default;
  explicit CommandBuffer(IContext&);
  ~CommandBuffer() override;

  [[nodiscard]] auto get_command_buffer() const
  {
    return wrapper->command_buffer;
  }

  auto cmd_begin_rendering(const RenderPass& render_pass,
                           const Framebuffer& framebuffer,
                           const Dependencies& deps) -> void override;
  auto cmd_end_rendering() -> void override;
  auto cmd_bind_viewport(const Viewport& viewport) -> void override;
  auto cmd_bind_scissor_rect(const ScissorRect& rect) -> void override;
  auto cmd_bind_graphics_pipeline(GraphicsPipelineHandle handle)
    -> void override;
  auto cmd_bind_depth_state(const DepthState& state) -> void override;
  auto cmd_draw(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
    -> void override;
  auto cmd_draw_indexed(std::uint32_t,
                        std::uint32_t,
                        std::uint32_t,
                        std::int32_t,
                        std::uint32_t) -> void override;
  auto cmd_push_constants(std::span<const std::byte>) -> void override;

private:
  Context* context{ nullptr };
  const CommandBufferWrapper* wrapper{ nullptr };

  Framebuffer framebuffer = {};
  SubmitHandle last_submit_handle = {};

  VkPipeline last_pipeline_bound = VK_NULL_HANDLE;

  bool is_rendering = false;
  std::uint32_t view_mask = 0;

  GraphicsPipelineHandle current_pipeline_graphics = {};
  ComputePipelineHandle current_pipeline_compute = {};

  friend class Context;
};

} // namespace VkBindless
