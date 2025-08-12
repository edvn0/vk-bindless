#pragma once

#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/handle.hpp"
#include <cstddef>
#include <cstdint>

namespace VkBindless {

static constexpr auto max_colour_attachments = 8U;

enum class IndexFormat : std::uint8_t {
  UI8,
  UI16,
  UI32,
};

enum Topology : uint8_t {
  Point,
  Line,
  LineStrip,
  Triangle,
  TriangleStrip,
  Patch,
};

enum class ColorSpace : std::uint8_t {
  SRGB_NONLINEAR,
  SRGB_EXTENDED_LINEAR,
  HDR10,
  BT709_LINEAR,
};

enum class TextureType : std::uint8_t {
  Two,
  Three,
  Cube,
};

struct Dimensions {
  std::uint32_t width = 1;
  std::uint32_t height = 1;
  std::uint32_t depth = 1;
  inline auto divide1D(std::integral auto v) const {
    return Dimensions{
        .width = width / v,
        .height = height,
        .depth = depth,
    };
  }
  inline auto divide2D(std::integral auto v) const {
    return Dimensions{
        .width = width / v,
        .height = height / v,
        .depth = depth,
    };
  }
  inline auto divide3D(std::integral auto v) const {
    return Dimensions{
        .width = width / v,
        .height = height / v,
        .depth = depth / v,
    };
  }
  inline bool operator==(const Dimensions &other) const {
    return width == other.width && height == other.height &&
           depth == other.depth;
  }
};

enum class CompareOp : std::uint8_t {
  Never = 0,
  Less,
  Equal,
  LessEqual,
  Greater,
  NotEqual,
  GreaterEqual,
  AlwaysPass
};

enum class StencilOp : std::uint8_t {
  Keep = 0,
  Zero,
  Replace,
  IncrementClamp,
  DecrementClamp,
  Invert,
  IncrementWrap,
  DecrementWrap
};

enum class BlendOp : std::uint8_t {
  BlendOp_Add = 0,
  BlendOp_Subtract,
  BlendOp_ReverseSubtract,
  BlendOp_Min,
  BlendOp_Max
};

enum class BlendFactor : std::uint8_t {
  BlendFactor_Zero = 0,
  BlendFactor_One,
  BlendFactor_SrcColor,
  BlendFactor_OneMinusSrcColor,
  BlendFactor_SrcAlpha,
  BlendFactor_OneMinusSrcAlpha,
  BlendFactor_DstColor,
  BlendFactor_OneMinusDstColor,
  BlendFactor_DstAlpha,
  BlendFactor_OneMinusDstAlpha,
  BlendFactor_SrcAlphaSaturated,
  BlendFactor_BlendColor,
  BlendFactor_OneMinusBlendColor,
  BlendFactor_BlendAlpha,
  BlendFactor_OneMinusBlendAlpha,
  BlendFactor_Src1Color,
  BlendFactor_OneMinusSrc1Color,
  BlendFactor_Src1Alpha,
  BlendFactor_OneMinusSrc1Alpha
};

enum class LoadOp : std::uint8_t {
  Invalid = 0,
  DontCare,
  Load,
  Clear,
  None,
};

enum class StoreOp : std::uint8_t {
  DontCare = 0,
  Store,
  MsaaResolve,
  None,
};

enum class ResolveMode : std::uint8_t {
  None = 0,
  SampleZero, // always supported
  Average,
  Min,
  Max,
};

enum class ShaderStage : std::uint8_t {
  Vert,
  Tesc,
  Tese,
  Geom,
  Frag,
  Comp,
  Task,
  Mesh,
  RayGen,
  AnyHit,
  ClosestHit,
  Miss,
  Intersection,
  Callable,
};

struct Dependencies {
  static constexpr auto max_dependencies = 4U;
  std::array<TextureHandle, max_dependencies> textures = {};
  std::array<BufferHandle, max_dependencies> buffers = {};
};

union ClearColorValue {
  std::array<float, 4> float32;
  std::array<std::uint32_t, 4> uint32;
  std::array<std::int32_t, 4> int32;
};

struct RenderPass final {
  struct AttachmentDesc final {
    LoadOp loadOp = LoadOp::Invalid;
    StoreOp storeOp = StoreOp::Store;
    ResolveMode resolveMode = ResolveMode::Average;
    std::uint8_t layer = 0;
    std::uint8_t level = 0;
    ClearColorValue clearColor = {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}};
    float clearDepth = 1.0f;
    uint32_t clearStencil = 0;
  };

  AttachmentDesc color[max_colour_attachments] = {};
  AttachmentDesc depth = {
      .loadOp = LoadOp::DontCare,
      .storeOp = StoreOp::DontCare,
  };
  AttachmentDesc stencil = {
      .loadOp = LoadOp::Invalid,
      .storeOp = StoreOp::DontCare,
  };

  std::uint32_t layer_count = 1;
  std::uint32_t view_mask = 0;

  auto get_colour_attachment_count() const {
    std::uint32_t n = 0;
    while (n < max_colour_attachments && color[n].loadOp != LoadOp::Invalid) {
      n++;
    }
    return n;
  }
};

struct Framebuffer final {
  struct AttachmentDesc {
    TextureHandle texture;
    TextureHandle resolve_texture;
  };

  std::array<AttachmentDesc, max_colour_attachments> color{};
  AttachmentDesc depth_stencil;

  std::string debug_name = "";

  auto get_colour_attachment_count() const {
    std::uint32_t n = 0;
    while (n < max_colour_attachments && color[n].texture) {
      n++;
    }
    return n;
  }
};

struct Viewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
  float minDepth = 1.0f;
  float maxDepth = 0.0f;
};

struct ScissorRect {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct StencilState {
  StencilOp stencil_failure_operation = StencilOp::Keep;
  StencilOp depth_failure_operation = StencilOp::Keep;
  StencilOp depth_stencil_pass_operation = StencilOp::Keep;
  CompareOp stencil_compare_op = CompareOp::AlwaysPass;
  std::uint32_t read_mask = (std::uint32_t)~0;
  std::uint32_t write_mask = (std::uint32_t)~0;
};

struct DepthState {
  CompareOp compare_operation{CompareOp::AlwaysPass};
  bool is_depth_write_enabled{false};
};

struct TextureLayers {
  std::uint32_t mip_level = 0;
  std::uint32_t layer = 0;
  std::uint32_t num_layers = 1;
};

struct Offset3D {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct ICommandBuffer {
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

    virtual auto cmd_begin_rendering(const RenderPass &render_pass,
                                     const Framebuffer &framebuffer,
                                     const Dependencies &deps = {}) -> void = 0;
    virtual auto cmd_end_rendering() -> void = 0;

    virtual auto cmd_bind_viewport(const Viewport &viewport) -> void = 0;
    virtual auto cmd_bind_scissor_rect(const ScissorRect &rect) -> void = 0;

    virtual auto cmd_bind_render_pipeline(GraphicsPipelineHandle handle)
        -> void = 0;
    virtual auto cmd_bind_depth_state(const DepthState &state) -> void = 0;

    virtual auto cmd_bind_vertex_buffer(std::uint32_t index, BufferHandle
    buffer, std::uint64_t buffer_offset = 0)
        -> void = 0;
    virtual auto cmd_bind_index_buffer(BufferHandle index_buffer,
                                       IndexFormat index_format,
                                       std::uint64_t index_buffer_offset = 0)
        -> void = 0;

    virtual auto cmd_push_constants(const void *data, std::size_t size,
                                    std::size_t offset = 0) -> void = 0;
    template <typename Struct>
    auto cmd_push_constants(const Struct &data, std::size_t offset = 0) -> void
    { this->cmd_push_constants(&data, sizeof(Struct), offset);
    }

    virtual auto cmd_fill_buffer(BufferHandle buffer, std::size_t buffer_offset,
                                 std::size_t size, std::uint32_t data)
        -> void = 0;
    virtual auto cmd_update_buffer(BufferHandle buffer, std::size_t
    buffer_offset, std::size_t size, const void *data)
        -> void = 0;
    template <typename Struct>
    auto cmd_update_buffer(BufferHandle buffer, const Struct &data,
                           std::size_t buffer_offset = 0) -> void {
      this->cmd_update_buffer(buffer, buffer_offset, sizeof(Struct), &data);
    }

    virtual auto cmd_draw(std::uint32_t vertex_count,
                          std::uint32_t instance_count = 1,
                          std::uint32_t first_vertex = 0,
                          std::uint32_t base_instance = 0) -> void = 0;
    virtual auto cmd_draw_indexed(std::uint32_t index_count,
                                  std::uint32_t instance_count = 1,
                                  std::uint32_t first_index = 0,
                                  std::int32_t vertex_offset = 0,
                                  std::uint32_t base_instance = 0) -> void = 0;

    virtual auto cmd_draw_indirect(BufferHandle indirect_buffer,
                                   std::size_t indirect_buffer_offset,
                                   std::uint32_t draw_count,
                                   std::uint32_t stride = 0) -> void = 0;
    virtual auto cmd_draw_indexed_indirect(BufferHandle indirect_buffer,
                                           std::size_t indirect_buffer_offset,
                                           std::uint32_t draw_count,
                                           std::uint32_t stride = 0) -> void =
    0; virtual auto cmd_draw_indexed_indirect_count( BufferHandle
    indirect_buffer, std::size_t indirect_buffer_offset, BufferHandle
    count_buffer, std::size_t count_buffer_offset, std::uint32_t max_draw_count,
    std::uint32_t stride = 0) -> void = 0;

    virtual auto cmd_draw_mesh_tasks(const Dimensions &threadgroup_count)
        -> void = 0;
    virtual auto cmd_draw_mesh_tasks_indirect(BufferHandle indirect_buffer,
                                              std::size_t
    indirect_buffer_offset, std::uint32_t draw_count, std::uint32_t stride = 0)
        -> void = 0;
    virtual auto cmd_draw_mesh_tasks_indirect_count(
        BufferHandle indirect_buffer, std::size_t indirect_buffer_offset,
        BufferHandle count_buffer, std::size_t count_buffer_offset,
        std::uint32_t max_draw_count, std::uint32_t stride = 0) -> void = 0;

    virtual auto cmd_trace_rays(std::uint32_t width, std::uint32_t height,
                                std::uint32_t depth = 1,
                                const Dependencies &deps = {}) -> void = 0;

    virtual auto cmd_set_blend_color(const float color[4]) -> void = 0;
    virtual auto cmd_set_depth_bias(float constant_factor, float slope_factor,
                                    float clamp = 0.0f) -> void = 0;
    virtual auto cmd_set_depth_bias_enable(bool enable) -> void = 0;

    virtual auto cmd_reset_query_pool(QueryPoolHandle pool,
                                      std::uint32_t first_query,
                                      std::uint32_t query_count) -> void = 0;
    virtual auto cmd_write_timestamp(QueryPoolHandle pool, std::uint32_t query)
        -> void = 0;

    virtual auto cmd_clear_color_image(TextureHandle tex,
                                       const ClearColorValue &value,
                                       const TextureLayers &layers = {})
        -> void = 0;
    virtual auto cmd_copy_image(TextureHandle src, TextureHandle dst,
                                const Dimensions &extent,
                                const Offset3D &src_offset = {},
                                const Offset3D &dst_offset = {},
                                const TextureLayers &src_layers = {},
                                const TextureLayers &dst_layers = {}) -> void =
    0; virtual auto cmd_generate_mipmap(TextureHandle handle) -> void = 0;
    */
};

class CommandBuffer final : public ICommandBuffer {
public:
  CommandBuffer() = default;
  CommandBuffer(IContext &ctx) : context(&ctx) {}
  ~CommandBuffer() override;

  auto get_command_buffer() const { return wrapper->command_buffer; }

private:
  IContext *context{nullptr};
  const CommandBufferWrapper *wrapper{nullptr};

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
