#include "vk-bindless/imgui_renderer.hpp"

#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/pipeline.hpp"
#include "vk-bindless/swapchain.hpp"

namespace VkBindless {

auto
ImGuiRenderer::create_pipeline(const Framebuffer& fb) const
  -> Holder<GraphicsPipelineHandle>
{
  const std::uint32_t is_non_linear_colour_space =
    context->get_swapchain().surface_format().colorSpace ==
      VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
    context->get_swapchain().surface_format().colorSpace ==
      VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;

  const auto span = std::span{ &is_non_linear_colour_space, 1 };
  return VkGraphicsPipeline::create(context, {
  .shader = *gui_shader,
    .specialisation_constants = {
      .entries={
        SpecialisationConstantDescription::SpecialisationConstantEntry{
          .constant_id = 0,
          .offset = 0,
          .size = sizeof(std::uint32_t),
        },
      },
      .data = std::as_bytes(span),
  },
    .color = {
      ColourAttachment{
        .format = context->get_format(fb.color.at(0).texture),
        .blend_enabled = true,
        .src_rgb_blend_factor = BlendFactor::SrcAlpha,
        .dst_rgb_blend_factor = BlendFactor::OneMinusSrcAlpha,
        },
    },
    .depth_format = fb.depth_stencil.texture.valid() ?
      context->get_format(fb.depth_stencil.texture) :
      Format::Invalid,
    .cull_mode = CullMode::None,
  });
}

 ImGuiRenderer::ImGuiRenderer(IContext& ctx,
                             std::string_view default_font_ttf,
                             float font_size)
                               : context(&ctx)
{
  ImGui::CreateContext();
#if defined(LVK_WITH_IMPLOT)
  ImPlot::CreateContext();
#endif // LVK_WITH_IMPLOT

  ImGuiIO& io = ImGui::GetIO();
  io.BackendRendererName = "imgui-lvk";
  io.BackendFlags |=
    ImGuiBackendFlags_RendererHasVtxOffset;

  update_font(default_font_ttf, font_size);
  gui_shader = VkShader::create(context, "assets/shaders/imgui.shader");
  sampler_clamp_to_edge = VkTextureSampler::create(*context, {
  .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
  .pNext = nullptr,
  .flags = {},
  .magFilter = VK_FILTER_LINEAR,
  .minFilter = VK_FILTER_LINEAR,
  .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
  .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  .mipLodBias = 0.0F,
  .anisotropyEnable = VK_FALSE,
  .maxAnisotropy = 0.0F,
  .compareEnable = VK_FALSE,
  .compareOp = VK_COMPARE_OP_ALWAYS,
  .minLod = 0.0F,
  .maxLod = 1.0F,
  .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
  .unnormalizedCoordinates = VK_FALSE,
  });
}

ImGuiRenderer::~ImGuiRenderer()
{
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->TexID = nullptr;
#if defined(LVK_WITH_IMPLOT)
  ImPlot::DestroyContext();
#endif // LVK_WITH_IMPLOT
  ImGui::DestroyContext();
}

auto ImGuiRenderer::beginFrame(
  const Framebuffer& desc)
{
  const auto& dim = context->get_dimensions(desc.color.at(0).texture);
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(dim.width / display_scale,
                          dim.height / display_scale);
  io.DisplayFramebufferScale = ImVec2(display_scale, display_scale);
  io.IniFilename = nullptr;
  if (graphics_pipeline.empty()) {
    graphics_pipeline = create_pipeline(desc);
  }
  ImGui::NewFrame();
}

void ImGuiRenderer::end_frame(ICommandBuffer& cmdBuffer)
{
  ImGui::EndFrame();
  ImGui::Render();
  ImDrawData* dd = ImGui::GetDrawData();
  const float fb_width =
    dd->DisplaySize.x * dd->FramebufferScale.x;
  const float fb_height =
    dd->DisplaySize.y * dd->FramebufferScale.y;

  cmdBuffer.cmd_bind_depth_state({});
  cmdBuffer.cmd_bind_viewport({
  .x = 0.0f,
  .y = 0.0f,
  .width =   fb_width ,
  .height = fb_height,
  });

  const float L = dd->DisplayPos.x;
  const float R = dd->DisplayPos.x + dd->DisplaySize.x;
  const float T = dd->DisplayPos.y;
  const float B = dd->DisplayPos.y + dd->DisplaySize.y;
  const ImVec2 clipOff = dd->DisplayPos;
  const ImVec2 clipScale = dd->FramebufferScale;

  auto& drawableData = drawables.at(frame_index);
  frame_index = (frame_index + 1) % std::size(drawables);


  if (drawableData.allocated_indices < dd->TotalIdxCount)
  {
    drawableData.index_buffer = IndexBuffer::create(context, {
      .usage = BufferUsageBits::Index,
      .storage = StorageType::HostVisible,
      .size = dd->TotalIdxCount * sizeof(ImDrawIdx),
      .debugName = "ImGui: drawable_data.index_buffer",
    });
    drawableData.allocated_indices = dd->TotalIdxCount;
  }

  if (drawableData.allocated_vertices < dd->TotalVtxCount)
  {
    drawableData.vertex_buffer = VertexBuffer::create(context, {
      .usage = BufferUsageBits::Storage,
      .storage = StorageType::HostVisible,
      .size = dd->TotalVtxCount * sizeof(ImDrawVert),
      .debugName = "ImGui: drawableData.vb_",
    });
    drawableData.allocated_vertices = dd->TotalVtxCount;
  }

  auto* vtx = context->get_mapped_pointer<ImDrawVert*>(drawableData.vertex_buffer);
  auto* idx = context->get_mapped_pointer<(uint16_t*)>(drawableData.index_buffer);
  for (int n = 0; n < dd->CmdListsCount; n++) {
    const ImDrawList* cmdList = dd->CmdLists[n];
    std::memcpy(vtx, cmdList->VtxBuffer.Data,
      cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
    std::memcpy(idx, cmdList->IdxBuffer.Data,
      cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
    vtx += cmdList->VtxBuffer.Size;
    idx += cmdList->IdxBuffer.Size;
  }

  context->flush_mapped_memory(drawableData.vertex_buffer, 0, dd->TotalVtxCount * sizeof(ImDrawVert));
  context->flush_mapped_memory(drawableData.index_buffer, 0, dd->TotalIdxCount * sizeof(ImDrawIdx));

  uint32_t idxOffset = 0;
  uint32_t vtxOffset = 0;
  cmdBuffer.cmd_bind_index_buffer(
    drawableData.index_buffer, IndexFormat::UI16);
  cmdBuffer.cmd_bind_graphics_pipeline(*gui_shader);
  for (int n = 0; n < dd->CmdListsCount; n++) {
    const ImDrawList* cmdList = dd->CmdLists[n];
    for (int cmd_i = 0; cmd_i < cmdList->CmdBuffer.Size; cmd_i++) {
      const ImDrawCmd& cmd = cmdList->CmdBuffer[cmd_i];
      ImVec2 clipMin(
        (cmd.ClipRect.x - clipOff.x) * clipScale.x,
        (cmd.ClipRect.y - clipOff.y) * clipScale.y);
      ImVec2 clipMax(
        (cmd.ClipRect.z - clipOff.x) * clipScale.x,
        (cmd.ClipRect.w - clipOff.y) * clipScale.y);
      if (clipMin.x < 0.0f) clipMin.x = 0.0f;
      if (clipMin.y < 0.0f) clipMin.y = 0.0f;
      if (clipMax.x > fb_width ) clipMax.x = fb_width;
      if (clipMax.y > fb_height) clipMax.y = fb_height;
      if (clipMax.x <= clipMin.x ||
          clipMax.y <= clipMin.y) continue;
      struct VulkanImguiBindData {
        float LRTB[4];
        uint64_t vb = 0;
        uint32_t textureId = 0;
        uint32_t samplerId = 0;
      } bindData = {
        .LRTB = {L, R, T, B},
        .vb = context->get_device_address(drawableData.vertex_buffer),
        .textureId = static_cast<uint32_t>(cmd.TextureId),
        .samplerId = sampler_clamp_to_edge.index(),
      };
      cmdBuffer.cmd_push_constants<VulkanImguiBindData>(bindData, 0);
      cmdBuffer.cmd_bind_scissor_rect({
      uint32_t(clipMin.x),
      uint32_t(clipMin.y),
      uint32_t(clipMax.x - clipMin.x),
      uint32_t(clipMax.y - clipMin.y)});
      cmdBuffer.cmd_draw_indexed(cmd.ElemCount, 1u,
      idxOffset + cmd.IdxOffset,
      int32_t(vtxOffset + cmd.VtxOffset));
    }
    idxOffset += cmdList->IdxBuffer.Size;
    vtxOffset += cmdList->VtxBuffer.Size;
  }

}
auto
ImGuiRenderer::update_font(std::string_view ttf_path, float font_size_pixels) -> void
{
  auto& io = ImGui::GetIO();
  ImFontConfig cfg = ImFontConfig();
  cfg.FontDataOwnedByAtlas = false;
  cfg.RasterizerMultiply = 1.5f;
  cfg.SizePixels = std::ceilf(font_size_pixels);
  cfg.PixelSnapH = true;
  cfg.OversampleH = 4;
  cfg.OversampleV = 4;
  ImFont* font = nullptr;
  if (!ttf_path.empty()) {
    font = io.Fonts->AddFontFromFileTTF(
      ttf_path.data(), cfg.SizePixels, &cfg);
  }
  io.Fonts->Flags |= ImFontAtlasFlags_NoPowerOfTwoHeight;

  unsigned char* pixels;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  font_texture = VkTexture::create(*context, {
  .data = {pixels, static_cast<std::size_t>(width * height * 4)},
  .format = Format::RGBA_UN8,
  .extent = {
    .width = static_cast<std::uint32_t>(width),
    .height = static_cast<std::uint32_t>(height),
    .depth = 1,
  },
  .usage_flags = TextureUsageFlags::Sampled |
    TextureUsageFlags::TransferDestination,
  .layers = 1,
  .mip_levels = 1, // No mipmaps for font texture
  .sample_count = VK_SAMPLE_COUNT_1_BIT,
  .tiling = VK_IMAGE_TILING_OPTIMAL,
  .debug_name ="ImGui Font Texture",
  });
  io.Fonts->TexID = font_texture.index();
  io.FontDefault = font;
}

}