#include "vk-bindless/buffer.hpp"
#include "vk-bindless/camera.hpp"
#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/common.hpp"
#include "vk-bindless/container.hpp"
#include "vk-bindless/event_system.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/imgui_renderer.hpp"
#include "vk-bindless/line_canvas.hpp"
#include "vk-bindless/mesh.hpp"
#include "vk-bindless/pipeline.hpp"
#include "vk-bindless/scope_exit.hpp"
#include "vk-bindless/shader.hpp"
#include "vk-bindless/transitions.hpp"
#include "vk-bindless/vulkan_context.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <imgui.h>
#include <thread>

#include "vk-bindless/file_watcher.hpp"

#define GLFW_INCLUDE_VULKAN
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/random.hpp"
#include "helper.hpp"
#include "implot.h"
#include <GLFW/glfw3.h>
#include <bit>
#include <cstdint>
#include <format>
#include <glm/glm.hpp>
#include <iostream>
#include <ktx.h>

static auto
destroy_glfw(GLFWwindow* window) -> void
{
  if (window) {
    glfwDestroyWindow(window);
    glfwTerminate();
  }
}

struct WindowState
{
  std::int32_t windowed_x{};
  std::int32_t windowed_y{};
  std::int32_t windowed_width{};
  std::int32_t windowed_height{};
  bool fullscreen{ false };
};

static auto
is_wayland() -> bool
{
  const int platform = glfwGetPlatform();
  return platform == GLFW_PLATFORM_WAYLAND;
}

bool
depth_state_widget(const char* label, VkBindless::DepthState& depth_state)
{
  bool changed = false;

  if (ImGui::TreeNode(label)) {
    // Compare operation dropdown
    const char* compare_op_names[] = { "Never",        "Less",      "Equal",
                                       "LessEqual",    "Greater",   "NotEqual",
                                       "GreaterEqual", "AlwaysPass" };

    int current_op = static_cast<int>(depth_state.compare_operation);
    if (ImGui::Combo("Compare Operation",
                     &current_op,
                     compare_op_names,
                     IM_ARRAYSIZE(compare_op_names))) {
      depth_state.compare_operation =
        static_cast<VkBindless::CompareOp>(current_op);
      changed = true;
    }

    // Depth test enabled checkbox
    if (ImGui::Checkbox("Depth Test Enabled",
                        &depth_state.is_depth_test_enabled)) {
      changed = true;
    }

    // Depth write enabled checkbox
    if (ImGui::Checkbox("Depth Write Enabled",
                        &depth_state.is_depth_write_enabled)) {
      changed = true;
    }

    ImGui::TreePop();
  }

  return changed;
}

class WindowManager final
  : public EventSystem::TypedEventHandler<EventSystem::KeyEvent,
                                          EventSystem::WindowResizeEvent>
{
  GLFWwindow* window;
  WindowState* window_state;

public:
  WindowManager(GLFWwindow* w, WindowState* state)
    : window(w)
    , window_state(state)
  {
  }
  [[nodiscard]] auto get_priority() const -> int override { return 1000; }

protected:
  bool handle_event(const EventSystem::KeyEvent& event) override
  {
    if (event.action != GLFW_PRESS)
      return false;
    if (event.key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      return true;
    }
    if (event.key == GLFW_KEY_F11) {
      toggle_fullscreen();
      return true;
    }
    return false;
  }
  bool handle_event(const EventSystem::WindowResizeEvent& event) override
  {
    std::cout << std::format(
      "Window resized to {}x{}\n", event.width, event.height);
    return false;
  }

private:
  auto toggle_fullscreen() -> void
  {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (!window_state->fullscreen) {
      if (!is_wayland()) {
        glfwGetWindowPos(
          window, &window_state->windowed_x, &window_state->windowed_y);
      }
      glfwGetWindowSize(
        window, &window_state->windowed_width, &window_state->windowed_height);
      glfwSetWindowMonitor(
        window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
      window_state->fullscreen = true;
    } else {
      if (is_wayland()) {
        glfwSetWindowMonitor(window,
                             nullptr,
                             0,
                             0,
                             window_state->windowed_width,
                             window_state->windowed_height,
                             0);
      } else {
        glfwSetWindowMonitor(window,
                             nullptr,
                             window_state->windowed_x,
                             window_state->windowed_y,
                             window_state->windowed_width,
                             window_state->windowed_height,
                             0);
      }
      window_state->fullscreen = false;
    }
  }
};

class GameLogicHandler final : public EventSystem::EventHandler
{
public:
  [[nodiscard]] auto get_priority() const -> int override { return 100; }

protected:
  bool handle_event(const EventSystem::KeyEvent&) override { return false; }
  bool handle_event(const EventSystem::MouseButtonEvent& event) override
  {
    if (event.action == GLFW_PRESS && event.button == GLFW_MOUSE_BUTTON_LEFT) {
      return true;
    }
    return false;
  }
  bool handle_event(const EventSystem::MouseMoveEvent&) override
  {
    return false; // Don't consume mouse movement
  }
};

class CameraInputHandler final
  : public EventSystem::TypedEventHandler<EventSystem::KeyEvent,
                                          EventSystem::MouseMoveEvent,
                                          EventSystem::MouseButtonEvent>
{
  GLFWwindow* window{};
  VkBindless::FirstPersonCameraBehaviour* behaviour{};
  bool mouse_held{ false };
  glm::vec2 mouse_norm{ 0 };

public:
  explicit CameraInputHandler(GLFWwindow* win,
                              VkBindless::FirstPersonCameraBehaviour* b)
    : window(win)
    , behaviour(b)
  {
  }
  [[nodiscard]] auto get_priority() const -> int override { return 800; }

protected:
  auto handle_event(const EventSystem::KeyEvent& e) -> bool override
  {
    if (ImGui::GetIO().WantCaptureKeyboard)
      return false;
    const bool pressed = e.action != GLFW_RELEASE;
    switch (e.key) {
      case GLFW_KEY_W:
        behaviour->movement.forward = pressed;
        break;
      case GLFW_KEY_S:
        behaviour->movement.backward = pressed;
        break;
      case GLFW_KEY_A:
        behaviour->movement.left = pressed;
        break;
      case GLFW_KEY_D:
        behaviour->movement.right = pressed;
        break;
      case GLFW_KEY_E:
        behaviour->movement.up = pressed;
        break;
      case GLFW_KEY_Q:
        behaviour->movement.down = pressed;
        break;
      case GLFW_KEY_LEFT_SHIFT:
        behaviour->movement.fast_speed = pressed;
        break;
      default:
        break;
    }
    return false;
  }
  auto handle_event(const EventSystem::MouseButtonEvent& e) -> bool override
  {
    if (ImGui::GetIO().WantCaptureMouse)
      return false;
    if (e.button == GLFW_MOUSE_BUTTON_RIGHT) {
      mouse_held = (e.action == GLFW_PRESS);
      glfwSetInputMode(window,
                       GLFW_CURSOR,
                       mouse_held ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
      if (mouse_held)
        behaviour->mouse_position = mouse_norm;
    }
    return mouse_held;
  }
  auto handle_event(const EventSystem::MouseMoveEvent& e) -> bool override
  {
    int w{}, h{};
    glfwGetFramebufferSize(window, &w, &h);
    if (w > 0 && h > 0) {
      mouse_norm = { static_cast<float>(e.x_pos) / static_cast<float>(w),
                     1.0f -
                       static_cast<float>(e.y_pos) / static_cast<float>(h) };
    }
    return mouse_held;
  }

public:
  auto tick(const double dt) const -> void
  {
    const bool imgui_block = ImGui::GetIO().WantCaptureMouse;
    behaviour->update(dt, mouse_norm, mouse_held && !imgui_block);
  }
};

class UIHandler final
  : public EventSystem::TypedEventHandler<EventSystem::KeyEvent,
                                          EventSystem::MouseButtonEvent>
{
public:
  [[nodiscard]] auto get_priority() const -> int override { return 50; }

protected:
  bool handle_event(const EventSystem::KeyEvent& event) override
  {
    if (event.action == GLFW_PRESS) {
      if (event.key == GLFW_KEY_TAB) {
        std::cout << "Toggle UI panel\n";
        return true;
      }
      if (event.key == GLFW_KEY_I) {
        std::cout << "Open inventory\n";
        return true;
      }
    }
    return false;
  }
  bool handle_event(const EventSystem::MouseButtonEvent& event) override
  {
    if (event.action == GLFW_PRESS && event.button == GLFW_MOUSE_BUTTON_RIGHT) {
      std::cout << "Context menu\n";
      return true;
    }
    return false;
  }
};

template<std::size_t Count>
struct FrameUniform
{
  std::array<VkBindless::Holder<VkBindless::BufferHandle>, Count> buffers{};
  static auto create(VkBindless::IContext& context,
                     const std::span<const std::byte> data)
  {
    const auto size = data.size_bytes();
    if (size == 0 || size % 16 != 0) {
      throw std::runtime_error(
        "FrameUniform data must be a non-zero multiple of 16 bytes");
    }
    FrameUniform uniform;
    const VkBindless::BufferDescription desc{
      .data = data,
      .size = size,
      .storage = VkBindless::StorageType::HostVisible,
      .usage = VkBindless::BufferUsageFlags::UniformBuffer,
      .debug_name = "FrameUniform Buffer",
    };
    for (auto& buf : uniform.buffers) {
      buf = VkBindless::VkDataBuffer::create(context, desc);
    }
    return uniform;
  }

  template<typename T>
  auto upload(VkBindless::IContext& context, const std::span<const T> data)
    -> void
  {
    upload(context, std::as_bytes(data));
  }
  template<typename T>
  auto upload(VkBindless::IContext& context, const std::span<T> data) -> void
  {
    upload(context, std::as_bytes(data));
  }

  auto upload(VkBindless::IContext& context,
              const std::span<const std::byte> data) -> void
  {
    auto index = context.get_frame_index() % Count;
    auto handle = *buffers[index];
    auto* ptr = context.get_mapped_pointer(handle);
    if (!ptr) {
      throw std::runtime_error("FrameUniform buffer is not mapped");
    }
    std::memcpy(ptr, data.data(), data.size_bytes());
    context.flush_mapped_memory(handle, 0, data.size_bytes());
  }
  [[nodiscard]] auto get_address(VkBindless::IContext& context)
  {
    auto index = context.get_swapchain().current_frame_index() % Count;
    return context.get_device_address(*buffers[index]);
  }
  [[nodiscard]] auto at(std::uint32_t index)
    -> VkBindless::Holder<VkBindless::BufferHandle>&
  {
    if (index >= Count)
      throw std::out_of_range("FrameUniform index out of range");
    return buffers[index];
  }
};

static auto
setup_event_callbacks(GLFWwindow* window, EventSystem::EventDispatcher* d)
  -> void
{
  glfwSetWindowUserPointer(window, d);
  glfwSetKeyCallback(
    window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
      auto* dispatcher = static_cast<EventSystem::EventDispatcher*>(
        glfwGetWindowUserPointer(win));
      dispatcher->handle_key_callback(win, key, scancode, action, mods);
      auto& io = ImGui::GetIO();
      io.AddKeyEvent(glfw_key_to_imgui_key(key), action != GLFW_RELEASE);
    });
  glfwSetMouseButtonCallback(
    window, [](GLFWwindow* win, int button, int action, int mods) {
      auto* dispatcher = static_cast<EventSystem::EventDispatcher*>(
        glfwGetWindowUserPointer(win));
      dispatcher->handle_mouse_button_callback(win, button, action, mods);
      double xpos, ypos;
      glfwGetCursorPos(win, &xpos, &ypos);
      const ImGuiMouseButton_ imgui_button =
        (button == GLFW_MOUSE_BUTTON_LEFT)
          ? ImGuiMouseButton_Left
          : (button == GLFW_MOUSE_BUTTON_RIGHT ? ImGuiMouseButton_Right
                                               : ImGuiMouseButton_Middle);
      auto& io = ImGui::GetIO();
      io.AddMouseButtonEvent(imgui_button, action != GLFW_RELEASE);
    });
  glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
    auto* dispatcher =
      static_cast<EventSystem::EventDispatcher*>(glfwGetWindowUserPointer(win));
    dispatcher->handle_cursor_pos_callback(win, x, y);
    ImGui::GetIO().AddMousePosEvent(static_cast<float>(x),
                                    static_cast<float>(y));
  });
  glfwSetScrollCallback(window,
                        [](GLFWwindow*, double xoffset, double yoffset) {
                          ImGuiIO& io = ImGui::GetIO();
                          io.AddMouseWheelEvent(static_cast<float>(xoffset),
                                                static_cast<float>(yoffset));
                        });
  glfwSetWindowSizeCallback(window, [](GLFWwindow* win, int width, int height) {
    auto* dispatcher =
      static_cast<EventSystem::EventDispatcher*>(glfwGetWindowUserPointer(win));
    dispatcher->handle_window_size_callback(win, width, height);
  });
}

template<typename K>
static constexpr auto
as_null(K* k = VK_NULL_HANDLE) -> decltype(k)
{
  return static_cast<decltype(k)>(VK_NULL_HANDLE);
}

auto
compute_and_cache_brdf(auto& context)
{
  using namespace VkBindless;
  auto brdf_lut_compute_shader =
    *VkShader::create(&context, "assets/shaders/brdf_lut_compute.shader");

  constexpr auto brdf_width = 512;
  constexpr auto brdf_height = 512;
  constexpr auto brdf_monte_carlo_sample_count = 1024;
  constexpr auto brdf_buffer_size =
    4ULL * sizeof(std::uint16_t) * brdf_width * brdf_height;
  (void)brdf_buffer_size;

  auto span = std::span{ &brdf_monte_carlo_sample_count, 1 };

  auto brdf_lut_pipeline = VkComputePipeline::create(
      &context,
        {
          .shader = *brdf_lut_compute_shader,
          .specialisation_constants = {
            .entries =
            {
              SpecialisationConstantDescription::SpecialisationConstantEntry {
               .constant_id = 0,
               .offset = 0,
               .size = sizeof(brdf_monte_carlo_sample_count),
             },
            },
            .data = std::as_bytes(span),
            },
          .entry_point = "main",
          .debug_name = "BRDF LUT Compute Pipeline",
        });

  auto buffer = VkDataBuffer::create(
    context,
    {
      .data = {},
      .size = brdf_buffer_size,
      .storage = StorageType::DeviceLocal,
      .usage = BufferUsageFlags::StorageBuffer | BufferUsageFlags::TransferDst,
      .debug_name = "BRDF LUT Buffer",
    });

  auto& buf = context.acquire_command_buffer();
  buf.cmd_bind_compute_pipeline(*brdf_lut_pipeline);
  struct
  {
    std::uint32_t w = brdf_width;
    std::uint32_t h = brdf_height;
    std::uint64_t addr;
  } pc{
    .addr = context.get_device_address(*buffer),
  };
  buf.cmd_push_constants(pc, 0);
  buf.cmd_dispatch_thread_groups({ brdf_width / 16, brdf_height / 16, 1 });

  context.wait_for(*context.submit(buf, {}));

  std::vector<ktx_uint8_t> bytes(brdf_buffer_size);
  std::memcpy(
    bytes.data(), context.get_mapped_pointer(*buffer), brdf_buffer_size);

  ktxTextureCreateInfo ci = {
    .glInternalformat = {},
    .vkFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
    .pDfd = {},
    .baseWidth = brdf_width,
    .baseHeight = brdf_height,
    .baseDepth = 1,
    .numDimensions = 2,
    .numLevels = 1,
    .numLayers = 1,
    .numFaces = 1,
    .isArray = KTX_FALSE,
    .generateMipmaps = KTX_FALSE,
  };

  ktxTexture2* tex = nullptr;
  {
    KTX_error_code rc =
      ktxTexture2_Create(&ci, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &tex);
    if (rc != KTX_SUCCESS) {
      std::cerr << "Could not create image.\n";
    }
  }

  // 3) Upload the level-0 image from memory (tight packing expected).
  {
    const ktx_uint32_t level = 0, layer = 0, faceSlice = 0;
    KTX_error_code rc =
      ktxTexture_SetImageFromMemory(ktxTexture(tex),
                                    level,
                                    layer,
                                    faceSlice,
                                    bytes.data(),
                                    static_cast<ktx_size_t>(bytes.size()));
    if (rc != KTX_SUCCESS) {
      std::cerr << "Could not set image data from memory.\n";
    }
  }

  // (Optional but nice for HDR data) declare linear transfer function.
  ktxTexture2_SetOETF(tex, KHR_DF_TRANSFER_LINEAR);

  // 4) Write out a KTX2 file and clean up.
  {
    bool create = true;
    if (!std::filesystem::is_directory("data") &&
        !std::filesystem::create_directory("data")) {
      std::cerr << "Could not create the directory, or find it.\n";
      create = false;
    }

    if (create) {

      KTX_error_code rc =
        ktxTexture2_WriteToNamedFile(tex, "data/brdfLUT.ktx2");
      if (rc != KTX_SUCCESS) {
        std::cerr << "Could not write to folder 'data' with the created file\n";
      }
    }
  }
  ktxTexture2_Destroy(tex);
}

void
draw_compact_wasd_qe_widget()
{
  ImGui::Begin("Compact WASD+QE");

  // Simple text-based display
  ImGui::Text("Keys: %s%s%s%s%s%s",
              ImGui::IsKeyDown(ImGuiKey_W) ? "[W]" : "W",
              ImGui::IsKeyDown(ImGuiKey_A) ? "[A]" : "A",
              ImGui::IsKeyDown(ImGuiKey_S) ? "[S]" : "S",
              ImGui::IsKeyDown(ImGuiKey_D) ? "[D]" : "D",
              ImGui::IsKeyDown(ImGuiKey_Q) ? "[Q]" : "Q",
              ImGui::IsKeyDown(ImGuiKey_E) ? "[E]" : "E");

  ImGui::End();
}

auto
run_main(WindowState& state, GLFWwindow* window, VkBindless::IContext& context)
  -> void
{
  using namespace VkBindless;

  // MeshFile::preload_mesh("assets/meshes/duck.glb");
  // auto duck_model_file =
  //   *MeshFile::create(context, "assets/.mesh_cache/duck.glb");
  // VkMesh duck_model{ context, duck_model_file };

  MeshFile::preload_mesh("assets/meshes/bistro_interior.glb");
  auto duck_model_file =
    *MeshFile::create(context, "assets/.mesh_cache/bistro_interior.glb");
  VkMesh duck_model{ context, duck_model_file };

  // auto duck_model = *Model::create(context, "");

  glfwSetWindowUserPointer(window, &state);
  EventSystem::EventDispatcher event_dispatcher;
  setup_event_callbacks(window, &event_dispatcher);

  // Create and register event handlers
  const auto window_manager = std::make_shared<WindowManager>(window, &state);
  const auto game_handler = std::make_shared<GameLogicHandler>();
  const auto ui_handler = std::make_shared<UIHandler>();

  Camera camera(
    std::make_unique<FirstPersonCameraBehaviour>(glm::vec3{ 0, 2.0F, -3.0F },
                                                 glm::vec3{ 0, 0, 0.0F },
                                                 glm::vec3{ 0, 1, 0 }));

  const auto camera_input = std::make_shared<CameraInputHandler>(
    window, dynamic_cast<FirstPersonCameraBehaviour*>(camera.get_behaviour()));

  event_dispatcher.subscribe<EventSystem::KeyEvent,
                             EventSystem::MouseMoveEvent,
                             EventSystem::MouseButtonEvent>(camera_input);
  event_dispatcher
    .subscribe<EventSystem::WindowResizeEvent, EventSystem::KeyEvent>(
      window_manager);
  event_dispatcher.subscribe<EventSystem::MouseMoveEvent,
                             EventSystem::KeyEvent,
                             EventSystem::MouseButtonEvent>(game_handler);
  event_dispatcher.subscribe<EventSystem::KeyEvent>(ui_handler);

  compute_and_cache_brdf(context);

  std::int32_t new_width = 0;
  std::int32_t new_height = 0;

  // MSAA sample count
  constexpr VkSampleCountFlagBits kMsaa = VK_SAMPLE_COUNT_1_BIT;

  VertexInput static_opaque_geometry_vertex_input = VertexInput::create({
    VertexFormat::Float3,             // position
    VertexFormat::Int_2_10_10_10_REV, // normal+roughness
    VertexFormat::HalfFloat2,         // uvs
    VertexFormat::Int_2_10_10_10_REV, // tangent+handedness
  });
  auto opaque_geometry =
    *VkShader::create(&context, "assets/shaders/opaque_geometry.shader");
  uint32_t uses_ssbo = 1;
  auto geometry_ssbo = VkGraphicsPipeline::create(
    &context,
    {
      .vertex_input = static_opaque_geometry_vertex_input,
      .shader = *opaque_geometry,
      .specialisation_constants = {
      .entries={
        SpecialisationConstantDescription::SpecialisationConstantEntry{
          .constant_id = 0,
          .offset = 0,
          .size = sizeof(std::uint32_t),
        },
      },
      .data = std::as_bytes(std::span{&uses_ssbo, 1}),
  },
      .color = { 
                 ColourAttachment{
                   .format = Format::RG_F16, //UVs
                 },
                 ColourAttachment{
                   .format = Format::RGBA_F16, // Normal roughness
                 },
                  ColourAttachment{
                   .format = Format::RGBA_UI16, // texture indices (albedo, normal, roughness, metallic)
                 },
                 },
      .depth_format = Format::Z_F32,
      .cull_mode = CullMode::Back,
      .winding = WindingMode::CW,
      .sample_count = kMsaa, // ensure pipeline is compatible with MSAA target
      .debug_name = "Static Opaque Pipeline",
    });
  uses_ssbo = 0;
  auto geometry_pc = VkGraphicsPipeline::create(
    &context,
    {
      .vertex_input = static_opaque_geometry_vertex_input,
      .shader = *opaque_geometry,
      .specialisation_constants = {
      .entries={
        SpecialisationConstantDescription::SpecialisationConstantEntry{
          .constant_id = 0,
          .offset = 0,
          .size = sizeof(std::uint32_t),
        },
      },
      .data = std::as_bytes(std::span{&uses_ssbo, 1}),
  },
      .color = { 
                 ColourAttachment{
                   .format = Format::RG_F16, //UVs
                 },
                 ColourAttachment{
                   .format = Format::RGBA_F16, // Normal roughness
                 },
                  ColourAttachment{
                   .format = Format::RGBA_UI16, // texture indices (albedo, normal, roughness, metallic)
                 },
                 },
      .depth_format = Format::Z_F32,
      .cull_mode = CullMode::Back,
      .winding = WindingMode::CW,
      .sample_count = kMsaa, // ensure pipeline is compatible with MSAA target
      .debug_name = "Static Opaque Pipeline",
    });

  context.on_shader_changed("assets/shaders/opaque_geometry.shader",
                            *geometry_ssbo);
  context.on_shader_changed("assets/shaders/opaque_geometry.shader",
                            *geometry_pc);

  auto lighting_shader =
    *VkShader::create(&context, "assets/shaders/lighting_gbuffer.shader");
  auto lighting_pipeline = VkGraphicsPipeline::create(
    &context,
    {
      .vertex_input =
        VertexInput::create({}), 
      .shader = *lighting_shader, 
      .color = { ColourAttachment{
        .format = Format::RGBA_F32, }, },
      .depth_format = Format::Invalid,  
      .cull_mode = CullMode::None,      
      .sample_count = VK_SAMPLE_COUNT_1_BIT, 
      .debug_name = "Lighting Pipeline",
    });
  context.on_shader_changed("assets/shaders/lighting_gbuffer.shader",
                            *lighting_pipeline);

  constexpr auto null_k_bytes = [](auto k = 0) {
    return std::vector(k, std::byte{ 0 });
  };

  struct UBO
  {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 camera_position;
    glm::vec4 light_direction;
    std::uint32_t texture;
    std::uint32_t cube_texture;
    std::uint64_t padding{ 0 };
  };

  static constexpr auto align_size = [](std::size_t size,
                                        std::size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
  };

  // Setup initial extent & create offscreen resources
  VkExtent3D offscreen_extent{
    .width = static_cast<std::uint32_t>(state.windowed_width),
    .height = static_cast<std::uint32_t>(state.windowed_height),
    .depth = 1,
  };

  auto null_ubo = null_k_bytes(align_size(sizeof(UBO), 16));
  auto main_ubo = FrameUniform<3>::create(context, null_ubo);

  // Create ImGui renderer
  auto imgui =
    std::make_unique<ImGuiRenderer>(context, "assets/fonts/Roboto-Regular.ttf");
  LineCanvas3D canvas_3d;

  //  auto bistro_model =
  //    *Model::create(context, "assets/meshes/bistro_interior.glb");

  Holder<TextureHandle> color_resolved{};
  Holder<TextureHandle> g_uvs; // texture uvs
  Holder<TextureHandle> g_normal_rough;
  Holder<TextureHandle> g_texture_indices; // world pos
  Holder<TextureHandle> g_depth;           // depth (Z_F32)

  auto create_offscreen_targets = [&](std::uint32_t w, std::uint32_t h) {
    offscreen_extent.width = w;
    offscreen_extent.height = h;

    // Resolved single-sample color (sampled to be used by post shader)
    color_resolved =
      VkTexture::create(context,
                        {
                          .format = Format::RGBA_F32,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,
                                      .depth = 1 },
                          .usage_flags = TextureUsageFlags::ColourAttachment |
                                         TextureUsageFlags::Sampled,
                          .mip_levels = 1,
                          .debug_name = "Offscreen Color Resolved",
                        });

    g_uvs =
      VkTexture::create(context,
                        { .format = Format::RG_F16,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,.depth = 1, },
                          .usage_flags = TextureUsageFlags::ColourAttachment |
                                         TextureUsageFlags::Sampled,
                          .layers = 1,
                          .mip_levels = 1,
                          .sample_count = VK_SAMPLE_COUNT_1_BIT,
                          .debug_name = "GBuffer UVs" });
    g_texture_indices =
      VkTexture::create(context,
                        { .format = Format::RGBA_UI16,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,.depth = 1, },
                          .usage_flags = TextureUsageFlags::ColourAttachment |
                                         TextureUsageFlags::Sampled,
                          .layers = 1,
                          .mip_levels = 1,
                          .sample_count = VK_SAMPLE_COUNT_1_BIT,
                          .debug_name = "GBuffer Texture Indices" });
    g_normal_rough =
      VkTexture::create(context,
                        { .format = Format::RGBA_F16,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,.depth = 1, },
                          .usage_flags = TextureUsageFlags::ColourAttachment |
                                         TextureUsageFlags::Sampled,
                          .layers = 1,
                          .mip_levels = 1,
                          .sample_count = VK_SAMPLE_COUNT_1_BIT,
                          .debug_name = "GBuffer NormalRoughness" });
    g_depth = VkTexture::create(
      context,
      { .format = Format::Z_F32,
        .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,.depth = 1, },
        .usage_flags = TextureUsageFlags::DepthStencilAttachment |
                       TextureUsageFlags::Sampled,
        .layers = 1,
        .mip_levels = 1,
        .sample_count = VK_SAMPLE_COUNT_1_BIT,
        .debug_name = "GBuffer Depth" });
  };

  // initial create
  create_offscreen_targets(static_cast<std::uint32_t>(state.windowed_width),
                           static_cast<std::uint32_t>(state.windowed_height));
  // Post pipeline & shader (samples resolved offscreen texture by index via
  // push constants)
  auto post_shader = *VkShader::create(&context, "assets/shaders/post.shader");

  auto post_pipeline = VkGraphicsPipeline::create(
    &context,
    {
      .shader = *post_shader,
      .color = { ColourAttachment{ .format = Format::BGRA_UN8 } },
      .depth_format = Format::Z_F32,
      .sample_count = VK_SAMPLE_COUNT_1_BIT,
      .debug_name = "Post Pipeline",
    });
  context.on_shader_changed("assets/shaders/post.shader", *post_pipeline);

  auto grid_shader = *VkShader::create(&context, "assets/shaders/grid.shader");
  auto grid_pipeline = VkGraphicsPipeline::create(
    &context,
    {
      .shader = *grid_shader,
      .color = { ColourAttachment{
        .format = Format::RGBA_F32,
        .blend_enabled = true,
        .src_rgb_blend_factor = BlendFactor::SrcAlpha,
        .dst_rgb_blend_factor = BlendFactor::OneMinusSrcAlpha,
      } },
      .depth_format = Format::Z_F32,
      .sample_count = kMsaa,
      .debug_name = "Grid Pipeline",
    });
  context.on_shader_changed("assets/shaders/grid.shader", *grid_pipeline);

  double last_time = glfwGetTime();

  auto ensure_size = [&](int w, int h) {
    if (w <= 0 || h <= 0)
      return;
    if (offscreen_extent.width != static_cast<std::uint32_t>(w) ||
        offscreen_extent.height != static_cast<std::uint32_t>(h)) {
      create_offscreen_targets(static_cast<std::uint32_t>(w),
                               static_cast<std::uint32_t>(h));
    }
  };

  constexpr DepthState gbuffer_depth_state{
    .compare_operation = CompareOp::Greater,
    .is_depth_test_enabled = true,
    .is_depth_write_enabled = true,
  };

  while (!glfwWindowShouldClose(window)) {
    event_dispatcher.process_events();
    const double now = glfwGetTime();
    const double dt = now - last_time;
    last_time = now;

    camera_input->tick(dt);

    glfwGetFramebufferSize(window, &new_width, &new_height);
    if (!new_width || !new_height)
      continue;

    // Update UBO
    glm::vec3 dir{};
    static float rad_phi = glm::radians(-37.76f);   // ≈ -0.659 rad
    static float rad_theta = glm::radians(126.16f); // ≈  2.202 rad
    // For demo: change light via ImGui in present pass; here just compute from
    // rad_phi/theta
    dir.x = glm::cos(rad_phi) * glm::cos(rad_theta);
    dir.y = glm::sin(rad_phi);
    dir.z = glm::cos(rad_phi) * glm::sin(rad_theta);
    dir = -glm::normalize(dir);

    const auto view = camera.get_view_matrix();
    auto projection = glm::infinitePerspectiveLH_ZO(
      glm::radians(70.0F),
      static_cast<float>(new_width) / static_cast<float>(new_height),
      0.1F);
    UBO ubo_data{
      .view = view,
      .proj = projection,
      .camera_position = glm::vec4(camera.get_position(), 1.0F),
      .light_direction = glm::vec4(dir, 0.0f),
      .texture = 0,
      .cube_texture = 0,
    };
    main_ubo.upload(context, std::span{ &ubo_data, 1 });

    // Recreate offscreen textures if window size changed
    ensure_size(new_width, new_height);

    // Acquire command buffer
    auto& buf = context.acquire_command_buffer();

    constexpr auto black = std::array{ 0.0F, 0.0F, 0.0F, 0.0F };

    // ---------------- PASS 1: OFFSCREEN GEOMETRY (MSAA -> resolve)
    // ----------------
    RenderPass gbuffer_pass {
            .color= {
                  RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = black,
                },
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = black,
                },
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = black,
                },
            },
            .depth ={ .load_op = LoadOp::Clear, .store_op = StoreOp::Store, .clear_depth = 0.0F, },
            .stencil = {},
            .layer_count = 1,
            .view_mask = 0,
        };

    // Framebuffer: render into color_msaa, resolve into color_resolved
    Framebuffer gbuffer_fb{
      .color = { 
                 Framebuffer::AttachmentDescription{
                   .texture = *g_uvs,
                 },
                 Framebuffer::AttachmentDescription{
                   .texture = *g_normal_rough,
                 },
                 Framebuffer::AttachmentDescription{
                   .texture = *g_texture_indices,
                 }, },
      .depth_stencil = { *g_depth },
      .debug_name = "GBuffer",
    };

    buf.cmd_begin_rendering(gbuffer_pass,
                            gbuffer_fb,
                            {
                              .textures = {},
                            });

    buf.cmd_bind_depth_state(gbuffer_depth_state);

    const struct
    {
      glm::mat4 model_transform;
      std::uint64_t ubo;                 // UBO
      std::uint64_t material_ssbo;       // MaterialSSBO
      std::uint64_t material_remap_ssbo; // MaterialSSBO
      std::uint32_t sampler_index;
      std::uint32_t material_index;
    } data{
      .model_transform = glm::scale(glm::mat4{ 1.0F }, glm::vec3{ 0.1F }),
      .ubo = main_ubo.get_address(context),
      .material_ssbo = duck_model.get_material_buffer_handle(context),
      .material_remap_ssbo =
        duck_model.get_material_remap_buffer_handle(context),
      .sampler_index = 0,
      .material_index = 0,
    };
    duck_model.draw(buf, duck_model_file, as_bytes(&data, 1));

    // buf.cmd_bind_graphics_pipeline(*geometry_ssbo);
    // draw_model(buf, duck_model, duck_count, duck_lod_choice);
    // buf.cmd_bind_graphics_pipeline(*geometry_pc);
    // draw_model(buf, bistro_model, 1, bistro_lod_choice);

    // End offscreen pass
    buf.cmd_end_rendering();

    // ---------------- PASS 3: GBUFFER SHADING (albedo,normal) -> MSAA Target
    // ----------------§
    RenderPass lighting_pass{
    .color = {
        RenderPass::AttachmentDescription{
            .load_op = LoadOp::Clear,
            .store_op = StoreOp::Store,
            .clear_colour = std::array{0.0F, 0.0F, 0.0F, 1.0F}, // clear HDR target
        },
    },
    .depth = {},
    .stencil = {},
    .layer_count = 1,
    .view_mask = 0,
};

    Framebuffer lighting_fb{
    .color = {
        Framebuffer::AttachmentDescription{ .texture = *color_resolved },
    },
    .debug_name = "Lighting FB",
};

    buf.cmd_begin_rendering(lighting_pass, lighting_fb, {});

    // Bind fullscreen pipeline
    buf.cmd_bind_graphics_pipeline(*lighting_pipeline);
    buf.cmd_bind_depth_state({
      .compare_operation = CompareOp::AlwaysPass,
      .is_depth_test_enabled = false,
      .is_depth_write_enabled = false,
    });
    struct LightingPC
    {
      std::uint32_t g_uv_index;
      std::uint32_t g_normal_rough_idx;
      std::uint32_t g_texture_indices_idx;
      std::uint32_t g_depth_idx;
      std::uint32_t g_sampler_index{ 0 };
      std::uint64_t ubo_address;
    };
    LightingPC lpc{
      .g_uv_index = g_uvs.index(),
      .g_normal_rough_idx = g_normal_rough.index(),
      .g_texture_indices_idx = g_texture_indices.index(),
      .g_depth_idx = g_depth.index(),
      .ubo_address = main_ubo.get_address(context),
    };
    buf.cmd_push_constants(lpc, 0);

    buf.cmd_draw(3, 1, 0, 0);

    buf.cmd_end_rendering();

    RenderPass forward_pass{
    .color = {
        RenderPass::AttachmentDescription{
            .load_op = LoadOp::Load,
            .store_op = StoreOp::Store,
        },
    },
    .depth = {.load_op = LoadOp::Load, .store_op = StoreOp::DontCare,},
    .stencil = {},
    .layer_count = 1,
    .view_mask = 0,
};

    Framebuffer forward_framebuffer{
    .color = {
        Framebuffer::AttachmentDescription{ .texture = *color_resolved },
    },
    .depth_stencil = {
      .texture = *g_depth
    },
    .debug_name = "Forward FB",
};
    buf.cmd_begin_rendering(forward_pass, forward_framebuffer, {});
    buf.cmd_bind_graphics_pipeline(*grid_pipeline);
    buf.cmd_bind_depth_state({
      .compare_operation = CompareOp::Greater,
    });
    struct GridPC
    {
      std::uint64_t ubo_address; // matches UBO pc
      std::uint64_t padding{ 0 };
      alignas(16) glm::vec4 origin;
      alignas(16) glm::vec4 grid_colour_thin;
      alignas(16) glm::vec4 grid_colour_thick;
      alignas(16) glm::vec4 grid_params;
    };
    GridPC grid_pc{
      .ubo_address = main_ubo.get_address(context),
      .origin = glm::vec4{ 0.0f },
      .grid_colour_thin = glm::vec4{ 0.5f, 0.5f, 0.5f, 1.0f },
      .grid_colour_thick = glm::vec4{ 0.15f, 0.15f, 0.15f, 1.0f },
      .grid_params = glm::vec4{ 100.0f, 0.025f, 2.0f, 0.0f },
    };
    buf.cmd_push_constants<GridPC>(grid_pc, 0);
    buf.cmd_draw(6, 1, 0, 0);

    canvas_3d.clear();
    canvas_3d.set_mvp(ubo_data.proj * ubo_data.view);
    canvas_3d.box(glm::translate(glm::mat4{ 1.0F }, glm::vec3{ 5, 5, 0 }),
                  BoundingBox(glm::vec3(-2), glm::vec3(+2)),
                  glm::vec4(1, 1, 0, 1));
    static auto initial_pos = camera.get_position().y;
    canvas_3d.frustum(
      glm::lookAt(
        glm::vec3(cos(glfwGetTime()), initial_pos, sin(glfwGetTime())),
        glm::vec3{ 0, 0, 0 },
        glm::vec3(0.0f, 1.0f, 0.0f)),
      glm::perspective(glm::radians(60.0f),
                       static_cast<float>(new_width) / new_height,
                       10.0f,
                       30.0f),
      glm::vec4(1, 1, 1, 1));
    canvas_3d.render(context, forward_framebuffer, buf, 1);

    buf.cmd_end_rendering();

    // ---------------- PASS 4: PRESENT (sample resolved -> swapchain)
    // ----------------
    auto swapchain_texture = context.get_current_swapchain_texture();
    if (!swapchain_texture)
      continue;

    RenderPass present_pass {
            .color = {
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = std::array{1.0F, 1.0F, 1.0F, 1.0F},
                },
            },
            .depth = {
                .load_op = LoadOp::Load, 
                .store_op = StoreOp::DontCare,
            },
            .stencil = {},
            .layer_count = 1,
            .view_mask = 0,
        };

    Framebuffer present_fb{
      .color = { Framebuffer::AttachmentDescription{
        .texture = swapchain_texture,
      } },
      .depth_stencil = { *g_depth },
      .debug_name = "Present FB",
    };

    buf.cmd_begin_rendering(present_pass, present_fb, {});

    // Begin ImGui for present pass
    imgui->begin_frame(present_fb);
    draw_compact_wasd_qe_widget();

    ImGui::Begin("Texture Viewer");
    ImGui::SliderAngle("Light Direction (phi)",
                       &rad_phi,
                       0.0F,
                       360.0F,
                       "%.1f",
                       ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderAngle("Light Direction (theta)",
                       &rad_theta,
                       -180.0F,
                       180.0F,
                       "%.1f",
                       ImGuiSliderFlags_AlwaysClamp);

    // ImGui::ShowDemoWindow();
    // ImPlot::ShowDemoWindow();
    ImGui::End();

    // Bind post pipeline and sample the resolved texture by index using push
    // constants
    buf.cmd_bind_graphics_pipeline(*post_pipeline);
    struct PostPC
    {
      std::uint32_t tex_index;
    };
    PostPC post_pc{
      .tex_index = color_resolved.index(),
    };

    buf.cmd_push_constants<PostPC>(post_pc, 0);

    // Full-screen triangle
    buf.cmd_draw(3, 1, 0, 0);

    imgui->end_frame(buf);
    buf.cmd_end_rendering();

    // Submit and present
    const auto result = context.submit(buf, swapchain_texture);
    (void)result; // handle errors as needed
  }
}

auto
main() -> std::int32_t
{
  using GLFWPointer = std::unique_ptr<GLFWwindow, decltype(&destroy_glfw)>;

#ifdef _WIN32
  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WIN32);
#else
  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif

  if (!glfwInit()) {
    const char* error{};
    glfwGetError(&error);
    std::cout << std::format("GLFW error: %s\n", error);
    return 1;
  }
  glfwSetErrorCallback([](int c, const char* d) {
    std::cerr << std::format("GLFW error {}: {}\n", c, d);
  });

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  constexpr std::int32_t initial_width = 1920;
  constexpr std::int32_t initial_height = 1080;

  const GLFWPointer window(
    glfwCreateWindow(
      initial_width, initial_height, "Test Window", nullptr, nullptr),
    &destroy_glfw);

  WindowState state{};
  if (!is_wayland())
    glfwGetWindowPos(window.get(), &state.windowed_x, &state.windowed_y);
  glfwGetWindowSize(
    window.get(), &state.windowed_width, &state.windowed_height);

  auto context =
    VkBindless::Context::create([win = window.get()](VkInstance instance) {
      VkSurfaceKHR surface;
      if (const auto res =
            glfwCreateWindowSurface(instance, win, nullptr, &surface);
          res != VK_SUCCESS) {
        return as_null(surface);
      }
      return surface;
    });
  if (!context) {
    std::cerr << "Failed to create Vulkan context: " << context.error().message
              << std::endl;
    return 1;
  }
  auto ctx = std::move(context.value());

  run_main(state, window.get(), *ctx);

  ctx.reset();

  return 0;
}
