#include "vk-bindless/buffer.hpp"
#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/common.hpp"
#include "vk-bindless/event_system.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/mesh.hpp"
#include "vk-bindless/pipeline.hpp"
#include "vk-bindless/scope_exit.hpp"
#include "vk-bindless/shader.hpp"
#include "vk-bindless/transitions.hpp"
#include "vk-bindless/vulkan_context.hpp"

#include <cstring>
#include <filesystem>
#include <imgui.h>
#include <thread>

#define GLFW_INCLUDE_VULKAN
#include "../third-party/KTX-Software/include/ktx.h"
#include "backends/imgui_impl_glfw.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "helper.hpp"
#include "implot.h"
#include "vk-bindless/camera.hpp"
#include "vk-bindless/imgui_renderer.hpp"
#include <GLFW/glfw3.h>
#include <bit>
#include <cstdint>
#include <format>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <iostream>

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
    if (w > 0 && h > 0)
      mouse_norm = { static_cast<float>(e.x_pos / w),
                     1.0f - static_cast<float>(e.y_pos / h) };
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
run_main() -> void
{
  using namespace VkBindless;
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
    return;
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

  auto context = Context::create([win = window.get()](VkInstance instance) {
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
    return;
  }
  auto vulkan_context = std::move(context.value());

  auto duck_mesh = *Mesh::create(*vulkan_context, "assets/meshes/Duck.glb");

  Holder<BufferHandle> ssbo_transforms;
  static constexpr auto duck_count = 1000;
  {
    auto transforms = std::views::iota(0, duck_count) |
                      std::views::transform([](const int) -> glm::mat4 {
                        auto translation =
                          glm::vec3{ glm::linearRand(-100.0F, 100.0F),
                                     glm::linearRand(-100.0F, 100.0F),
                                     glm::linearRand(-100.0F, 100.0F) };
                        return glm::translate(glm::mat4{ 1.0F }, translation);
                      }) |
                      std::ranges::to<std::vector<glm::mat4>>();
    auto as_bytes = std::span{ std::bit_cast<std::byte*>(transforms.data()),
                               transforms.size() * sizeof(glm::mat4) };
    ssbo_transforms = VkDataBuffer::create(
      *vulkan_context,
      {
        .data = as_bytes,
        .size = as_bytes.size_bytes(),
        .storage = VkBindless::StorageType::HostVisible,
        .usage = VkBindless::BufferUsageFlags::StorageBuffer,
        .debug_name = "Duck transforms",
      });
  }

  glfwSetWindowUserPointer(window.get(), &state);
  EventSystem::EventDispatcher event_dispatcher;
  setup_event_callbacks(window.get(), &event_dispatcher);

  // Create and register event handlers
  const auto window_manager =
    std::make_shared<WindowManager>(window.get(), &state);
  const auto game_handler = std::make_shared<GameLogicHandler>();
  const auto ui_handler = std::make_shared<UIHandler>();

  Camera camera(
    std::make_unique<FirstPersonCameraBehaviour>(glm::vec3{ 0, 2.0F, -3.0F },
                                                 glm::vec3{ 0, 0, 0.0F },
                                                 glm::vec3{ 0, 1, 0 }));

  const auto camera_input = std::make_shared<CameraInputHandler>(
    window.get(),
    dynamic_cast<FirstPersonCameraBehaviour*>(camera.get_behaviour()));

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

  if (!std::filesystem::is_regular_file("data/brdfLUT.ktx2")) {
    auto brdf_lut_compute_shader = VkShader::create(
      vulkan_context.get(), "assets/shaders/brdf_lut_compute.shader");

    constexpr auto brdf_width = 512;
    constexpr auto brdf_height = 512;
    constexpr auto brdf_monte_carlo_sample_count = 1024;
    constexpr auto brdf_buffer_size =
      4ULL * sizeof(std::uint16_t) * brdf_width * brdf_height;
    (void)brdf_buffer_size;

    auto span = std::span{ &brdf_monte_carlo_sample_count,
                           sizeof(brdf_monte_carlo_sample_count) };

    auto brdf_lut_pipeline = VkComputePipeline::create(
      vulkan_context.get(),
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

    auto buffer =
      VkDataBuffer::create(*vulkan_context,
                           {
                             .data = {},
                             .size = brdf_buffer_size,
                             .storage = StorageType::DeviceLocal,
                             .usage = BufferUsageFlags::StorageBuffer |
                                      BufferUsageFlags::TransferDst,
                             .debug_name = "BRDF LUT Buffer",
                           });

    auto& buf = vulkan_context->acquire_command_buffer();
    buf.cmd_bind_compute_pipeline(*brdf_lut_pipeline);
    struct
    {
      std::uint32_t w = brdf_width;
      std::uint32_t h = brdf_height;
      std::uint64_t addr;
      std::array<std::uint64_t, 6> _pad{};
    } pc{
      .addr = vulkan_context->get_device_address(*buffer),
    };
    static_assert(sizeof(decltype(pc)) == 64);
    buf.cmd_push_constants<decltype(pc)>(pc, 0);
    buf.cmd_dispatch_thread_groups({ brdf_width / 16, brdf_height / 16, 1 });

    vulkan_context->wait_for(*vulkan_context->submit(buf, {}));

    std::vector<ktx_uint8_t> bytes(brdf_buffer_size);
    std::memcpy(bytes.data(),
                vulkan_context->get_mapped_pointer(*buffer),
                brdf_buffer_size);

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
          std::cerr
            << "Could not write to folder 'data' with the created file\n";
        }
      }
    }
    ktxTexture2_Destroy(tex);
  }

  std::int32_t new_width = 0;
  std::int32_t new_height = 0;

  // MSAA sample count
  constexpr VkSampleCountFlagBits kMsaa = VK_SAMPLE_COUNT_1_BIT;

  VertexInput static_opaque_geometry_vertex_input = VertexInput::create(
    { VertexFormat::Float3, VertexFormat::Float3, VertexFormat::Float2 });
  auto opaque_geometry = VkShader::create(
    vulkan_context.get(), "assets/shaders/opaque_geometry.shader");
  auto static_opaque_geometry_pipeline_handle = VkGraphicsPipeline::create(
    vulkan_context.get(),
    {
      .vertex_input = static_opaque_geometry_vertex_input,
      .shader = *opaque_geometry,
      .color = { ColourAttachment{
                   .format = Format::R_UI8,
                 },
                 ColourAttachment{
                   .format = Format::RG_F16,
                 },
                 ColourAttachment{
                   .format = Format::RGBA_F16,
                 } },
      .depth_format = Format::Z_F32,
      .cull_mode = CullMode::Back,
      .winding = WindingMode::CW,
      .sample_count = kMsaa, // ensure pipeline is compatible with MSAA target
      .debug_name = "Static Opaque Pipeline",
    });

  auto lighting_shader = VkShader::create(
    vulkan_context.get(), "assets/shaders/lighting_gbuffer.shader");
  auto lighting_pipeline = VkGraphicsPipeline::create(
    vulkan_context.get(),
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

  constexpr auto null_k_bytes = [](auto k = 0) {
    return std::vector<std::byte>(k, std::byte{ 0 });
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
    .width = static_cast<std::uint32_t>(initial_width),
    .height = static_cast<std::uint32_t>(initial_height),
    .depth = 1,
  };

  auto null_ubo = null_k_bytes(align_size(sizeof(UBO), 16));
  auto main_ubo = FrameUniform<1>::create(*vulkan_context, null_ubo);

  // Create ImGui renderer
  auto imgui = std::make_unique<ImGuiRenderer>(
    *vulkan_context, "assets/fonts/Roboto-Regular.ttf");

  Holder<TextureHandle> color_resolved{};
  Holder<TextureHandle> g_albedo; // texture indices
  Holder<TextureHandle> g_uvs;    // texture uvs
  Holder<TextureHandle>
    g_normal_rough;              // normal.xyz (view space), roughness in .w
  Holder<TextureHandle> g_depth; // depth (Z_F32)

  auto create_offscreen_targets = [&](std::uint32_t w, std::uint32_t h) {
    offscreen_extent.width = w;
    offscreen_extent.height = h;

    // Resolved single-sample color (sampled to be used by post shader)
    color_resolved =
      VkTexture::create(*vulkan_context,
                        {
                          .data = {},
                          .format = Format::RGBA_F32,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,
                                      .depth = 1 },
                          .usage_flags = TextureUsageFlags::ColourAttachment |
                                         TextureUsageFlags::Sampled,
                          .layers = 1,
                          .mip_levels = 1,
                          .sample_count = VK_SAMPLE_COUNT_1_BIT,
                          .tiling = VK_IMAGE_TILING_OPTIMAL,
                          .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                          .is_owning = true,
                          .is_swapchain = false,

                          .externally_created_image = {},
                          .debug_name = "Offscreen Color Resolved",
                        });
    g_albedo =
      VkTexture::create(*vulkan_context,
                        { .format = Format::R_UI8,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,.depth = 1, },
                          .usage_flags = TextureUsageFlags::ColourAttachment |
                                         TextureUsageFlags::Sampled,
                          .layers = 1,
                          .mip_levels = 1,
                          .sample_count = VK_SAMPLE_COUNT_1_BIT,
                          .debug_name = "GBuffer AlbedoMetallic" });
    g_uvs =
      VkTexture::create(*vulkan_context,
                        { .format = Format::RG_F16,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,.depth = 1, },
                          .usage_flags = TextureUsageFlags::ColourAttachment |
                                         TextureUsageFlags::Sampled,
                          .layers = 1,
                          .mip_levels = 1,
                          .sample_count = VK_SAMPLE_COUNT_1_BIT,
                          .debug_name = "GBuffer UVs" });
    g_normal_rough =
      VkTexture::create(*vulkan_context,
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
      *vulkan_context,
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
  create_offscreen_targets(static_cast<std::uint32_t>(initial_width),
                           static_cast<std::uint32_t>(initial_height));
  // Post pipeline & shader (samples resolved offscreen texture by index via
  // push constants)
  auto post_shader =
    VkShader::create(vulkan_context.get(), "assets/shaders/post.shader");

  auto post_pipeline = VkGraphicsPipeline::create(
    vulkan_context.get(),
    {
      .shader = *post_shader,
      .color = { ColourAttachment{ .format = Format::BGRA_UN8 } },
      .depth_format = Format::Z_F32,
      .sample_count = VK_SAMPLE_COUNT_1_BIT,
      .debug_name = "Post Pipeline",
    });

  auto grid_shader =
    VkShader::create(vulkan_context.get(), "assets/shaders/grid.shader");

  auto grid_pipeline = VkGraphicsPipeline::create(
    vulkan_context.get(),
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

  static int lod_choice = 0;

  constexpr DepthState gbuffer_depth_state{
    .compare_operation = CompareOp::Greater,
    .is_depth_test_enabled = true,
    .is_depth_write_enabled = true,
  };
  static DepthState grid_depth_state{
    .compare_operation = CompareOp::Greater,
    .is_depth_test_enabled = true,
    .is_depth_write_enabled = false,
  };

  while (!glfwWindowShouldClose(window.get())) {
    event_dispatcher.process_events();
    const double now = glfwGetTime();
    const double dt = now - last_time;
    last_time = now;

    camera_input->tick(dt);

    glfwGetFramebufferSize(window.get(), &new_width, &new_height);
    if (!new_width || !new_height)
      continue;

    // Update UBO
    glm::vec3 dir{};
    static float rad_phi{};
    static float rad_theta{};
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
      .texture = duck_mesh.get_mesh_data().material.albedo_texture,
      .cube_texture = 0,
    };
    main_ubo.upload(*vulkan_context, std::span{ &ubo_data, 1 });

    struct PC
    {
      std::uint64_t ubo_address;
      std::uint64_t ssbo_address;
    };
    PC pc{
      .ubo_address = main_ubo.get_address(*vulkan_context),
      .ssbo_address = vulkan_context->get_device_address(*ssbo_transforms),
    };

    // Recreate offscreen textures if window size changed
    ensure_size(new_width, new_height);

    // Acquire command buffer
    auto& buf = vulkan_context->acquire_command_buffer();

    // ---------------- PASS 1: OFFSCREEN GEOMETRY (MSAA -> resolve)
    // ----------------
    RenderPass gbuffer_pass {
            .color= {
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = std::array{0.0F,0.0F,0.0F,0.0F},
                },
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = std::array{0.0F,0.0F,0.0F,0.0F},
                },
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = std::array{0.0F,0.0F,0.0F,0.0F},
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
                    .texture = *g_albedo,
                },
                Framebuffer::AttachmentDescription{
                    .texture = *g_uvs,
                },
                Framebuffer::AttachmentDescription{
                    .texture = *g_normal_rough,
                },
            },
            .depth_stencil = { *g_depth },
            .debug_name = "GBuffer",
        };

    buf.cmd_begin_rendering(gbuffer_pass,
                            gbuffer_fb,
                            {
                              .textures = {},
                            });

    buf.cmd_bind_graphics_pipeline(*static_opaque_geometry_pipeline_handle);
    buf.cmd_push_constants<PC>(pc, 0);
    buf.cmd_bind_depth_state(gbuffer_depth_state);
    buf.cmd_bind_vertex_buffer(0, duck_mesh.get_vertex_buffer(), 0);
    auto&& [dcount, doffset] = duck_mesh.get_index_binding_data(lod_choice);
    buf.cmd_bind_index_buffer(duck_mesh.get_index_buffer(),
                              IndexFormat::UI32,
                              doffset * sizeof(std::uint32_t));

    buf.cmd_draw_indexed(dcount, duck_count, 0, 0, 0);

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
      std::uint32_t g_material_index;
      std::uint32_t g_uv_index;
      std::uint32_t g_normal_rough_idx;
      std::uint32_t g_depth_idx;
      std::uint32_t g_sampler_index{ 0 };
      std::uint64_t ubo_address;
    };
    LightingPC lpc{
      .g_material_index = g_albedo.index(),
      .g_uv_index = g_uvs.index(),
      .g_normal_rough_idx = g_normal_rough.index(),
      .g_depth_idx = g_depth.index(),
      .ubo_address = main_ubo.get_address(*vulkan_context),
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
    buf.cmd_bind_depth_state(grid_depth_state);
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
      .ubo_address = main_ubo.get_address(*vulkan_context),
      .origin = glm::vec4{ 0.0f },
      .grid_colour_thin = glm::vec4{ 0.5f, 0.5f, 0.5f, 1.0f },
      .grid_colour_thick = glm::vec4{ 0.15f, 0.15f, 0.15f, 1.0f },
      .grid_params = glm::vec4{ 100.0f, 0.025f, 2.0f, 0.0f },
    };
    buf.cmd_push_constants<GridPC>(grid_pc, 0);
    buf.cmd_draw(6, 1, 0, 0);
    buf.cmd_end_rendering();

    // ---------------- PASS 4: PRESENT (sample resolved -> swapchain)
    // ----------------
    auto swapchain_texture = vulkan_context->get_current_swapchain_texture();

    RenderPass present_pass {
            .color = {
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::Store,
                    .clear_colour = std::array{1.0F, 1.0F, 1.0F, 1.0F},
                },
            },
            .depth = {.load_op = LoadOp::Load, .store_op = StoreOp::DontCare,},
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
    ImGui::Begin("Texture Viewer");
    ImGui::Image(
      ImTextureID{ duck_mesh.get_mesh_data().material.albedo_texture },
      ImVec2(512, 512));
    draw_compact_wasd_qe_widget();
    depth_state_widget("Grid depth state", grid_depth_state);
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

    // Lod choice 0->4 inclusive
    auto max = duck_mesh.get_mesh_data().lod_levels.size();
    ImGui::SliderInt(
      "Duck LOD", &lod_choice, 0, static_cast<std::int32_t>(max) - 1);

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
    const auto result = vulkan_context->submit(buf, swapchain_texture);
    (void)result; // handle errors as needed
  }
}

auto
main() -> std::int32_t
{
  run_main();

  return 0;
}
