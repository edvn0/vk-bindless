#include "vk-bindless/buffer.hpp"
#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/common.hpp"
#include "vk-bindless/event_system.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/mesh.hpp"
#include "vk-bindless/scope_exit.hpp"
#include "vk-bindless/shader.hpp"
#include "vk-bindless/transitions.hpp"
#include "vk-bindless/vulkan_context.hpp"
#include <imgui.h>

#define GLFW_INCLUDE_VULKAN
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
  bool handle_event(const EventSystem::KeyEvent& event) override
  {
    if (event.action == GLFW_PRESS) {
      if (event.key == GLFW_KEY_W) {
        std::cout << "Move forward\n";
        return false;
      }
      if (event.key == GLFW_KEY_SPACE) {
        std::cout << "Jump action\n";
        return true;
      }
    }
    return false;
  }
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
    FrameUniform  uniform;
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
auto upload(VkBindless::IContext& context, const std::span<T> data)
  -> void
  {
    upload(context, std::as_bytes(data));
  }

  auto upload(VkBindless::IContext& context,
              const std::span<const std::byte> data) -> void
  {
    auto index = context.get_swapchain().current_frame_index() % Count;
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

auto run_main() -> void
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
  SCOPE_EXIT
  {
    vulkan_context.reset();
  };

  auto duck_mesh = *Mesh::create(*vulkan_context, "assets/meshes/Duck.glb");

  glfwSetWindowUserPointer(window.get(), &state);
  EventSystem::EventDispatcher event_dispatcher;
  setup_event_callbacks(window.get(), &event_dispatcher);

  // Create and register event handlers
  const auto window_manager =
    std::make_shared<WindowManager>(window.get(), &state);
  const auto game_handler = std::make_shared<GameLogicHandler>();
  const auto ui_handler = std::make_shared<UIHandler>();

    Camera camera(std::make_unique<FirstPersonCameraBehaviour>(
      glm::vec3{ 0, 140, -200.F }, glm::vec3{ 0, 0, 0.0F }, glm::vec3{ 0, 1, 0 }));

  const auto camera_input =
    std::make_shared<CameraInputHandler>(window.get(), dynamic_cast<FirstPersonCameraBehaviour*>(
        camera.get_behaviour()));

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

  std::int32_t new_width = 0;
  std::int32_t new_height = 0;

  // Create shaders & pipelines
  auto shader =
    VkShader::create(vulkan_context.get(), "assets/shaders/cube.shader");

  auto opaque_geometry = VkShader::create(
    vulkan_context.get(), "assets/shaders/opaque_geometry.shader");

  auto prepass = VkShader::create(
    vulkan_context.get(), "assets/shaders/prepass.shader");

  // MSAA sample count
  constexpr VkSampleCountFlagBits kMsaa = VK_SAMPLE_COUNT_4_BIT;

  // Create cube pipeline with MSAA (if your wrapper supports sample_count flag)
  auto cube_pipeline_handle = VkGraphicsPipeline::create(
    vulkan_context.get(),
    {
      .shader = *shader,
      .color = { ColourAttachment{ .format = Format::R_F32 } },
      .depth_format = Format::Z_F32,
      .sample_count = kMsaa, // ensure pipeline is compatible with MSAA target
      .debug_name = "Cube Pipeline",
    });

  VertexInput static_opaque_geometry_vertex_input = VertexInput::create({VertexFormat::Float3,
                                                                      VertexFormat::Float3,
                                                                      VertexFormat::Float2});
  auto static_opaque_geometry_pipeline_handle = VkGraphicsPipeline::create(
    vulkan_context.get(),
    {
      .vertex_input = static_opaque_geometry_vertex_input,
      .shader = *opaque_geometry,
      .color = { ColourAttachment{ .format = Format::R_F32 } },
      .depth_format = Format::Z_F32,
      .cull_mode = CullMode::Back,
      .sample_count = kMsaa, // ensure pipeline is compatible with MSAA target
      .debug_name = "Static Opaque Pipeline",
    });

  auto static_opaque_prepass_handle = VkGraphicsPipeline::create(
  vulkan_context.get(),
  {
    .vertex_input = VertexInput::create({VertexFormat::Float3,}),
    .shader = *prepass,
    .depth_format = Format::Z_F32,
    .cull_mode = CullMode::Back,
    .sample_count = kMsaa, // ensure pipeline is compatible with MSAA target
    .debug_name = "Static Opaque Prepass Pipeline",
  });

  // Offscreen textures: MSAA color, resolved single-sample color, MSAA depth
  auto null_k_bytes = [](auto k = 0) {
    return std::vector<std::byte>(k, std::byte{ 0 });
  };

  struct UBO
  {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 camera_position;
    glm::vec4 light_direction;
    std::uint32_t texture;
    std::uint32_t cube_texture;
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
  auto main_ubo = FrameUniform<3>::create(*vulkan_context, null_ubo);

  // Create ImGui renderer
  auto imgui = std::make_unique<ImGuiRenderer>(
    *vulkan_context, "assets/fonts/Roboto-Regular.ttf");

  // Offscreen resource handles (create helper lambda so we can recreate on
  // resize)
  Holder<TextureHandle> color_msaa{};
  Holder<TextureHandle> color_resolved{};
  Holder<TextureHandle> depth_msaa{};

  auto create_offscreen_targets = [&](std::uint32_t w, std::uint32_t h) {
    offscreen_extent.width = w;
    offscreen_extent.height = h;

    // destroy old if present (wrapper should handle)
    color_msaa.reset();
    color_resolved.reset();
    depth_msaa.reset();

    // MSAA color
    color_msaa =
      VkTexture::create(*vulkan_context,
                        {
                          .data = {},
                          .format = Format::R_F32,
                          .extent = { .width = offscreen_extent.width,
                                      .height = offscreen_extent.height,
                                      .depth = 1 },
                          .usage_flags = TextureUsageFlags::ColourAttachment,
                          .layers = 1,
                          .mip_levels = 1,
                          .sample_count = kMsaa,
                          .tiling = VK_IMAGE_TILING_OPTIMAL,
                          .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                          .is_owning = true,
                          .is_swapchain = false,
                          .externally_created_image = {},
                          .debug_name = "Offscreen Color MSAA",
                        });

    // Resolved single-sample color (sampled to be used by post shader)
    color_resolved =
      VkTexture::create(*vulkan_context,
                        {
                          .data = {},
                          .format = Format::R_F32,
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

    // MSAA depth
    depth_msaa = VkTexture::create(
      *vulkan_context,
      {
        .data = {},
        .format = Format::Z_F32,
        .extent = { .width = offscreen_extent.width,
                    .height = offscreen_extent.height,
                    .depth = 1 },
        .usage_flags = TextureUsageFlags::DepthStencilAttachment,
        .layers = 1,
        .mip_levels = 1,
        .sample_count = kMsaa,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
        .is_owning = true,
        .is_swapchain = false,
        .externally_created_image = {},
        .debug_name = "Depth MSAA",
      });
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
      .depth_format = Format::Invalid,
      .sample_count = VK_SAMPLE_COUNT_1_BIT,
      .debug_name = "Post Pipeline",
    });


  double last_time = glfwGetTime();

  // Helper to check swapchain size changes and recreate offscreen targets if
  // needed
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
  static int shadow_lod_choice = 0;

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
    glm::vec3 dir;
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
    const auto rotation = glm::rotate(glm::mat4(1.0F),
                                      static_cast<float>(glfwGetTime()),
                                      glm::vec3(0.0F, 1.0F, 0.0F));
    UBO ubo_data{
      .model = rotation,
      .view = view,
      .proj = projection,
      .camera_position = glm::vec4(camera.get_position(), 1.0F),
      .light_direction = glm::vec4(dir, 0.0f),
      .texture = 0,
      .cube_texture = 0,
    };
    main_ubo.upload(*vulkan_context, std::span{ &ubo_data, 1 });

    struct PC
    {
      std::uint64_t ubo_address;
    };
    PC pc{ .ubo_address = main_ubo.get_address(*vulkan_context) };

    // Recreate offscreen textures if window size changed
    ensure_size(new_width, new_height);

    // Acquire command buffer
    auto& buf = vulkan_context->acquire_command_buffer();

    // PASS 0: Predepth
        buf.cmd_begin_rendering(
            RenderPass{
                .color = {},
                .depth = { .load_op = LoadOp::Clear, .store_op = StoreOp::Store, .clear_depth = 0.0F },
                .stencil = {},
                .layer_count = 1,
                .view_mask = 0,
            },
            Framebuffer{
                .color = {},
                .depth_stencil = { .texture = *depth_msaa },
                .debug_name = "Predepth FB",
            },
            {});
        buf.cmd_bind_graphics_pipeline(*static_opaque_prepass_handle);
        buf.cmd_bind_depth_state({
            .compare_operation = CompareOp::Greater,
            .is_depth_write_enabled = true,
        });
      buf.cmd_push_constants<PC>(pc, 0);

        buf.cmd_bind_vertex_buffer(0, duck_mesh.get_shadow_vertex_buffer(), 0);
        auto&& [count, offset] = duck_mesh.get_shadow_index_binding_data(shadow_lod_choice);
        buf.cmd_bind_index_buffer(
            duck_mesh.get_shadow_index_buffer(), IndexFormat::UI32, offset * sizeof(std::uint32_t));
        buf.cmd_draw_indexed(count, 1, 0, 0, 0);
        buf.cmd_end_rendering();

    // ---------------- PASS 1: OFFSCREEN GEOMETRY (MSAA -> resolve)
    // ----------------
    RenderPass gbuffer_pass {
            .color= {
                RenderPass::AttachmentDescription{
                    .load_op = LoadOp::Clear,
                    .store_op = StoreOp::MsaaResolve,
                    .clear_colour = std::array{0.1F, 0.1F, 0.1F, 1.0F},
                },
            },
            .depth ={ .load_op = LoadOp::Load, .store_op = StoreOp::DontCare, },
            .stencil = {},
            .layer_count = 1,
            .view_mask = 0,
        };

    // Framebuffer: render into color_msaa, resolve into color_resolved
    Framebuffer gbuffer_fb{
            .color = {
                Framebuffer::AttachmentDescription{
                    .texture = *color_msaa,
                    .resolve_texture = *color_resolved,
                },
            },
            .depth_stencil = { .texture = *depth_msaa },
            .debug_name = "GBuffer (MSAA + Resolve)",
        };

    buf.cmd_begin_rendering(gbuffer_pass, gbuffer_fb, {});

    buf.cmd_bind_graphics_pipeline(*static_opaque_geometry_pipeline_handle);
    buf.cmd_push_constants<PC>(pc, 0);
    buf.cmd_bind_depth_state({
      .compare_operation = CompareOp::GreaterEqual,
      .is_depth_test_enabled = true,
      .is_depth_write_enabled = false,
    });
    buf.cmd_bind_vertex_buffer(0, duck_mesh.get_vertex_buffer(), 0);
    auto&& [dcount, doffset] = duck_mesh.get_index_binding_data(lod_choice);
    buf.cmd_bind_index_buffer(
      duck_mesh.get_index_buffer(), IndexFormat::UI32, doffset * sizeof(std::uint32_t));

    buf.cmd_draw_indexed(dcount, 1, 0, 0, 0);

    // End offscreen pass
    buf.cmd_end_rendering();

    // ---------------- PASS 2: PRESENT (sample resolved -> swapchain)
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
            .depth = {},
            .stencil = {},
            .layer_count = 1,
            .view_mask = 0,
        };

    Framebuffer present_fb{
      .color = { Framebuffer::AttachmentDescription{
        .texture = swapchain_texture,
      } },
      .depth_stencil = {},
      .debug_name = "Present FB",
    };

    buf.cmd_begin_rendering(present_pass,
                            present_fb,
                            {
                              //.textures = { *color_msaa, *color_resolved, },
                            });

    // Begin ImGui for present pass
    imgui->begin_frame(present_fb);
    ImGui::Begin("Texture Viewer");
    ImGui::Image(ImTextureID{ 0 }, ImVec2(512, 512));
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
        ImGui::SliderInt("Duck LOD", &lod_choice, 0, static_cast<std::int32_t>(max) - 1);
    auto max_shadow = duck_mesh.get_mesh_data().shadow_lod_levels.size();
        ImGui::SliderInt("Duck Shadow LOD", &shadow_lod_choice, 0, static_cast<std::int32_t>(max_shadow) - 1);

    //ImGui::ShowDemoWindow();
    //ImPlot::ShowDemoWindow();
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
