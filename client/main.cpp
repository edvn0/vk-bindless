#include "imgui.h"
#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/event_system.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/scope_exit.hpp"
#include "vk-bindless/shader.hpp"
#include "vk-bindless/transitions.hpp"
#include "vk-bindless/vulkan_context.hpp"

#define GLFW_INCLUDE_VULKAN
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
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

  auto handle_event(const EventSystem::MouseButtonEvent& e) -> bool override {
    if (ImGui::GetIO().WantCaptureMouse) return false;
    if (e.button == GLFW_MOUSE_BUTTON_RIGHT) {
      mouse_held = (e.action == GLFW_PRESS);
      glfwSetInputMode(window, GLFW_CURSOR,
                       mouse_held ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
      if (mouse_held) behaviour->mouse_position = mouse_norm;
    }
    return mouse_held;
  }

  auto handle_event(const EventSystem::MouseMoveEvent& e) -> bool override {
    int w{}, h{};
    glfwGetFramebufferSize(window, &w, &h);
    if (w > 0 && h > 0)
      mouse_norm = { static_cast<float>(e.x_pos / w),
                     1.0f - static_cast<float>(e.y_pos / h) };
    return mouse_held;
  }

public:
  auto tick(double dt) const -> void
  {
    const bool imgui_block = ImGui::GetIO().WantCaptureMouse;
    behaviour->update(dt, mouse_norm, mouse_held && !imgui_block);
  }
};

// UI handler (lower priority than game logic)
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
        return true; // Consume tab key
      }
      if (event.key == GLFW_KEY_I) {
        std::cout << "Open inventory\n";
        return true; // Consume inventory key
      }
    }
    return false;
  }

  bool handle_event(const EventSystem::MouseButtonEvent& event) override
  {
    if (event.action == GLFW_PRESS && event.button == GLFW_MOUSE_BUTTON_RIGHT) {
      std::cout << "Context menu\n";
      return true; // Consume right click for UI
    }
    return false;
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
      ImGuiIO& io = ImGui::GetIO();
      io.MousePos = ImVec2(static_cast<float>(xpos), static_cast<float>(ypos));
      io.MouseDown[imgui_button] = action == GLFW_PRESS;
    });

  glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
    auto* dispatcher =
      static_cast<EventSystem::EventDispatcher*>(glfwGetWindowUserPointer(win));
    dispatcher->handle_cursor_pos_callback(win, x, y);

    ImGui::GetIO().MousePos =
      ImVec2(static_cast<float>(x), static_cast<float>(y));
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
main() -> std::int32_t
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
    return 1;
  }

  auto vulkan_context = std::move(context.value());
  SCOPE_EXIT
  {
    vulkan_context.reset();
  };

  glfwSetWindowUserPointer(window.get(), &state);

  EventSystem::EventDispatcher event_dispatcher;
  setup_event_callbacks(window.get(), &event_dispatcher);

  // Create and register event handlers
  const auto window_manager =
    std::make_shared<WindowManager>(window.get(), &state);
  const auto game_handler = std::make_shared<GameLogicHandler>();
  const auto ui_handler = std::make_shared<UIHandler>();
  auto camera_behaviour = std::make_unique<FirstPersonCameraBehaviour>(
    glm::vec3{ 0, 3, -5.F }, glm::vec3{ 0, 1.5F, 0.0F }, glm::vec3{ 0, 1, 0 });
  auto* camera_behaviour_ptr = camera_behaviour.get();
  Camera camera(std::move(camera_behaviour));

  const auto camera_input =
    std::make_shared<CameraInputHandler>(window.get(), camera_behaviour_ptr);

  event_dispatcher.subscribe<EventSystem::KeyEvent,
                             EventSystem::MouseMoveEvent,
                             EventSystem::MouseButtonEvent>(camera_input);

  // Subscribe handlers to events they care about
  event_dispatcher
    .subscribe<EventSystem::WindowResizeEvent, EventSystem::KeyEvent>(
      window_manager);

  event_dispatcher.subscribe<EventSystem::MouseMoveEvent,
                             EventSystem::KeyEvent,
                             EventSystem::MouseButtonEvent>(game_handler);

  event_dispatcher.subscribe<EventSystem::KeyEvent>(ui_handler);

  std::int32_t new_width = 0;
  std::int32_t new_height = 0;

  auto shader =
    VkShader::create(vulkan_context.get(), "assets/shaders/cube.shader");

  auto cube_pipeline_handle = VkGraphicsPipeline::create(
    vulkan_context.get(),
    { .shader = static_cast<ShaderModuleHandle>(shader),
      .color = {
        ColourAttachment{
          .format = Format::BGRA_UN8,
        },
      },
    .depth_format = Format::Z_F32,
    });

  SCOPE_EXIT
  {
    shader.reset();
    cube_pipeline_handle.reset();
  };

  auto depth_texture = VkTexture::create(*vulkan_context, {
  .data = {},
  .format = Format::Z_F32,
  .extent = {
    .width = static_cast<std::uint32_t>(initial_width),
    .height = static_cast<std::uint32_t>(initial_height),
    .depth = 1,
  },
  .usage_flags =
    TextureUsageFlags::DepthStencilAttachment | TextureUsageFlags::Sampled,
  .layers = 1,
  .mip_levels = 1,
  .sample_count = VK_SAMPLE_COUNT_1_BIT,
  .tiling = VK_IMAGE_TILING_OPTIMAL,
  .initial_layout =
    VK_IMAGE_LAYOUT_UNDEFINED, // Initial layout for depth texture
  .is_owning = true,
  .is_swapchain = false,
  .externally_created_image = {},
  .debug_name = "Depth Texture",
  });

  auto imgui = std::make_unique<ImGuiRenderer>(
    *vulkan_context, "assets/fonts/Roboto-Regular.ttf");

  double last_time = glfwGetTime();
  while (!glfwWindowShouldClose(window.get())) {
    event_dispatcher.process_events();

    const double now = glfwGetTime();
    const double dt = now - last_time;
    last_time = now;

    camera_input->tick(dt);

    glfwGetFramebufferSize(window.get(), &new_width, &new_height);
    if (!new_width || !new_height)
      continue;

    auto& buf = vulkan_context->acquire_command_buffer();

    RenderPass render_pass {
      .color= { RenderPass::AttachmentDescription{
  .load_op = LoadOp::Clear,
        .clear_colour = std::array{1.0F, 1.0F, 1.0F, 1.0F},
      }, },
      .depth ={
        .load_op = LoadOp::Clear,
        .store_op = StoreOp::Store,
        .clear_depth = 0.0F,
      },
      .stencil =  {},
    .layer_count = 1,
      .view_mask = 0,
    };
    auto swapchain_texture = vulkan_context->get_current_swapchain_texture();
    Framebuffer framebuffer{
      .color = { Framebuffer::AttachmentDescription{
          .texture = swapchain_texture, }, },
      .depth_stencil = {.texture = *depth_texture,},
      .debug_name = "Main Framebuffer",
    };

    buf.cmd_begin_rendering(render_pass, framebuffer, {});
    imgui->begin_frame(framebuffer);

    ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Image(framebuffer.color.at(0).texture.index(), ImVec2(512, 512));
    float rad_phi{};
    float rad_theta{};
    ImGui::SliderAngle("Light Direction (phi, theta)",
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
    ImGui::ShowDemoWindow();
    ImGui::End();

    struct PC
    {
      glm::mat4 mvp{ 1.0F };
      glm::vec4 light_direction{ 0.0F };
    } pc{};
    // angles in radians
    // spherical → Cartesian
    glm::vec3 dir;
    dir.x = std::cos(rad_phi) * std::cos(rad_theta);
    dir.y = std::sin(rad_phi);
    dir.z = std::cos(rad_phi) * std::sin(rad_theta);

    // normalize for safety
    dir = glm::normalize(dir);

    // store in push constants
    pc.light_direction = glm::vec4(dir, 0.0f); // w=0 for a direction
    const auto view = camera.get_view_matrix();
    auto projection = glm::infinitePerspectiveLH_ZO(
      glm::radians(70.0F),
      static_cast<float>(new_width) / static_cast<float>(new_height),
      0.1F);
    const auto rotation = glm::rotate(glm::mat4(1.0F),
                                      static_cast<float>(glfwGetTime()),
                                      glm::vec3(0.0F, 1.0F, 0.0F));

    pc.mvp = projection * view * rotation;

    buf.cmd_bind_graphics_pipeline(*cube_pipeline_handle);
    buf.cmd_bind_depth_state({
      .compare_operation = CompareOp::Greater,
      .is_depth_write_enabled = true,
    });
    buf.cmd_push_constants<PC>(pc, 0);
    buf.cmd_draw(36, 1, 0, 0);

    imgui->end_frame(buf);
    buf.cmd_end_rendering();
    const auto result = vulkan_context->submit(buf, swapchain_texture);
  }

  return 0;
}
