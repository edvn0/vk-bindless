#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/event_system.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/scope_exit.hpp"
#include "vk-bindless/shader.hpp"
#include "vk-bindless/transitions.hpp"
#include "vk-bindless/vulkan_context.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <bit>
#include <cstdint>
#include <format>
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

template<typename T>
  requires std::is_pointer_v<T>
static auto
launder_cast(const void* ptr)
{
  return std::bit_cast<T>(ptr);
}

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

class GameLogicHandler : public EventSystem::EventHandler
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

// UI handler (lower priority than game logic)
class UIHandler
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
setup_event_callbacks(GLFWwindow* window,
                      EventSystem::EventDispatcher* dispatcher) -> void
{
  glfwSetWindowUserPointer(window, dispatcher);

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
    });

  glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
    auto* dispatcher =
      static_cast<EventSystem::EventDispatcher*>(glfwGetWindowUserPointer(win));
    dispatcher->handle_cursor_pos_callback(win, x, y);
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
    if (auto res = glfwCreateWindowSurface(instance, win, nullptr, &surface);
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

  // Subscribe handlers to events they care about
  event_dispatcher.subscribe<EventSystem::KeyEvent>(window_manager);
  event_dispatcher.subscribe<EventSystem::WindowResizeEvent>(window_manager);

  event_dispatcher.subscribe<EventSystem::KeyEvent>(game_handler);
  event_dispatcher.subscribe<EventSystem::MouseButtonEvent>(game_handler);
  event_dispatcher.subscribe<EventSystem::MouseMoveEvent>(game_handler);

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
      } });

  SCOPE_EXIT
  {
    shader.reset();
    cube_pipeline_handle.reset();
  };

  while (!glfwWindowShouldClose(window.get())) {
    event_dispatcher.process_events();

    glfwGetFramebufferSize(window.get(), &new_width, &new_height);
    if (!new_width || !new_height)
      continue;

    auto& buf = vulkan_context->acquire_command_buffer();

    RenderPass render_pass {
      .color= { RenderPass::AttachmentDescription{
  .load_op = LoadOp::Clear,
        .clear_colour = std::array{1.0F, 0.0F, 0.0F, 1.0F},
      }, },
      .depth ={},
      .stencil =  {},
    .layer_count = 1,
      .view_mask = 0,
    };
    auto swapchain_texture = vulkan_context->get_current_swapchain_texture();
    Framebuffer framebuffer{
      .color = { Framebuffer::AttachmentDescription{
          .texture = swapchain_texture, }, },
      .depth_stencil = {},
      .debug_name = "Main Framebuffer",
    };
    buf.cmd_begin_rendering(render_pass, framebuffer, {});
    buf.cmd_bind_graphics_pipeline(*cube_pipeline_handle);
    // buf.cmd_draw(36, 1, 0, 0);
    buf.cmd_end_rendering();
    const auto result = vulkan_context->submit(buf, swapchain_texture);
  }

  return 0;
}
