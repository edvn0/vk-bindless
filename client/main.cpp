#include "vk-bindless/vulkan_context.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <bit>
#include <cstdint>

static auto destroy_glfw(GLFWwindow *window) -> void {
  if (window) {
    glfwDestroyWindow(window);
    glfwTerminate();
  }
}

struct WindowState {
  std::int32_t windowed_x{};
  std::int32_t windowed_y{};
  std::int32_t windowed_width{};
  std::int32_t windowed_height{};
  bool fullscreen{false};
};

template <typename T> static auto launder_cast(const void *ptr) -> T {
  return std::bit_cast<T>(ptr);
}

static auto is_wayland() -> bool {
  int platform = glfwGetPlatform();
  return platform == GLFW_PLATFORM_WAYLAND;
}

void key_callback(GLFWwindow *win, int key, int, int action, int,
                  void *user_data) {
  if (action != GLFW_PRESS)
    return;
  auto state = launder_cast<WindowState *>(user_data);

  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(win, GLFW_TRUE);
    return;
  }

  if (key == GLFW_KEY_F11) {
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);

    if (!state->fullscreen) {
      if (!is_wayland()) {
        glfwGetWindowPos(win, &state->windowed_x, &state->windowed_y);
      }
      glfwGetWindowSize(win, &state->windowed_width, &state->windowed_height);
      glfwSetWindowMonitor(win, monitor, 0, 0, mode->width, mode->height,
                           mode->refreshRate);
      state->fullscreen = true;
    } else {
      if (is_wayland()) {
        glfwSetWindowMonitor(win, nullptr, 0, 0, state->windowed_width,
                             state->windowed_height, 0);
      } else {
        glfwSetWindowMonitor(win, nullptr, state->windowed_x, state->windowed_y,
                             state->windowed_width, state->windowed_height, 0);
      }
      state->fullscreen = false;
    }
  }
}

static constexpr auto as_null(auto k = VK_NULL_HANDLE) -> decltype(k) {
  return static_cast<decltype(k)>(VK_NULL_HANDLE);
}

auto main() -> std::int32_t {
  using namespace VkBindless;
  using GLFWPointer = std::unique_ptr<GLFWwindow, decltype(&destroy_glfw)>;

#ifndef _WIN32
  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
  if (!glfwInit()) {
    const char *error{};
    glfwGetError(&error);
    fprintf(stderr, "GLFW error: %s\n", error);
    return 1;
  }
  glfwSetErrorCallback([](int c, const char *d) {
    fprintf(stderr, "GLFW error %d: %s\n", c, d);
  });
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  std::int32_t initial_width = 800;
  std::int32_t initial_height = 600;

  GLFWPointer window(glfwCreateWindow(initial_width, initial_height,
                                      "Test Window", nullptr, nullptr),
                     destroy_glfw);

  WindowState state{};
  if (!is_wayland())
    glfwGetWindowPos(window.get(), &state.windowed_x, &state.windowed_y);
  glfwGetWindowSize(window.get(), &state.windowed_width,
                    &state.windowed_height);

  auto context = Context::create([win = window.get()](VkInstance instance) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, win, nullptr, &surface) !=
        VK_SUCCESS) {
      return as_null(surface);
    }
    return surface;
  });

  auto vulkan_context = std::move(context.value());

  glfwSetWindowUserPointer(window.get(), &state);
  glfwSetKeyCallback(window.get(), [](GLFWwindow *win, int key, int scancode,
                                      int action, int mods) {
    auto state = launder_cast<WindowState *>(glfwGetWindowUserPointer(win));
    key_callback(win, key, scancode, action, mods, state);
  });

  std::int32_t new_width = 0, new_height = 0;

  while (!glfwWindowShouldClose(window.get())) {
    glfwPollEvents();
    glfwGetFramebufferSize(window.get(), &new_width, &new_height);
    if (!new_width || !new_height)
      continue;

    auto& swapchain = vulkan_context->get_swapchain();
    if (static_cast<std::uint32_t>(new_width) != swapchain.width() || static_cast<std::uint32_t>(new_height) != swapchain.height()) {
        swapchain.resize(new_width, new_height);
        continue;
    }

    auto &buf = vulkan_context->acquire_command_buffer();
    auto swapchain_texture = vulkan_context->get_current_swapchain_texture();
    vulkan_context->submit(buf, swapchain_texture);
  }

  return 0;
}
