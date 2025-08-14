#include "vk-bindless/transitions.hpp"
#include "vk-bindless/vulkan_context.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <bit>
#include <cstdint>
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
static auto
launder_cast(const void* ptr) -> T
{
  return std::bit_cast<T>(ptr);
}

static auto
is_wayland() -> bool
{
  int platform = glfwGetPlatform();
  return platform == GLFW_PLATFORM_WAYLAND;
}

void
key_callback(GLFWwindow* win, int key, int, int action, int, void* user_data)
{
  if (action != GLFW_PRESS)
    return;
  auto state = launder_cast<WindowState*>(user_data);

  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(win, GLFW_TRUE);
    return;
  }

  if (key == GLFW_KEY_F11) {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    if (!state->fullscreen) {
      if (!is_wayland()) {
        glfwGetWindowPos(win, &state->windowed_x, &state->windowed_y);
      }
      glfwGetWindowSize(win, &state->windowed_width, &state->windowed_height);
      glfwSetWindowMonitor(
        win, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
      state->fullscreen = true;
    } else {
      if (is_wayland()) {
        glfwSetWindowMonitor(
          win, nullptr, 0, 0, state->windowed_width, state->windowed_height, 0);
      } else {
        glfwSetWindowMonitor(win,
                             nullptr,
                             state->windowed_x,
                             state->windowed_y,
                             state->windowed_width,
                             state->windowed_height,
                             0);
      }
      state->fullscreen = false;
    }
  }
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

#ifndef _WIN32
  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
  if (!glfwInit()) {
    const char* error{};
    glfwGetError(&error);
    std::print(std::cerr, "GLFW error: %s\n", error);
    return 1;
  }
  glfwSetErrorCallback([](int c, const char* d) {
    std::print(std::cerr, "GLFW error %d: %s\n", c, d);
  });
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  std::int32_t initial_width = 1920;
  std::int32_t initial_height = 1080;

  GLFWPointer window(
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
    if (glfwCreateWindowSurface(instance, win, nullptr, &surface) !=
        VK_SUCCESS) {
      return as_null(surface);
    }
    return surface;
  });

  auto vulkan_context = std::move(context.value());

  glfwSetWindowUserPointer(window.get(), &state);
  glfwSetKeyCallback(
    window.get(),
    [](GLFWwindow* win, int key, int scancode, int action, int mods) {
      auto s = launder_cast<WindowState*>(glfwGetWindowUserPointer(win));
      key_callback(win, key, scancode, action, mods, s);
    });

  std::int32_t new_width = 0;
  std::int32_t new_height = 0;

  while (!glfwWindowShouldClose(window.get())) {
    glfwPollEvents();
    glfwGetFramebufferSize(window.get(), &new_width, &new_height);
    if (!new_width || !new_height)
      continue;

    if (auto& swapchain = vulkan_context->get_swapchain();
        static_cast<std::uint32_t>(new_width) != swapchain.width() ||
        static_cast<std::uint32_t>(new_height) != swapchain.height()) {
      swapchain.resize(new_width, new_height);
      continue;
    }

    auto& buf = vulkan_context->acquire_command_buffer();

    constexpr auto make_floats = [](const auto val) {
      return std::array<float, 4>{ val, val, 0.0F, 1.0F };
    };

    buf.cmd_begin_rendering(
      { .color = { RenderPass::AttachmentDesc{
          .load_op = LoadOp::Clear,
          .clear_colour = make_floats(1.0F),
        }, }, },
      { .color = { Framebuffer::AttachmentDesc{
          .texture = vulkan_context->get_current_swapchain_texture(), }, },
         });
    buf.cmd_end_rendering();
    auto result = vulkan_context->submit(
      buf, vulkan_context->get_current_swapchain_texture());
    if (!result) {
      std::cerr << "Failed to submit command buffer." << std::endl;
      break;
    }
  }

  vulkan_context.reset();

  std::cout << "Application exited cleanly." << std::endl;

  return 0;
}
