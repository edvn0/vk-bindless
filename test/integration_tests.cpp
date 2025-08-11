#include "doctest/doctest.h"

#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/holder.hpp"
#include "vk-bindless/object_pool.hpp"
#include "vk-bindless/vulkan_context.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

TEST_CASE("Integration test with real VkSurfaceKHR") {
  // Get a surface
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  // Dont show it
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow *window =
      glfwCreateWindow(800, 600, "Test Window", nullptr, nullptr);
  auto context =
      VkBindless::Context::create([win = window](VkInstance instance) {
        VkSurfaceKHR surface;
        if (glfwCreateWindowSurface(instance, win, nullptr, &surface) !=
            VK_SUCCESS) {
          glfwDestroyWindow(win);
          glfwTerminate();
        }
        return surface;
      });

  REQUIRE(context.has_value());
  auto vulkan_context = std::move(context.value());
  REQUIRE(vulkan_context != nullptr);

  glfwTerminate();
}

TEST_CASE("Create a white texture and make sure it's valid") {
  // Get a surface
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  // Dont show it
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow *window =
      glfwCreateWindow(800, 600, "Test Window", nullptr, nullptr);
  auto vulkan_context =
      VkBindless::Context::create([win = window](VkInstance instance) {
        VkSurfaceKHR surface;
        if (glfwCreateWindowSurface(instance, win, nullptr, &surface) !=
            VK_SUCCESS) {
          glfwDestroyWindow(win);
          glfwTerminate();
        }
        return surface;
      });

  REQUIRE(vulkan_context.has_value());
  auto &context = vulkan_context.value();

  using RGBAPixel = std::array<const std::uint8_t, 4>;

  const RGBAPixel dummy_white_texture = {255, 255, 255, 255};

  auto texture_desc = VkBindless::VkTextureDescription{
      .data = dummy_white_texture,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {1, 1, 1},
      .usage_flags = VkBindless::TextureUsageFlags::Sampled |
                     VkBindless::TextureUsageFlags::Storage,
      .debug_name = "White Texture"};

  auto texture_handle = VkBindless::VkTexture::create(*context, texture_desc);

  REQUIRE(texture_handle.valid());
  auto &texture_holder = texture_handle;

  auto maybe_texture = context->get_texture_pool().get(texture_holder);

  REQUIRE(maybe_texture.has_value());

  auto texture = maybe_texture.value();
  REQUIRE(texture->is_sampled());
  REQUIRE(texture->is_storage());
}