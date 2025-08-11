#pragma once

#include <format>
#include <memory>
#include <string_view>
#include <vulkan/vulkan_core.h>

namespace VkBindless {

namespace detail {

struct TypeDeleter {
  template <typename K> auto operator()(const K *ptr) const -> void {
    delete ptr;
  }
};

auto vk_result_to_string(VkResult result) -> std::string_view;

auto log_verification(std::string &&message) -> void;

template <typename... Args>
auto log_verification(std::format_string<Args...> fmt, Args &&...args) -> void {
  log_verification(std::format(fmt, std::forward<Args>(args)...));
}
} // namespace detail
template <typename T> using Unique = std::unique_ptr<T, detail::TypeDeleter>;

template <typename T> constexpr auto default_deleter = detail::TypeDeleter{};

#ifdef NDEBUG
#define VK_VERIFY(call) (call)
#else
#define VK_VERIFY(call)                                                        \
  do {                                                                         \
    VkResult vk_result = (call);                                               \
    if (vk_result != VK_SUCCESS) {                                             \
      detail::log_verification("Vulkan verification failed: {} ({})",          \
                               detail::vk_result_to_string(vk_result),         \
                               std::format("{}", #call));                      \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#endif

} // namespace VkBindless