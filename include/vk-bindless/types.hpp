#pragma once

#include <format>
#include <memory>
#include <string_view>
#include <vulkan/vulkan_core.h>

namespace VkBindless {

template <typename T> using TypeDeleter = decltype(+[](const T *) -> void {});

template <typename T> using Unique = std::unique_ptr<T, TypeDeleter<T>>;

template <typename T>
constexpr auto default_deleter = +[](const T *) -> void {};

namespace detail {
std::string_view vk_result_to_string(VkResult result);

auto log_verification(std::string &&message) -> void;

template <typename... Args>
auto log_verification(std::format_string<Args...> fmt, Args &&...args) -> void {
  log_verification(std::format(fmt, std::forward<Args>(args)...));
}
} // namespace detail

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