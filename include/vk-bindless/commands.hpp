#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vulkan/vulkan.h>

namespace VkBindless {

struct SubmitHandle {
  std::uint32_t buffer_index{0};
  std::uint32_t submit_identifier{0};

  SubmitHandle() = default;
  explicit SubmitHandle(std::uint64_t handle)
      : buffer_index(std::uint32_t(handle & 0xffffffff)),
        submit_identifier(std::uint32_t(handle >> 32)) {}

  auto empty() const { return submit_identifier == 0; }
  auto handle() const {
    return (static_cast<std::uint64_t>(submit_identifier) << 32) + buffer_index;
  }
};

struct CommandBufferWrapper {
  VkCommandBuffer command_buffer{VK_NULL_HANDLE};
  VkCommandBuffer command_buffer_allocated{VK_NULL_HANDLE};
  SubmitHandle handle{};
  VkFence fence{VK_NULL_HANDLE};
  VkSemaphore semaphore{VK_NULL_HANDLE};
  bool is_encoding{false};
};

class ImmediateCommands final {
public:
  static constexpr auto max_command_buffers = 64U;
  ImmediateCommands(VkDevice, std::uint32_t queue_family_index,
                    std::string_view);
  ~ImmediateCommands();

  auto acquire() -> CommandBufferWrapper &;
  auto submit(const CommandBufferWrapper &) -> SubmitHandle;

  auto wait_semaphore(VkSemaphore) -> void;
  auto signal_semaphore(VkSemaphore, std::uint64_t) -> void;
  auto acquire_last_submit_semaphore() -> VkSemaphore;

  auto get_last_submit_handle() const -> SubmitHandle {
    return last_submit_handle;
  }
  auto is_ready(SubmitHandle) const -> bool;
  auto wait(SubmitHandle) -> void;
  auto wait_all() -> void;

private:
  auto purge() -> void;

  VkDevice device{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  VkCommandPool command_pool{VK_NULL_HANDLE};
  std::uint32_t queue_family_index{0};
  std::string debug_name{};
  std::array<CommandBufferWrapper, max_command_buffers> command_buffers{};

  VkSemaphoreSubmitInfo last_submit_semaphore = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .pNext = nullptr,
      .semaphore = VK_NULL_HANDLE,
      .value = 0,
      .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      .deviceIndex = 0,
  };
  VkSemaphoreSubmitInfo wait_semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .pNext = nullptr,
      .semaphore = VK_NULL_HANDLE,
      .value = 0,
      .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      .deviceIndex = 0,
  };
  VkSemaphoreSubmitInfo signal_semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .pNext = nullptr,
      .semaphore = VK_NULL_HANDLE,
      .value = 0,
      .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      .deviceIndex = 0,
  };

  SubmitHandle last_submit_handle{};
  SubmitHandle next_submit_handle{};

  std::uint32_t available_command_buffers = max_command_buffers;
  std::uint32_t submit_counter = 1;
};

} // namespace VkBindless