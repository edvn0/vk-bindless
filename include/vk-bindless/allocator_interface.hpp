#pragma once

#include "vk-bindless/expected.hpp"
#include "vk-bindless/types.hpp"

#include <cstddef>
#include <string>
#include <vulkan/vulkan.h>

namespace VkBindless {

struct AllocationError {
  std::string message;
};

struct AllocationInfo {
  VkDeviceMemory memory{};
  VkDeviceSize offset{};
  VkDeviceSize size{};
  void *mapped_data = nullptr;
};

enum struct MemoryUsage {
  GpuOnly,
  CpuOnly,
  CpuToGpu,
  GpuToCpu,
  CpuCopy,
  GpuLazilyAllocated,
  Auto,
  AutoPreferDevice,
  AutoPreferHost
};

constexpr std::uint32_t any_memory_type_bits = 0;

struct AllocationCreateInfo {
  MemoryUsage usage = MemoryUsage::Auto;
  bool map_memory = false;
  std::uint32_t preferred_memory_bits =
      any_memory_type_bits; // If set to 0, the allocator will choose the best
                            // memory type
  std::uint32_t required_memory_bits =
      any_memory_type_bits; // If set to 0, the allocator will choose the best
                            // memory type
  std::string debug_name;   // Optional debug name for the allocation
};

struct IAllocator {
  virtual ~IAllocator() = default;

  [[nodiscard]] virtual auto
  allocate_buffer(const VkBufferCreateInfo &buffer_info,
                  const AllocationCreateInfo &alloc_info)
      -> Expected<std::pair<VkBuffer, AllocationInfo>, AllocationError> = 0;

  virtual auto deallocate_buffer(VkBuffer buffer) -> void = 0;

  [[nodiscard]] virtual auto
  allocate_image(const VkImageCreateInfo &image_info,
                 const AllocationCreateInfo &alloc_info)
      -> Expected<std::pair<VkImage, AllocationInfo>, AllocationError> = 0;

  virtual auto deallocate_image(VkImage image) -> void = 0;

  [[nodiscard]] virtual auto map_memory(VkBuffer buffer)
      -> Expected<void *, AllocationError> = 0;
  [[nodiscard]] virtual auto map_memory(VkImage image)
      -> Expected<void *, AllocationError> = 0;

  virtual auto unmap_memory(VkBuffer buffer) -> void = 0;
  virtual auto unmap_memory(VkImage image) -> void = 0;

  virtual auto flush_allocation(VkBuffer buffer) -> void = 0;
  virtual auto flush_allocation(VkImage image) -> void = 0;
  virtual auto invalidate_allocation(VkBuffer buffer) -> void = 0;
  virtual auto invalidate_allocation(VkImage image) -> void = 0;

  [[nodiscard]] virtual auto get_memory_usage() const
      -> std::pair<size_t, size_t> = 0; // used, total

  static auto create_allocator(VkInstance, VkPhysicalDevice, VkDevice)
      -> Unique<IAllocator>;
};

} // namespace VkBindless