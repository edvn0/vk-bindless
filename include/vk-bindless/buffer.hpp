#pragma once

#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/common.hpp"
#include "vk-bindless/forward.hpp"
#include "vk-bindless/holder.hpp"

#include <cstdint>

#include <ranges>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

namespace VkBindless {

enum class BufferUsageFlags : std::uint32_t
{
  TransferSrc = 0x00000001,
  TransferDst = 0x00000002,
  UniformTexelBuffer = 0x00000004,
  StorageTexelBuffer = 0x00000008,
  UniformBuffer = 0x00000010,
  StorageBuffer = 0x00000020,
  IndexBuffer = 0x00000040,
  VertexBuffer = 0x00000080,
  IndirectBuffer = 0x00000100,
  ShaderDeviceAddress = 0x00020000,
};
MAKE_BIT_FIELD(BufferUsageFlags)

enum class StorageType : std::uint32_t
{
  DeviceLocal,
  HostVisible,
  HostCoherent,
  HostCached,
  DeviceCoherent,
  DeviceCached,
  MemoryLess,
};

struct BufferDescription
{
  std::span<const std::byte>
    data = {}; // Optional initial data. Overwrites size if set.
  std::size_t size{ 0 };
  StorageType storage{ StorageType::HostVisible };
  BufferUsageFlags usage{ BufferUsageFlags::TransferDst };
  std::string_view debug_name{};
};

class VkDataBuffer
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  AllocationInfo allocation{};
  VkDeviceSize size{ 0 };
  VkBufferUsageFlags usage_flags{ 0 };
  VkMemoryPropertyFlags memory_flags{ 0 };

public:
  static auto create(IContext& context, const BufferDescription& desc)
    -> Holder<BufferHandle>;

  auto get_buffer() const -> VkBuffer { return buffer; }
  auto get_mapped_pointer() const -> void* { return allocation.mapped_data; }
  auto is_mapped() const -> bool { return allocation.mapped_data != nullptr; }
  auto get_size() const -> VkDeviceSize { return size; }
  auto get_memory() const -> VkDeviceMemory { return allocation.memory; }

  auto flush_mapped_memory(IContext&,
                           std::uint64_t offset = 0,
                           std::uint64_t size = VK_WHOLE_SIZE) -> void;
  auto invalidate_mapped_memory(IContext&,
                                std::uint64_t offset = 0,
                                std::uint64_t size = VK_WHOLE_SIZE) -> void;

  auto upload(std::ranges::contiguous_range auto R)
  {
    constexpr auto Size = std::ranges::range_size_t<decltype(R)>::value;
    if (!is_mapped()) {
      throw std::runtime_error("Buffer is not mapped");
    }
    if (R.size() * Size > size) {
      throw std::runtime_error("Data size exceeds buffer size");
    }
    std::memcpy(allocation.mapped_data, R.data(), R.size() * Size);
  }
  auto upload(std::span<const std::byte> data)
  {
    if (!is_mapped()) {
      throw std::runtime_error("Buffer is not mapped");
    }
    if (data.size() > size) {
      throw std::runtime_error("Data size exceeds buffer size");
    }
    std::memcpy(allocation.mapped_data, data.data(), data.size());
  }
  template<typename T>
  auto upload(std::span<const T> data)
  {
    if (!is_mapped()) {
      throw std::runtime_error("Buffer is not mapped");
    }
    if (data.size_bytes() > size) {
      throw std::runtime_error("Data size exceeds buffer size");
    }
    std::memcpy(allocation.mapped_data, data.data(), data.size_bytes());
  }
};

}