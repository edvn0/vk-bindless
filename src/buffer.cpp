#include "vk-bindless/buffer.hpp"

#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/object_pool.hpp"
#include "vk-bindless/vulkan_context.hpp"

namespace VkBindless {

static constexpr auto storage_type_to_vk_memory_property_flags =
  [](StorageType storage) -> VkMemoryPropertyFlags {
  VkMemoryPropertyFlags memory_flags{ 0 };

  switch (storage) {
    case StorageType::DeviceLocal:
      memory_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      break;
    case StorageType::HostVisible:
      memory_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
    case StorageType::MemoryLess:
      memory_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                      VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
      break;
    case StorageType::HostCoherent:
    case StorageType::HostCached:
    case StorageType::DeviceCoherent:
    case StorageType::DeviceCached:
      break;
  }
  return memory_flags;
};

auto
VkDataBuffer::create(IContext& context, const BufferDescription& desc)
  -> Holder<BufferHandle>
{
  if (!desc.data.empty() && desc.size != desc.data.size_bytes()) {
    throw std::runtime_error(
      "BufferDescription size must match data size when data is provided");
  }

  auto storage = desc.storage;
  if (!context.use_staging() && (desc.storage == StorageType::DeviceLocal)) {
    storage = StorageType::HostVisible;
  }
  VkBufferUsageFlags usage_flags =
    storage == StorageType::DeviceLocal
      ? VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
      : 0;

  assert(desc.usage != BufferUsageFlags{ 0 });

  if ((desc.usage & BufferUsageFlags::IndexBuffer) != BufferUsageFlags{ 0 })
    usage_flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  if ((desc.usage & BufferUsageFlags::VertexBuffer) != BufferUsageFlags{ 0 })
    usage_flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if ((desc.usage & BufferUsageFlags::UniformBuffer) != BufferUsageFlags{ 0 })
    usage_flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  if ((desc.usage & BufferUsageFlags::StorageBuffer) != BufferUsageFlags{ 0 })
    usage_flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  if ((desc.usage & BufferUsageFlags::IndirectBuffer) != BufferUsageFlags{ 0 })
    usage_flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  auto memory_flags = storage_type_to_vk_memory_property_flags(storage);

  VkDataBuffer buffer{};
  buffer.size = desc.size;
  buffer.usage_flags = usage_flags;
  buffer.memory_flags = memory_flags;

  AllocationCreateInfo allocation_create_info{
    .usage = MemoryUsage::AutoPreferDevice,
    .map_memory = true,
    .preferred_memory_bits = 0,
    .required_memory_bits = 0,
    .debug_name = std::string{ desc.debug_name },
  };
  if (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    allocation_create_info.map_memory = true;
    allocation_create_info.preferred_memory_bits =
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    allocation_create_info.required_memory_bits =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }
  const VkBufferCreateInfo ci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .size = desc.size,
    .usage = usage_flags,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = 0,
    .pQueueFamilyIndices = nullptr,
  };
  auto&& [buf, allocation] =
    *context.get_allocator_implementation().allocate_buffer(
      ci, allocation_create_info);

  buffer.buffer = buf;
  buffer.allocation = allocation;
  if (!desc.data.empty()) {
    buffer.upload(desc.data);
  }

  assert(!desc.debug_name.empty());
  if (!desc.debug_name.empty()) {
    set_name_for_object(context.get_device(),
                        VK_OBJECT_TYPE_BUFFER,
                        buffer.buffer,
                        desc.debug_name);
  }

  return Holder<BufferHandle>{
    &context,
    context.get_buffer_pool().create(std::move(buffer)),
  };
}

auto
VkDataBuffer::flush_mapped_memory(IContext& context,
                                  std::uint64_t offset,
                                  std::uint64_t s) -> void
{
  if (!is_mapped())
    return;
  auto& allocator = context.get_allocator_implementation();
  allocator.flush_allocation(get_buffer(), offset, s);
}

auto
VkDataBuffer::invalidate_mapped_memory(IContext& context,
                                       std::uint64_t offset,
                                       std::uint64_t s) -> void
{
  if (!is_mapped())
    return;
  auto& allocator = context.get_allocator_implementation();
  allocator.invalidate_allocation(get_buffer(), offset, s);
}

}