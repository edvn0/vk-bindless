#include "vk-bindless/allocator_interface.hpp"
#include "vk-bindless/debug_name.hpp"
#include "vk-bindless/expected.hpp"

#define VMA_IMPLEMENTATION
/*#define VMA_DEBUG_LOG_FORMAT(format, ...) \
  do {                                                                         \
    printf((format), __VA_ARGS__);                                             \
    printf("\n");                                                              \
  } while (false)*/
#include <vk_mem_alloc.h>

#include <unordered_map>

namespace VkBindless {

class VmaAllocatorImpl final : public IAllocator
{
public:
  VmaAllocatorImpl(VkInstance instance,
                   VkPhysicalDevice physical_device,
                   VkDevice device)
  {
    VmaAllocatorCreateInfo create_info{};
    create_info.instance = instance;
    create_info.physicalDevice = physical_device;
    create_info.device = device;
    create_info.vulkanApiVersion = VK_API_VERSION_1_4;
    create_info.flags = VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
                        VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&create_info, &allocator) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create VMA allocator");
    }
  }

  ~VmaAllocatorImpl() override
  {
    if (allocator) {
      vmaDestroyAllocator(allocator);
    }
  }

  auto allocate_buffer(const VkBufferCreateInfo& buffer_info,
                       const AllocationCreateInfo& alloc_info)
    -> Expected<std::pair<VkBuffer, AllocationInfo>, AllocationError> override
  {
    VmaAllocationCreateInfo vma_alloc_info{};
    vma_alloc_info.usage = to_vma_usage(alloc_info.usage);
    if (alloc_info.map_memory) {
      vma_alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT |
                              VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }
    vma_alloc_info.requiredFlags = alloc_info.required_memory_bits;
    vma_alloc_info.preferredFlags = alloc_info.preferred_memory_bits;

    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;

    const auto result = vmaCreateBuffer(allocator,
                                        &buffer_info,
                                        &vma_alloc_info,
                                        &buffer,
                                        &allocation,
                                        &allocation_info);

    static auto index = 0;
    auto name = std::format("VMA_Buffer_{}_{}", alloc_info.debug_name, index++);
    set_name_for_object(allocator->m_hDevice,
                        VK_OBJECT_TYPE_DEVICE_MEMORY,
                        allocation_info.deviceMemory,
                        name.c_str());

    if (result != VK_SUCCESS) {
      return unexpected<AllocationError>(
        AllocationError{ "Failed to allocate buffer" });
    }

    // Store the allocation for later cleanup
    buffer_allocations[buffer] = allocation;

    AllocationInfo info{};
    info.memory = allocation_info.deviceMemory;
    info.offset = allocation_info.offset;
    info.size = allocation_info.size;
    info.mapped_data = allocation_info.pMappedData;

    return std::make_pair(buffer, info);
  }

  auto deallocate_buffer(const VkBuffer buffer) -> void override
  {
    if (const auto it = buffer_allocations.find(buffer);
        it != buffer_allocations.end()) {
      vmaDestroyBuffer(allocator, buffer, it->second);
      buffer_allocations.erase(it);
    }
  }

  auto allocate_image(const VkImageCreateInfo& image_info,
                      const AllocationCreateInfo& alloc_info)
    -> Expected<std::pair<VkImage, AllocationInfo>, AllocationError> override
  {
    VmaAllocationCreateInfo vma_alloc_info{};
    vma_alloc_info.usage = to_vma_usage(alloc_info.usage);
    if (alloc_info.map_memory) {
      vma_alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    vma_alloc_info.preferredFlags = alloc_info.preferred_memory_bits;
    vma_alloc_info.requiredFlags = alloc_info.required_memory_bits;

    VkImage image{};
    VmaAllocation allocation{ VK_NULL_HANDLE };
    VmaAllocationInfo allocation_info{};

    const auto result = vmaCreateImage(allocator,
                                       &image_info,
                                       &vma_alloc_info,
                                       &image,
                                       &allocation,
                                       &allocation_info);

    vmaSetAllocationName(allocator, allocation, alloc_info.debug_name.c_str());

    static auto index = 0;
    auto name = std::format("VMA_Image_{}_{}", alloc_info.debug_name, index++);
    set_name_for_object(allocator->m_hDevice,
                        VK_OBJECT_TYPE_DEVICE_MEMORY,
                        allocation_info.deviceMemory,
                        name.c_str());

    if (result != VK_SUCCESS) {
      return unexpected<AllocationError>(
        AllocationError{ "Failed to allocate image" });
    }

    image_allocations[image] = allocation;

    AllocationInfo info{};
    info.memory = allocation_info.deviceMemory;
    info.offset = allocation_info.offset;
    info.size = allocation_info.size;
    info.mapped_data = allocation_info.pMappedData;

    return std::make_pair(image, info);
  }

  auto deallocate_image(const VkImage image) -> void override
  {
    assert(image != VK_NULL_HANDLE && "Cannot deallocate a null image");
    if (const auto it = image_allocations.find(image);
        it != image_allocations.end()) {
      vmaDestroyImage(allocator, image, it->second);
      image_allocations.erase(it);
    }
  }

  auto map_memory(const VkBuffer buffer)
    -> Expected<void*, AllocationError> override
  {
#ifdef IS_RELEASE
    void* mapped_data;
    if (const auto result =
          vmaMapMemory(allocator, buffer_allocations.at(buffer), &mapped_data);
        result == VK_SUCCESS) {
      return mapped_data;
    }
    return unexpected<AllocationError>(
      AllocationError{ "Failed to map buffer memory" });
#else
    if (const auto it = buffer_allocations.find(buffer);
        it != buffer_allocations.end()) {
      void* mapped_data;
      if (const auto result = vmaMapMemory(allocator, it->second, &mapped_data);
          result == VK_SUCCESS) {
        return mapped_data;
      }
    }
    return unexpected<AllocationError>(
      AllocationError{ "Failed to map buffer memory" });
#endif
  }

  auto map_memory(const VkImage image)
    -> Expected<void*, AllocationError> override
  {
#ifdef IS_RELEASE
    void* mapped_data;
    if (const auto result =
          vmaMapMemory(allocator, image_allocations.at(image), &mapped_data);
        result == VK_SUCCESS) {
      return mapped_data;
    }
    return unexpected<AllocationError>(
      AllocationError{ "Failed to map image memory" });
#else
    if (const auto it = image_allocations.find(image);
        it != image_allocations.end()) {
      void* mapped_data;
      if (const auto result = vmaMapMemory(allocator, it->second, &mapped_data);
          result == VK_SUCCESS) {
        return mapped_data;
      }
    }
    return unexpected<AllocationError>(
      AllocationError{ "Failed to map image memory" });
#endif
  }

  auto unmap_memory(const VkBuffer buffer) -> void override
  {
#ifdef IS_RELEASE
    vmaUnmapMemory(allocator, buffer_allocations.at(buffer));
#else
    if (const auto it = buffer_allocations.find(buffer);
        it != buffer_allocations.end()) {
      vmaUnmapMemory(allocator, it->second);
    }
#endif
  }

  auto unmap_memory(const VkImage image) -> void override
  {
#ifdef IS_RELEASE
    vmaUnmapMemory(allocator, image_allocations.at(image));
#else
    if (const auto it = image_allocations.find(image);
        it != image_allocations.end()) {
      vmaUnmapMemory(allocator, it->second);
    }
#endif
  }

  auto flush_allocation(const VkBuffer buffer,
                        const VkDeviceSize offset,
                        const VkDeviceSize size) -> void override
  {
#ifdef IS_RELEASE
    vmaFlushAllocation(allocator, buffer_allocations.at(buffer), offset, size);
#else
    if (const auto it = buffer_allocations.find(buffer);
        it != buffer_allocations.end()) {
      vmaFlushAllocation(allocator, it->second, offset, size);
    }
#endif
  }

  auto flush_allocation(const VkImage image,
                        const VkDeviceSize offset,
                        const VkDeviceSize size) -> void override
  {
#ifdef IS_RELEASE
    vmaFlushAllocation(allocator, image_allocations.at(image), offset, size);
#else
    if (const auto it = image_allocations.find(image);
        it != image_allocations.end()) {
      vmaFlushAllocation(allocator, it->second, offset, size);
    }
#endif
  }

  auto invalidate_allocation(const VkBuffer buffer,
                             const VkDeviceSize offset,
                             const VkDeviceSize size) -> void override
  {
#ifdef IS_RELEASE
    vmaInvalidateAllocation(
      allocator, buffer_allocations.at(buffer), offset, size);
#else
    if (const auto it = buffer_allocations.find(buffer);
        it != buffer_allocations.end()) {
      vmaInvalidateAllocation(allocator, it->second, offset, size);
    }
#endif
  }

  auto invalidate_allocation(const VkImage image,
                             const VkDeviceSize offset,
                             const VkDeviceSize size) -> void override
  {
#ifdef IS_RELEASE
    vmaInvalidateAllocation(
      allocator, image_allocations.at(image), offset, size);
#else
    if (const auto it = image_allocations.find(image);
        it != image_allocations.end()) {
      vmaInvalidateAllocation(allocator, it->second, offset, size);
    }
#endif
  }

  [[nodiscard]] auto get_memory_usage() const
    -> std::pair<size_t, size_t> override
  {
    VmaBudget budget;
    vmaGetHeapBudgets(allocator, &budget);
    return { budget.usage, budget.budget };
  }

private:
  VmaAllocator allocator = VK_NULL_HANDLE;
  std::unordered_map<VkBuffer, VmaAllocation> buffer_allocations;
  std::unordered_map<VkImage, VmaAllocation> image_allocations;

  static auto to_vma_usage(MemoryUsage usage) -> VmaMemoryUsage
  {
    switch (usage) {
      case MemoryUsage::GpuOnly:
        return VMA_MEMORY_USAGE_GPU_ONLY;
      case MemoryUsage::CpuOnly:
        return VMA_MEMORY_USAGE_CPU_ONLY;
      case MemoryUsage::CpuToGpu:
        return VMA_MEMORY_USAGE_CPU_TO_GPU;
      case MemoryUsage::GpuToCpu:
        return VMA_MEMORY_USAGE_GPU_TO_CPU;
      case MemoryUsage::CpuCopy:
        return VMA_MEMORY_USAGE_CPU_COPY;
      case MemoryUsage::GpuLazilyAllocated:
        return VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED;
      case MemoryUsage::Auto:
        return VMA_MEMORY_USAGE_AUTO;
      case MemoryUsage::AutoPreferDevice:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
      case MemoryUsage::AutoPreferHost:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    return VMA_MEMORY_USAGE_AUTO;
  }
};

// Factory function to create the allocator
auto
IAllocator::create_allocator(VkInstance instance,
                             VkPhysicalDevice physical_device,
                             VkDevice device) -> Unique<IAllocator>
{
  return Unique<IAllocator>(
    new VmaAllocatorImpl(instance, physical_device, device));
}

} // namespace VkBindless