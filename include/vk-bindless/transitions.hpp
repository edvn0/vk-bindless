#pragma once

#include <vulkan/vulkan.h>

namespace Transition {

inline auto
image(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout,
      VkImageLayout new_layout, const VkImageSubresourceRange &range,
      VkPipelineStageFlags2 src_stage_mask, VkAccessFlags2 src_access_mask,
      VkPipelineStageFlags2 dst_stage_mask, VkAccessFlags2 dst_access_mask,
      uint32_t src_queue_family = VK_QUEUE_FAMILY_IGNORED,
      uint32_t dst_queue_family = VK_QUEUE_FAMILY_IGNORED) -> void {
  VkImageMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .pNext = nullptr,
      .srcStageMask = src_stage_mask,
      .srcAccessMask = src_access_mask,
      .dstStageMask = dst_stage_mask,
      .dstAccessMask = dst_access_mask,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = src_queue_family,
      .dstQueueFamilyIndex = dst_queue_family,
      .image = image,
      .subresourceRange = range,
  };

  VkDependencyInfo dep_info{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .pNext = nullptr,
      .dependencyFlags = 0,
      .memoryBarrierCount = 0,
      .pMemoryBarriers = nullptr,
      .bufferMemoryBarrierCount = 0,
      .pBufferMemoryBarriers = nullptr,
      .imageMemoryBarrierCount = 1u,
      .pImageMemoryBarriers = &barrier,
  };

  vkCmdPipelineBarrier2(cmd, &dep_info);
}
} // namespace Transition
