#pragma once

#include <bit>
#include <string_view>
#include <vulkan/vulkan.h>


namespace VkBindless {

inline auto
set_name_for_object(auto device,
                    const auto type,
                    const auto handle,
                    const std::string_view name)
{
  VkDebugUtilsObjectNameInfoEXT name_info{
    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
    .pNext = nullptr,
    .objectType = type,
    .objectHandle = std::bit_cast<std::uint64_t>(handle),
    .pObjectName = name.data(),
  };
  static auto vkSetDebugUtilsObjectNameEXT =
    reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
      vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
  if (vkSetDebugUtilsObjectNameEXT) {
    vkSetDebugUtilsObjectNameEXT(device, &name_info);
  }
};

}