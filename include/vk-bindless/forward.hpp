#pragma once

namespace VkBindless {

struct IContext;
struct IAllocator;
struct ICommandBuffer;

class Swapchain;
class Context;

class VkComputePipeline;
class VkGraphicsPipeline;

template<typename Tag, typename Impl>
class Pool;

} // namespace VkBindless