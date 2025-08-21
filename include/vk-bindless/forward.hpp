#pragma once

namespace VkBindless {

struct IContext;
struct IAllocator;
struct ICommandBuffer;

class CommandBuffer;

class Swapchain;
class Context;

class VkComputePipeline;
class VkGraphicsPipeline;

struct Framebuffer;

template<typename Tag, typename Impl>
class Pool;

} // namespace VkBindless