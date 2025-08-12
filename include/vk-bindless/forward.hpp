#pragma once

namespace VkBindless {

struct IContext;
struct IAllocator;
struct ICommandBuffer;

class Swapchain;
class Context;

template <typename Tag, typename Impl> class Pool;

} // namespace VkBindless