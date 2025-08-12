#pragma once

namespace VkBindless {

struct IContext;
struct IAllocator;
struct ICommandBuffer;

class Context;

template <typename Tag, typename Impl> class Pool;

} // namespace VkBindless