#include "vk-bindless/command_buffer.hpp"
#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/vulkan_context.hpp"

#include <cassert>

namespace VkBindless {

CommandBuffer::~CommandBuffer() { assert(!is_rendering); }

CommandBuffer::CommandBuffer(IContext &ctx)
    : context(static_cast<Context *>(&ctx)),
      wrapper(&context->get_immediate_commands().acquire()) {}

} // namespace VkBindless