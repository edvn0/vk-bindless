#include "vk-bindless/command_buffer.hpp"
#include <cassert>

namespace VkBindless {

CommandBuffer::~CommandBuffer() { assert(!is_rendering); }



} // namespace VkBindless