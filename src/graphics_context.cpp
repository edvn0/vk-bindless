#include "vk-bindless/graphics_context.hpp"

namespace VkBindless {

auto
context_destroy(IContext* context, const TextureHandle handle) -> void
{
  if (context != nullptr)
    context->destroy_texture(handle);
}

auto
context_destroy(IContext* context, const SamplerHandle handle) -> void
{
  if (context != nullptr)
    context->destroy_sampler(handle);
}

}
