#include "vk-bindless/graphics_context.hpp"

#include "vk-bindless/object_pool.hpp"

namespace VkBindless {

#define CONTEXT_DESTROY_HANDLE_X_MACRO(type)                                   \
  auto context_destroy(IContext *ctx, type h) -> void {                        \
    if (ctx != nullptr) {                                                      \
      ctx->destroy(h);                                                         \
    }                                                                          \
  }
FOR_EACH_HANDLE_TYPE(CONTEXT_DESTROY_HANDLE_X_MACRO)
#undef CONTEXT_DESTROY_HANDLE_X_MACRO

auto
IContext::get_format(TextureHandle handle) -> Format
{
  if (handle.empty()) return Format::Invalid;

  const auto* texture = *get_texture_pool().get(handle);
  if (!texture) {
    return Format::Invalid;
  }

  return texture->get_format();
}


} // namespace VkBindless
