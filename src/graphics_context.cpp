#include "vk-bindless/graphics_context.hpp"

namespace VkBindless {

#define CONTEXT_DESTROY_HANDLE_X_MACRO(type)                                   \
  auto context_destroy(IContext *ctx, type h) -> void {                        \
    if (ctx != nullptr) {                                                      \
      ctx->destroy(h);                                                         \
    }                                                                          \
  }
FOR_EACH_HANDLE_TYPE(CONTEXT_DESTROY_HANDLE_X_MACRO)
#undef CONTEXT_DESTROY_HANDLE_X_MACRO

} // namespace VkBindless
