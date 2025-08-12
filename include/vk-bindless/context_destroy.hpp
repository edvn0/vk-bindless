#pragma once

#include "vk-bindless/forward.hpp"
#include "vk-bindless/handle.hpp"

namespace VkBindless {

auto context_destroy(IContext *context, TextureHandle) -> void;
auto context_destroy(IContext *context, SamplerHandle) -> void;
auto context_destroy(IContext *context, BufferHandle) -> void;
auto context_destroy(IContext *context, ComputePipelineHandle) -> void;
auto context_destroy(IContext *context, GraphicsPipelineHandle) -> void;
auto context_destroy(IContext *context, ShaderModuleHandle) -> void;
auto context_destroy(IContext *context, QueryPoolHandle) -> void;

} // namespace VkBindless