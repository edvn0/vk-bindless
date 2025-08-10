#pragma once

#include <bit>
#include <compare>

namespace VkBindless {

static constexpr std::uint32_t invalid_generation = 0U;

template<typename T>
class Handle final
{
  Handle(const std::uint32_t index, const std::uint32_t generation)
    : handle_index(index)
    , handle_generation(generation)
  {
  }

  std::uint32_t handle_index{ 0 };
  std::uint32_t handle_generation{ 0 };

  template<typename T_, typename TImpl>
  friend class Pool;

public:
  Handle() = default;

  [[nodiscard]] auto valid() const -> bool { return handle_generation != invalid_generation; }
  [[nodiscard]] auto empty() const -> bool { return handle_generation == invalid_generation; }
  explicit operator bool() const { return !valid(); }

  [[nodiscard]]auto index() const -> std::uint32_t { return handle_index; }
  [[nodiscard]]auto generation() const -> std::uint32_t { return handle_generation; }

  template<typename V = void*>
  [[nodiscard]]auto explicit_cast() const -> V*
  {
    return std::bit_cast<V*>(static_cast<std::ptrdiff_t>(handle_index));
  }

  auto operator<=>(const Handle& other) const = default;
};
static_assert(sizeof(Handle<class K>) == sizeof(std::uint64_t),
              "Handle size mismatch");

using ComputePipelineHandle = Handle<class ComputePipeline>;
using GraphicsPipelineHandle = Handle<class GraphicsPipeline>;
using ShaderModuleHandle = Handle<class ShaderModule>;
using SamplerHandle = Handle<class Sampler>;
using BufferHandle = Handle<class Buffer>;
using TextureHandle = Handle<class Texture>;

#define FOR_EACH_HANDLE_TYPE(MACRO) MACRO(TextureHandle, texture) \
  MACRO(SamplerHandle, sampler) \
  MACRO(BufferHandle, buffer) \
  MACRO(ShaderModuleHandle, shader_module) \
  MACRO(GraphicsPipelineHandle, graphics_pipeline) \
  MACRO(ComputePipelineHandle, compute_pipeline)

#define FOR_EACH_HANDLE_TYPE_WITH_TYPE_NAME(MACRO) MACRO(TextureHandle, texture, Texture) \
  MACRO(SamplerHandle, sampler, Sampler) \
  MACRO(BufferHandle, buffer, Buffer) \
  MACRO(ShaderModuleHandle, shader_module, ShaderModule) \
  MACRO(GraphicsPipelineHandle, graphics_pipeline, GraphicsPipeline) \
  MACRO(ComputePipelineHandle, compute_pipeline, ComputePipeline)

}