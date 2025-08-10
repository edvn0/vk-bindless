#pragma once

#include "vk-bindless/forward.hpp"
#include "vk-bindless/context_destroy.hpp"

#include <compare>

namespace VkBindless {

template<typename T>
concept CanBeDestroyed = requires(IContext* context, T handle) {
  context_destroy(context, handle);
};

template<CanBeDestroyed HandleType>
class Holder final
{
public:
  Holder()noexcept = default;

  explicit Holder(IContext* ctx, HandleType handle)  noexcept
    : context(ctx)
    , handle(handle)
  {
  }
  ~Holder()noexcept { context_destroy(context, handle); }
  Holder(const Holder&) = delete;
  Holder(Holder&& other) noexcept
    : context(other.context)
    , handle(std::exchange(other.handle, HandleType{}))
  {
    other.context = nullptr;
  }
  auto operator=(const Holder&) noexcept-> Holder& = delete;
  auto operator=(Holder&& other) noexcept -> Holder&
  {
    if (this != &other) {
      context_destroy(context, handle);
      context = other.context;
      handle = std::exchange(other.handle, HandleType{});
      other.context = nullptr;
    }
    return *this;
  }
  auto operator=(std::nullptr_t)noexcept -> Holder&
  {
    if (context) {
      context_destroy(context, handle);
      context = nullptr;
      handle = HandleType{};
    }
    return *this;
  }
  explicit operator HandleType() const noexcept { return handle; }
  [[nodiscard]] auto valid() const noexcept-> bool { return handle.valid(); }
  [[nodiscard]] auto empty() const noexcept-> bool { return handle.empty(); }

  auto reset() noexcept-> void
  {
    context_destroy(context, handle);
    context = nullptr;
    handle = HandleType{};
  }
  auto release()noexcept-> HandleType
  {
    context = nullptr;
    return std::exchange(handle, HandleType{});
  }
  [[nodiscard]] auto index() const noexcept-> std::uint32_t { return handle.index(); }
  [[nodiscard]] auto generation() const noexcept-> std::uint32_t
  {
    return handle.generation();
  }
  template<typename V = void*>
  [[nodiscard]] auto explicit_cast() const -> V*
  {
    return handle.template explicit_cast<V>();
  }

  auto operator<=>(const Holder& other) const noexcept= default;

  static auto invalid() noexcept -> Holder
  {
    return Holder{nullptr, HandleType{}};
  }

private:
  IContext* context{ nullptr };
  HandleType handle;
};

}