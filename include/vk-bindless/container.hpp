#pragma once

#include <cstddef>
#include <ranges>
#include <span>

namespace VkBindless {

template<typename T>
auto
as_bytes(T* ptr, std::size_t count) -> std::span<std::byte>
{
  return { reinterpret_cast<std::byte*>(ptr), count * sizeof(T) };
}

template<typename T>
auto
as_bytes(const T* ptr, std::size_t count) -> std::span<const std::byte>
{
  return { reinterpret_cast<const std::byte*>(ptr), count * sizeof(T) };
}

template<std::ranges::contiguous_range R>
auto
as_bytes(R& range) -> std::span<std::byte>
{
  using T = std::ranges::range_value_t<R>;
  return as_bytes(reinterpret_cast<std::byte*>(std::data(range)),
                  std::size(range) * sizeof(T));
}

template<std::ranges::contiguous_range R>
auto
as_bytes(const R& range) -> std::span<const std::byte>
{
  using T = std::ranges::range_value_t<R>;
  return as_bytes(reinterpret_cast<const std::byte*>(std::data(range)),
                  std::size(range) * sizeof(T));
}

}