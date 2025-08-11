#pragma once

#include "vk-bindless/handle.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <expected>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <vector>

namespace VkBindless {

enum class PoolError { InvalidHandle, StaleHandle, IndexOutOfBounds };

constexpr auto to_string(const PoolError error) -> std::string_view {
  switch (error) {
  case PoolError::InvalidHandle:
    return "Invalid handle";
  case PoolError::StaleHandle:
    return "Stale handle";
  case PoolError::IndexOutOfBounds:
    return "Index out of bounds";
  }
  return "Unknown error";
}

template <typename ObjectType, typename ImplObjectType> class Pool {
  static constexpr std::uint32_t list_end_sentinel = 0xffffffff;

  struct PoolEntryMetadata {
    std::uint32_t generation{1};
    std::uint32_t next_free{list_end_sentinel};
  };

  std::uint32_t free_list_head = list_end_sentinel;
  std::vector<ImplObjectType> objects;
  std::vector<PoolEntryMetadata> metadata;
  std::uint32_t num_objects = 0;

public:
  [[nodiscard]]
  auto create(ImplObjectType &&impl) -> Handle<ObjectType> {
    std::uint32_t index = 0;
    if (free_list_head != list_end_sentinel) {
      index = free_list_head;
      free_list_head = metadata[index].next_free;
      auto it_at_index = objects.begin() + index;
      objects.emplace(it_at_index, std::move(impl));
    } else {
      index = static_cast<std::uint32_t>(objects.size());
      objects.emplace_back(std::move(impl));
      metadata.emplace_back();
    }
    num_objects++;
    return Handle<ObjectType>(index, metadata[index].generation);
  }

  [[nodiscard]]
  auto destroy(Handle<ObjectType> handle) -> std::expected<void, PoolError> {
    if (!handle.valid()) {
      return std::unexpected(PoolError::InvalidHandle);
    }

    if (num_objects == 0) {
      return std::unexpected(PoolError::InvalidHandle);
    }

    const auto index = handle.index();
    if (index >= objects.size()) {
      return std::unexpected(PoolError::IndexOutOfBounds);
    }

    if (handle.generation() != metadata[index].generation) {
      return std::unexpected(PoolError::StaleHandle);
    }

    objects[index] = ImplObjectType{};
    auto &data = metadata[index];
    ++data.generation;
    data.next_free = free_list_head;
    free_list_head = index;
    num_objects--;
    return {};
  }

  [[nodiscard]]
  auto get(Handle<ObjectType> handle)
      -> std::expected<ImplObjectType *, PoolError> {
    if (!handle.valid()) {
      return std::unexpected(PoolError::InvalidHandle);
    }

    const auto index = handle.index();
    if (index >= objects.size()) {
      return std::unexpected(PoolError::IndexOutOfBounds);
    }

    if (handle.generation() != metadata[index].generation) {
      return std::unexpected(PoolError::StaleHandle);
    }

    return &objects[index];
  }

  [[nodiscard]] auto get(const Holder<Handle<ObjectType>> &holder) {
    return get(static_cast<Handle<ObjectType>>(holder));
  }

  [[nodiscard]] auto get(Handle<ObjectType> handle) const
      -> std::expected<const ImplObjectType *, PoolError> {
    if (!handle.valid()) {
      return std::unexpected(PoolError::InvalidHandle);
    }

    const auto index = handle.index();
    if (index >= objects.size()) {
      return std::unexpected(PoolError::IndexOutOfBounds);
    }

    if (handle.generation() != metadata[index].generation) {
      return std::unexpected(PoolError::StaleHandle);
    }

    return &objects[index];
  }

  [[nodiscard]] auto size() const -> std::uint32_t { return num_objects; }
  [[nodiscard]] auto empty() const -> bool { return num_objects == 0; }

  auto clear() -> void {
    free_list_head = list_end_sentinel;
    objects.clear();
    metadata.clear();
    num_objects = 0;
  }

  [[nodiscard]]
  auto unsafe_handle(const std::uint32_t index) const -> Handle<ObjectType> {
    if (index >= objects.size()) {
      return Handle<ObjectType>();
    }
    return Handle<ObjectType>(index, metadata[index].generation);
  }

  [[nodiscard]]
  auto find_object(const ImplObjectType *obj) -> Handle<ObjectType> {
    if (nullptr == obj || objects.empty()) {
      return Handle<ObjectType>();
    }

    auto found = std::ranges::find(objects, *obj);
    if (found == objects.end()) {
      return Handle<ObjectType>();
    }

    const auto index =
        static_cast<std::uint32_t>(std::distance(objects.begin(), found));
    return unsafe_handle(index);
  }

  auto at(const std::uint32_t index) -> decltype(objects.at(0)) {
    assert(!objects.empty() && "Pool is empty");
    assert(index < objects.size() && "Index out of bounds");
    return objects.at(index);
  }
  auto at(const std::uint32_t index) const -> decltype(objects.at(0)) {
    assert(!objects.empty() && "Pool is empty");
    assert(index < objects.size() && "Index out of bounds");
    return objects.at(index);
  }

  auto begin() const -> decltype(objects.begin()) { return objects.begin(); }
  auto end() const -> decltype(objects.end()) { return objects.end(); }
  auto cbegin() const -> decltype(objects.cbegin()) { return objects.cbegin(); }
  auto cend() const -> decltype(objects.cend()) { return objects.cend(); }
};

} // namespace VkBindless
