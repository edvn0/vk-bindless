#include <vulkan/vulkan_core.h>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "vk-bindless/graphics_context.hpp"
#include "vk-bindless/holder.hpp"
#include "vk-bindless/object_pool.hpp"
#include "vk-bindless/vulkan_context.hpp"

using namespace VkBindless;

TEST_CASE("Vulkan Context Creation") {
  ContextError error;
  auto context =
      Context::create([](auto) -> VkSurfaceKHR { return VK_NULL_HANDLE; });
  CHECK(context.has_value() == false);
  CHECK(context.error().message == "Failed to select Vulkan physical device");

  // Assuming a valid surface is provided, this should succeed
  // auto surface = ...; // Create a valid VkSurfaceKHR
  // context = VkBindless::Context::create(surface);
  // CHECK(context.has_value() == true);
}

struct DummyTag {};
struct DummyImpl {
  int value;

  explicit DummyImpl(int v = 0) : value(v) {}

  auto operator<=>(const DummyImpl &other) const = default;
};

namespace VkBindless {
auto context_destroy(IContext *, Handle<DummyTag>) -> void {
  // No-op for this test
}
} // namespace VkBindless

TEST_CASE("Pool basic create/destroy") {
  Pool<DummyTag, DummyImpl> pool;

  auto handle = pool.create(DummyImpl{42});
  REQUIRE(handle.valid());

  auto obj = pool.get(handle);
  REQUIRE(obj.has_value());
  CHECK(obj.value()->value == 42);

  CHECK(pool.destroy(handle).has_value());

  REQUIRE(!pool.get(handle).has_value()); // stale handle
  REQUIRE(pool.get(handle).error() == PoolError::StaleHandle);
}

TEST_CASE("Pool double destroy is detected") {
  Pool<DummyTag, DummyImpl> pool;
  auto handle = pool.create(DummyImpl{7});
  auto second_handle = pool.create(DummyImpl{7});

  CHECK(pool.destroy(handle).has_value());
  auto result = pool.destroy(handle);
  REQUIRE(!result.has_value());
  REQUIRE(result.error() == PoolError::StaleHandle);
}

TEST_CASE("Pool reuse handle slot") {
  Pool<DummyTag, DummyImpl> pool;
  auto h1 = pool.create(DummyImpl{1});
  CHECK(pool.destroy(h1).has_value());
  auto h2 = pool.create(DummyImpl{2});

  CHECK(h1.index() == h2.index());
  CHECK(h1.generation() != h2.generation());
}

TEST_CASE("Pool clear removes all") {
  Pool<DummyTag, DummyImpl> pool;
  auto h1 = pool.create(DummyImpl{9});
  pool.clear();
  auto value = pool.get(h1);
  REQUIRE(!value.has_value());
  CHECK(value.error() == PoolError::IndexOutOfBounds);
}

TEST_CASE("Pool get with invalid handle returns error") {
  Pool<DummyTag, DummyImpl> pool;
  Handle<DummyTag> invalid;
  auto result = pool.get(invalid);
  REQUIRE(!result.has_value());
  CHECK(result.error() == PoolError::InvalidHandle);
}

TEST_CASE("Pool destroy with invalid handle returns error") {
  Pool<DummyTag, DummyImpl> pool;
  Handle<DummyTag> invalid;
  auto result = pool.destroy(invalid);
  REQUIRE(!result.has_value());
  CHECK(result.error() == PoolError::InvalidHandle);
}

TEST_CASE("Pool unsafe_handle returns valid handle for in-bounds index") {
  Pool<DummyTag, DummyImpl> pool;
  auto h = pool.create(DummyImpl{1});
  auto raw = pool.unsafe_handle(h.index());
  auto ptr = pool.get(raw);
  REQUIRE(ptr.has_value());
  CHECK(ptr.value()->value == 1);
}

TEST_CASE("Pool find_object returns correct handle") {
  Pool<DummyTag, DummyImpl> pool;
  auto h = pool.create(DummyImpl{99});
  auto ptr = pool.get(h);
  REQUIRE(ptr.has_value());

  auto found = pool.find_object(ptr.value());
  REQUIRE(found.valid());
  CHECK(found.index() == h.index());
}
