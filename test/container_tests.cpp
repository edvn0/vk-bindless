#include "doctest/doctest.h"

#include "vk-bindless/container.hpp"

using namespace VkBindless;

#include <array>
#include <vector>

void
test_as_bytes()
{
  {
    std::vector<int> v{ 1, 2, 3 };
    auto bytes = as_bytes(v);
    REQUIRE(bytes.size() == v.size() * sizeof(int));
    REQUIRE(reinterpret_cast<int*>(bytes.data())[0] == 1);
    REQUIRE(reinterpret_cast<int*>(bytes.data())[2] == 3);
  }

  {
    std::array<double, 2> a{ 1.5, 2.5 };
    auto bytes = as_bytes(a);
    REQUIRE(bytes.size() == a.size() * sizeof(double));
    REQUIRE(reinterpret_cast<double*>(bytes.data())[1] == 2.5);
  }

  {
    const std::vector<char> c{ 'a', 'b', 'c' };
    auto bytes = as_bytes(c);
    REQUIRE(bytes.size() == c.size() * sizeof(char));
    REQUIRE(bytes[1] == std::byte('b'));
  }

  {
    // Using std::span
    std::array<int, 3> a{ 10, 20, 30 };
    std::span<int> s(a);
    auto bytes = as_bytes(s);
    REQUIRE(bytes.size() == s.size() * sizeof(int));
    REQUIRE(reinterpret_cast<const int*>(bytes.data())[2] == 30);
  }
}