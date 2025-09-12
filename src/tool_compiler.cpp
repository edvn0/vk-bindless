#include <ranges>
#include <string_view>
#include <vector>

#include "vk-bindless/shader_compilation.hpp"

auto
main(int argc, char** argv) -> int
{
  auto paths = std::views::iota(0, argc) |
               std::views::transform([&args = argv](const auto i) {
                 return std::string_view{ args[i] };
               }) |
               std::ranges::to<std::vector<std::string_view>>();

  return 0;
}