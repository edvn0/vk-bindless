#pragma once

// Stolen from https://github.com/SergiusTheBest/ScopeExit/

#include <utility>

#define SCOPE_EXIT_CAT2(x, y) x##y
#define SCOPE_EXIT_CAT(x, y) SCOPE_EXIT_CAT2(x, y)
#define SCOPE_EXIT const auto SCOPE_EXIT_CAT(scopeExit_, __COUNTER__) = ScopeExit::MakeScopeExit() += [&]

namespace ScopeExit
{
template<typename F>
class ScopeExit
{
public:
  explicit ScopeExit(F&& fn) : m_fn(fn)
  {
  }

  ~ScopeExit()
  {
    m_fn();
  }

  ScopeExit(ScopeExit&& other) noexcept
    : m_fn(std::move(other.m_fn))
  {
  }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

private:
  F m_fn;
};

struct MakeScopeExit
{
  template<typename F>
  ScopeExit<F> operator+=(F&& fn)
  {
    return ScopeExit<F>(std::forward<F>(fn));
  }
};
}