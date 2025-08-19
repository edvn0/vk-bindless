#include "vk-bindless/event_system.hpp"

#include <GLFW/glfw3.h>

namespace EventSystem {

auto
EventDispatcher::process_events() -> void
{
  glfwPollEvents();
}


}