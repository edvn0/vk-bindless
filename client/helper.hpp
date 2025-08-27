#pragma once

#include <GLFW/glfw3.h>
#include <cstdint>
#include <imgui.h>

auto
glfw_key_to_imgui_key(std::int32_t key) -> ImGuiKey;