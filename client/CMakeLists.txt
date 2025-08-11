project(vk_bindless_client CXX)
add_executable(vk_bindless_client main.cpp)
target_link_libraries(vk_bindless_client VkBindless::VkBindless glfw)