project(vk_bindless_client CXX)
add_executable(vk_bindless_client main.cpp)
target_link_libraries(vk_bindless_client VkBindless::VkBindless glfw)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()