cmake_minimum_required(VERSION 3.21)
project(VkBindless LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DISABLE_EXCEPTIONS ON)

find_package(Vulkan REQUIRED)

add_library(${PROJECT_NAME} STATIC src/graphics_context.cpp src/vulkan_context.cpp src/texture.cpp src/types.cpp src/vma_allocator_impl.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap PUBLIC GPUOpen::VulkanMemoryAllocator)

if(ENABLE_TESTING)
    find_package(doctest REQUIRED CONFIG)

    enable_testing()

    add_executable(test_vk_bindless test/main.cpp test/integration_tests.cpp)
    target_link_libraries(test_vk_bindless VkBindless::VkBindless doctest::doctest Vulkan::Vulkan vk-bootstrap::vk-bootstrap glfw)

    add_test(NAME test_vk_bindless COMMAND test_vk_bindless)
endif()

add_library(VkBindless::VkBindless ALIAS VkBindless)
