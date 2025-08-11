cmake_minimum_required(VERSION 3.21)
project(VkBindless LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DISABLE_EXCEPTIONS ON)

add_subdirectory(third-party)

find_package(Vulkan REQUIRED)

add_library(${PROJECT_NAME} STATIC src/graphics_context.cpp src/vulkan_context.cpp src/texture.cpp src/types.cpp src/vma_allocator_impl.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan vk-bootstrap GPUOpen::VulkanMemoryAllocator)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/third-party/vk-bootstrap/src)

if(ENABLE_TESTING)
    enable_testing()

    add_executable(test_vk_bindless test/main.cpp test/integration_tests.cpp)
    target_link_libraries(test_vk_bindless VkBindless::VkBindless doctest Vulkan::Vulkan vk-bootstrap glfw)

    add_test(NAME test_vk_bindless COMMAND test_vk_bindless)
endif()

add_library(VkBindless::VkBindless ALIAS VkBindless)

# if clang/gcc, -Wnonullability-completeness
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-nullability-completeness)
endif()

add_subdirectory(client)
