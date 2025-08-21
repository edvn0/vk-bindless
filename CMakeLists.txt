cmake_minimum_required(VERSION 3.21)

include(CheckCXXSourceCompiles)

project(VkBindless LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Testing option
option(ENABLE_TESTING "Enable testing" ON)

find_package(Vulkan REQUIRED COMPONENTS glslang glslc SPIRV-Tools shaderc_combined glslangValidator)

check_cxx_source_compiles("
#include <expected>
int main() {
    std::expected<int, int> v{42};
    return v.value();
}
" HAS_STD_EXPECTED)

add_subdirectory(third-party)

add_library(CustomVMA STATIC src/vma_allocator_impl.cpp)
target_include_directories(CustomVMA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_SOURCE_DIR}/third-party/VulkanMemoryAllocator/include ${Vulkan_INCLUDE_DIRS})
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # -Wmissing-field-initializers,-Wunused-parameter
    target_compile_options(CustomVMA PRIVATE -Wno-missing-field-initializers -Wno-unused-parameter)
endif ()
if (HAS_STD_EXPECTED)
    target_compile_definitions(CustomVMA PUBLIC HAS_STD_EXPECTED)
else ()
    target_include_directories(CustomVMA PUBLIC ${CMAKE_SOURCE_DIR}/third-party/expected/include)
endif ()


add_library(${PROJECT_NAME} STATIC src/graphics_context.cpp src/vulkan_context.cpp src/texture.cpp src/types.cpp src/pipeline.cpp src/shader_compilation.cpp src/shader.cpp src/commands.cpp src/command_buffer.cpp src/swapchain.cpp src/event_system.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_SOURCE_DIR}/third-party/VulkanMemoryAllocator/include)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan vk-bootstrap CustomVMA glfw Vulkan::glslang Vulkan::SPIRV-Tools Vulkan::shaderc_combined)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/third-party/vk-bootstrap/src)
target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:IS_DEBUG>)

if (HAS_STD_EXPECTED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAS_STD_EXPECTED)
else ()
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/third-party/expected/include)
endif ()

if (ENABLE_TESTING)
    enable_testing()

    add_executable(test_vk_bindless test/main.cpp test/integration_tests.cpp test/shader_compilation_tests.cpp)
    target_link_libraries(test_vk_bindless VkBindless::VkBindless doctest Vulkan::Vulkan vk-bootstrap glfw)

    add_test(NAME test_vk_bindless COMMAND test_vk_bindless)
endif ()

add_library(VkBindless::VkBindless ALIAS VkBindless)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
    if (ENABLE_TESTING)
        target_compile_options(test_vk_bindless PRIVATE /W4 /WX)
    endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(COMPILE_FLAGS "-Wall;-Wextra;-Wpedantic;-Werror;-Wno-nullability-completeness;-Wno-nullability-extension")
    target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_FLAGS})
    if (ENABLE_TESTING)
        target_compile_options(test_vk_bindless PRIVATE ${COMPILE_FLAGS})
    endif ()
endif ()

add_subdirectory(client)
