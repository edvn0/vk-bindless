add_subdirectory(vk-bootstrap)
add_subdirectory(VulkanMemoryAllocator)
set(GLFW_BUILD_WAYLAND OFF)
set(GLFW_BUILD_X11 ON)
add_subdirectory(glfw)
if (NOT HAS_STD_EXPECTED)
    set(EXPECTED_BUILD_TESTS OFF)
    set(EXPECTED_BUILD_PACKAGE_DEB OFF)
    add_subdirectory(expected)
endif()

set(SPIRV_REFLECT_EXECUTABLE OFF)
set(SPIRV_REFLECT_STATIC_LIB ON)
set(SPIRV_REFLECT_INSTALL OFF)
add_subdirectory(SPIRV-Reflect)

add_subdirectory(glm)

function(ADD_IMGUI)
     # Prevent redefinition if called multiple times
    if (TARGET imgui::imgui)
        return()
    endif()

    
    file(GLOB IMGUI_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui/*.cpp"
    )
    
    # Add vulkan
    list(APPEND IMGUI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_vulkan.cpp")
    
    add_library(imgui STATIC ${IMGUI_SOURCES})
    add_library(imgui::imgui ALIAS imgui)

    target_include_directories(imgui PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include
        ${Vulkan_INCLUDE_DIRS}
    )
endfunction()

ADD_IMGUI()

if (ENABLE_TESTING)
    add_subdirectory(doctest)
endif()