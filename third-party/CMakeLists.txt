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

function(ADD_IMPLOT)
    if (TARGET implot::implot)
        return()
    endif()

    set(implot_dir "${CMAKE_CURRENT_SOURCE_DIR}/implot")

    if (EXISTS "${implot_dir}/implot.h")
        set(implot_sources
                "${implot_dir}/implot.cpp"
                "${implot_dir}/implot_items.cpp"
        )
        # optional demo:
        list(APPEND implot_sources "${implot_dir}/implot_demo.cpp")

        add_library(implot STATIC ${implot_sources})
        add_library(implot::implot ALIAS implot)

        target_include_directories(implot PUBLIC
                "${implot_dir}"
                "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
        )
        target_link_libraries(implot PUBLIC imgui::imgui)
    endif()
endfunction()

function(ADD_IMGUIZMO)
    if (TARGET imguizmo::imguizmo)
        return()
    endif()

    # Try common folder names
    set(_imguizmo_dirs
            "${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo"
            "${CMAKE_CURRENT_SOURCE_DIR}/imguizmo"
    )

    set(_found_dir "")
    foreach(d IN LISTS _imguizmo_dirs)
        if (EXISTS "${d}/ImGuizmo.h" AND EXISTS "${d}/ImGuizmo.cpp")
            set(_found_dir "${d}")
            break()
        endif()
    endforeach()

    if (NOT _found_dir)
        return()
    endif()

    add_library(imguizmo STATIC "${_found_dir}/ImGuizmo.cpp" "${_found_dir}/ImGuizmo.h" "${_found_dir}/ImCurveEdit.h" "${_found_dir}/ImCurveEdit.cpp" "${_found_dir}/ImGradient.h" "${_found_dir}/ImGradient.cpp" "${_found_dir}/ImSequencer.h" "${_found_dir}/ImSequencer.cpp")
    add_library(imguizmo::imguizmo ALIAS imguizmo)

    target_include_directories(imguizmo PUBLIC
            "${_found_dir}"
            "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
    )
    target_link_libraries(imguizmo PUBLIC imgui::imgui)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(imguizmo PRIVATE -Wno-unused-parameter -Wno-sign-conversion)
    elseif (MSVC)
        target_compile_options(imguizmo PRIVATE /wd4244 /wd4267)
    endif()
endfunction()

ADD_IMPLOT()
ADD_IMGUI()
ADD_IMGUIZMO()

if (ENABLE_TESTING)
    add_subdirectory(doctest)
endif()