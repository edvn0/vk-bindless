add_subdirectory(vk-bootstrap)
add_subdirectory(VulkanMemoryAllocator)
set(GLFW_BUILD_WAYLAND OFF)
set(GLFW_BUILD_X11 ON)
add_subdirectory(glfw)

if(NOT HAS_STD_EXPECTED)
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
    if(TARGET imgui::imgui)
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
    if(TARGET implot::implot)
        return()
    endif()

    set(implot_dir "${CMAKE_CURRENT_SOURCE_DIR}/implot")

    if(EXISTS "${implot_dir}/implot.h")
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
    if(TARGET imguizmo::imguizmo)
        return()
    endif()

    # Try common folder names
    set(_imguizmo_dirs
        "${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo"
        "${CMAKE_CURRENT_SOURCE_DIR}/imguizmo"
    )

    set(_found_dir "")

    foreach(d IN LISTS _imguizmo_dirs)
        if(EXISTS "${d}/ImGuizmo.h" AND EXISTS "${d}/ImGuizmo.cpp")
            set(_found_dir "${d}")
            break()
        endif()
    endforeach()

    if(NOT _found_dir)
        return()
    endif()

    add_library(imguizmo STATIC "${_found_dir}/ImGuizmo.cpp" "${_found_dir}/ImGuizmo.h" "${_found_dir}/ImCurveEdit.h" "${_found_dir}/ImCurveEdit.cpp" "${_found_dir}/ImGradient.h" "${_found_dir}/ImGradient.cpp" "${_found_dir}/ImSequencer.h" "${_found_dir}/ImSequencer.cpp")
    add_library(imguizmo::imguizmo ALIAS imguizmo)

    target_include_directories(imguizmo PUBLIC
        "${_found_dir}"
        "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
    )
    target_link_libraries(imguizmo PUBLIC imgui::imgui)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(imguizmo PRIVATE -Wno-unused-parameter -Wno-sign-conversion)
    elseif(MSVC)
        target_compile_options(imguizmo PRIVATE /wd4244 /wd4267)
    endif()
endfunction()

function(ADD_ASSIMP)
    if(TARGET assimp::assimp)
        return()
    endif()

    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
    set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

    # Disable all exporters
    set(ASSIMP_BUILD_NO_EXPORT ON)

    # Disable all importers first
    set(ASSIMP_IMPORTERS_ENABLED GLTF OBJ)
    set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT OFF)
    set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF)

    # Keep OBJ importer (default enabled)
    # Enable glTF importer (+ optional Draco)
    option(ASSIMP_BUILD_GLTF_IMPORTER "Enable glTF importer" ON)
    option(ASSIMP_BUILD_DRACO "Enable Draco compression for glTF" OFF)

    add_subdirectory(assimp)
endfunction()

function(ADD_MESHOPTIMIZER)
    if(TARGET meshoptimizer::meshoptimizer)
        return()
    endif()

    # Sensible defaults: no demo, no gltfpack, no shared libs, no Werror
    set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
    set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
    set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(MESHOPT_STABLE_EXPORTS OFF CACHE BOOL "" FORCE)
    set(MESHOPT_WERROR OFF CACHE BOOL "" FORCE)

    # Still allow install for consumers
    set(MESHOPT_INSTALL ON CACHE BOOL "" FORCE)

    add_subdirectory(meshoptimizer)
    add_library(meshoptimizer::meshoptimizer ALIAS meshoptimizer)
endfunction()

function(ADD_STB)
    if(TARGET stb::stb)
        return()
    endif()

    file(GLOB STB_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/stb/*.h"
    )

    add_library(stb INTERFACE ${STB_SOURCES})
    add_library(stb::stb ALIAS stb)

    target_include_directories(stb INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/stb"
    )
endfunction()

function (ADD_KTX)
    option(KTX_FEATURE_DOC "Create KTX documentation." OFF)
    option(KTX_FEATURE_JNI "Create Java bindings for libktx." OFF)
    option(KTX_FEATURE_PY "Create Python source distribution." OFF)
    option(KTX_FEATURE_TESTS "Create unit tests." OFF)
    option(KTX_FEATURE_TOOLS_CTS "Enable KTX CLI Tools CTS tests." OFF)
    option(KTX_FEATURE_ETC_UNPACK "ETC decoding support." ON)
    option(KTX_FEATURE_KTX1 "Enable KTX 1 support." ON)
    option(KTX_FEATURE_KTX2 "Enable KTX 2 support." ON)
    option(KTX_FEATURE_VK_UPLOAD "Enable Vulkan texture upload." ON)
    option(KTX_FEATURE_GL_UPLOAD "Enable OpenGL texture upload." ON)
    option(KTX_FEATURE_TOOLS "Create KTX tools" OFF)
    option(KTX_WERROR "Make all warnings in KTX code into errors." OFF)
    option(BASISU_SUPPORT_SSE "Compile with SSE support." ON)
    option(BASISU_SUPPORT_OPENCL "Compile with OpenCL support." OFF)
    option(BUILD_SHARED "Build shared" OFF)

    add_subdirectory(KTX-Software)
endfunction()

function(ADD_SPIRV_TOOLS)
    option(BUILD_SHARED "Build shared" OFF)
    set(SPIRV_TOOLS_BUILD_STATIC ON)
    set(SPIRV_SKIP_EXECUTABLES ON)
    add_subdirectory(SPIRV-Tools)
endfunction(ADD_SPIRV_TOOLS)

function(ADD_GLSLANG)
    add_subdirectory(glslang)
endfunction(ADD_GLSLANG)


ADD_IMPLOT()
ADD_IMGUI()
ADD_IMGUIZMO()
ADD_ASSIMP()
ADD_MESHOPTIMIZER()
ADD_STB()
ADD_KTX()
ADD_SPIRV_TOOLS()
ADD_GLSLANG()

if(ENABLE_TESTING)
    add_subdirectory(doctest)
endif()