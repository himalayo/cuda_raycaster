cmake_minimum_required(VERSION 4.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
project(cuda_raycaster CUDA)

option(USE_NVTX "Enable NVTX markers for improved profiling (if aviailable)" ON)

set(CMAKE_CUDA_STANDARD 20)

find_package(SDL2 REQUIRED)

add_subdirectory(vendored/SDL_ttf EXCLUDE_FROM_ALL)

add_executable(cuda_raycaster main.cu
        raycast.cuh
        raycast.cu)

target_include_directories(cuda_raycaster PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(cuda_raycaster PRIVATE SDL2::SDL2 SDL2::SDL2main SDL2_ttf::SDL2_ttf)

set_target_properties(cuda_raycaster PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_raycaster PROPERTY CUDA_ARCHITECTURES native)

configure_file(sans.ttf sans.ttf COPYONLY)
# Define link libraries for the target
if(USE_NVTX)
    add_executable(benchmark benchmark.cu raycast.cuh raycast.cu)
    set_property(TARGET benchmark PROPERTY CUDA_ARCHITECTURES native)
    set_target_properties(benchmark PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    # Find the nvtx library using custom cmake module
    find_package(NVTX)
    # If found, apply
    if(NVTX_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
        set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES} "${NVTX_INCLUDE_DIRS}")
        if(NVTX_VERSION VERSION_LESS "3")
            set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} ${NVTX_LIBRARIES})
        endif()
        target_include_directories(benchmark PRIVATE ${NVTX_INCLUDE_DIRS})
    else()
        # If not found, disable.
        message("-- NVTX not available")
    endif()
endif()
