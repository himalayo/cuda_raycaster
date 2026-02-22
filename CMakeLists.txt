cmake_minimum_required(VERSION 4.1)
project(cuda_raycaster CUDA)

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