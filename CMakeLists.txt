cmake_minimum_required(VERSION 3.12)

project(particles)

set(CXX_STANDARD 20)
set(CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-v")

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

include_directories(include ${CUDA_INCLUDE_DIRS})
enable_language(CUDA)

add_executable(particles main.cu glad.c)
target_link_libraries(particles PRIVATE glfw OpenGL::GL ${CUDA_LIBRARIES})
