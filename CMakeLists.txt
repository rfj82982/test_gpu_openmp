cmake_minimum_required(VERSION 3.18)
project(HelloGPU CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP with GPU offloading
find_package(OpenMP REQUIRED)

# Set the NVIDIA HPC SDK compiler (assuming `nvhpc` is installed and available)
set(CMAKE_CXX_COMPILER nvc++)

# Compilation flags for NVIDIA HPC SDK
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu -fast")

# Add the executable
add_executable(hello_gpu_collapse hello_gpu_collapse.cpp)

# Link OpenMP
target_link_libraries(hello_gpu_collapse OpenMP::OpenMP_CXX)

# Set compilation flags using target_compile_options
target_compile_options(hello_gpu_collapse PUBLIC -mp=gpu -fast)
