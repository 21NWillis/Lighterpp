# Require a modern version of CMake
cmake_minimum_required(VERSION 3.10)

project(Lighter++)

# Set the C++ Standard 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add compiler flags 
if(MSVC)
    # Windows/Visual Studio flags
    add_compile_options(/W4 /O2)
else()
    # Linux/GCC/Clang flags
    # -O3 = Max optimization
    # -Wall = Warn all 
    add_compile_options(-O3 -Wall -Wextra)
endif()

# Define the executables
set (SOURCES src/tensor.cpp src/loader.cpp src/ops.cpp src/model.cpp)


add_executable(Lighter++ src/main.cpp ${SOURCES})
add_executable(runtests src/runtests.cpp ${SOURCES})