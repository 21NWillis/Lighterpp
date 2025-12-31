# 1. Require a modern version of CMake
cmake_minimum_required(VERSION 3.10)

# 2. Name your project
project(Lighter++)

# 3. Set the C++ Standard (We want C++17 for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 4. Add compiler flags (Optional, but good for "Strict" coding)
if(MSVC)
    # Windows/Visual Studio flags
    add_compile_options(/W4 /O2)
else()
    # Linux/GCC/Clang flags
    # -O3 = Max optimization (Fast math)
    # -Wall = Warn all (Catch bugs early)
    add_compile_options(-O3 -Wall -Wextra)
endif()

# 5. Define the executable
# syntax: add_executable(NAME source1.cpp source2.cpp ...)
add_executable(Lighter++ 
    src/main.cpp
    src/tensor.cpp
    # You will add src/loader.cpp here later
)