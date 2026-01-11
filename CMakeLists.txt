# Require a modern version of CMake
cmake_minimum_required(VERSION 3.18)

project(Lighter++ LANGUAGES CXX)

# ============================================================================
# CUDA Option - OFF by default for GitHub Actions compatibility
# Build with: cmake -DUSE_CUDA=ON .. 
# ============================================================================
option(USE_CUDA "Enable CUDA acceleration" OFF)

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

# Define source files
set(SOURCES src/tensor.cpp src/loader.cpp src/ops.cpp src/model.cpp src/tokenizer.cpp)

# CUDA Configuration
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    
    # Add CUDA sources
    set(CUDA_SOURCES src/matmul.cu src/RMSNorm.cu)
    
    # Define USE_CUDA preprocessor macro
    add_definitions(-DUSE_CUDA)
    
    message(STATUS "CUDA acceleration: ENABLED")
else()
    set(CUDA_SOURCES "")
    message(STATUS "CUDA acceleration: DISABLED (CPU-only build)")
endif()

# Define the executables
add_executable(Lighter++ src/main.cpp ${SOURCES} ${CUDA_SOURCES})
add_executable(runtests src/runtests.cpp ${SOURCES} ${CUDA_SOURCES})