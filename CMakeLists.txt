cmake_minimum_required(VERSION 3.10)
project(off_policyPredictionProblem)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})  # Include current directory for matplotlibcpp.h

# Add the executable
add_executable(off_policyPredictionProblem off-policyPredictionProblem.cpp)

# Link the Python library
target_link_libraries(off_policyPredictionProblem ${Python3_LIBRARIES})
