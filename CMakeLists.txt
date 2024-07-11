cmake_minimum_required(VERSION 3.0)
project(YourProjectName)

# Add executable
add_executable(your_executable_name windyGridworld.cpp)

# Find required packages (including Boost)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GNUPLOT REQUIRED gnuplot-iostream)

# Include directories from gnuplot-iostream
include_directories(${GNUPLOT_INCLUDE_DIRS})

# Link against gnuplot-iostream libraries if necessary
target_link_libraries(your_executable_name ${GNUPLOT_LIBRARIES})

# Link against Boost libraries
find_package(Boost REQUIRED COMPONENTS iostreams)
target_link_libraries(your_executable_name Boost::iostreams)

# Other configurations and settings as needed
