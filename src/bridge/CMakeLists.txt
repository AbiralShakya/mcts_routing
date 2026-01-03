# CMakeLists.txt for nextpnr bridge bindings
#
# Build instructions:
#   mkdir build && cd build
#   cmake .. -DNEXTPNR_SOURCE_DIR=/path/to/nextpnr
#   make
#
# This will produce _nextpnr_bindings.cpython-*.so

cmake_minimum_required(VERSION 3.14)
project(nextpnr_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    # Fallback: try to find pybind11 via pip
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED)
endif()

# nextpnr source directory (optional, for full integration)
set(NEXTPNR_SOURCE_DIR "" CACHE PATH "Path to nextpnr source directory")

# Create the Python module
pybind11_add_module(_nextpnr_bindings
    bindings.cpp
)

# Include directories
target_include_directories(_nextpnr_bindings PRIVATE
    ${Python3_INCLUDE_DIRS}
)

# If nextpnr source is available, link against it
if(NEXTPNR_SOURCE_DIR AND EXISTS ${NEXTPNR_SOURCE_DIR})
    message(STATUS "Found nextpnr source at: ${NEXTPNR_SOURCE_DIR}")

    target_include_directories(_nextpnr_bindings PRIVATE
        ${NEXTPNR_SOURCE_DIR}/common
        ${NEXTPNR_SOURCE_DIR}/xilinx
        ${NEXTPNR_SOURCE_DIR}/3rdparty
    )

    # Find Boost (required by nextpnr)
    find_package(Boost REQUIRED COMPONENTS filesystem thread program_options)
    target_link_libraries(_nextpnr_bindings PRIVATE
        Boost::filesystem
        Boost::thread
        Boost::program_options
    )

    # Link nextpnr library if built
    set(NEXTPNR_BUILD_DIR "${NEXTPNR_SOURCE_DIR}/build" CACHE PATH "nextpnr build directory")
    if(EXISTS ${NEXTPNR_BUILD_DIR})
        target_link_directories(_nextpnr_bindings PRIVATE ${NEXTPNR_BUILD_DIR})
        # target_link_libraries(_nextpnr_bindings PRIVATE nextpnr-xilinx)
    endif()
else()
    message(STATUS "nextpnr source not found - building stub bindings")
    message(STATUS "Set NEXTPNR_SOURCE_DIR to enable full router integration")
endif()

# Compiler flags
target_compile_options(_nextpnr_bindings PRIVATE
    -Wall
    -Wextra
    -O3
)

# Install target
install(TARGETS _nextpnr_bindings
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/mcts_routing/bridge
)

# Alternative: install to source directory for development
install(TARGETS _nextpnr_bindings
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)
