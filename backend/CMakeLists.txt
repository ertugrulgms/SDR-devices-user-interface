cmake_minimum_required(VERSION 3.10)
project(sdr_core)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find pybind11
execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Convert string to list by replacing spaces with semicolons
string(REPLACE "-I" "" PYBIND11_INCLUDES "${PYBIND11_INCLUDES}")
string(REPLACE " " ";" PYBIND11_INCLUDES "${PYBIND11_INCLUDES}")

# Add python headers
find_package(Python3 COMPONENTS Development REQUIRED)

# Find SoapySDR
find_library(SOAPY_LIBRARY SoapySDR)
find_path(SOAPY_INCLUDE_DIR SoapySDR/Device.hpp)

if(NOT SOAPY_LIBRARY OR NOT SOAPY_INCLUDE_DIR)
    message(FATAL_ERROR "SoapySDR not found!")
endif()

# Since we might not have pybind11 cmake files system-wide if installed via pip,
# we'll build a standard shared library module for python
add_library(sdr_core MODULE sdr_core.cpp)
set_target_properties(sdr_core PROPERTIES PREFIX "" SUFFIX ".so")

target_include_directories(sdr_core PRIVATE 
    ${PYBIND11_INCLUDES} 
    ${Python3_INCLUDE_DIRS}
    ${SOAPY_INCLUDE_DIR}
)

target_link_libraries(sdr_core PRIVATE 
    ${SOAPY_LIBRARY}
)
