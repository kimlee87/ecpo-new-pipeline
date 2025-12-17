cmake_minimum_required(VERSION 3.22)

# Set a name and a version number for your project:
project(
  cover-heurstic
  VERSION 0.0.1
  LANGUAGES CXX)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add Python bindings
find_package(pybind11 REQUIRED)

# Compile the Pybind11 modules - one for each library, because they cannot be
# mixed into one compilation unit.
pybind11_add_module(_cover_heuristic ecpo_pipeline/_cover_heuristic.cpp)

# Install the Python module shared library
install(TARGETS _cover_heuristic DESTINATION .)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
