cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_NAMESPACE machines-in-motion)
set(PROJECT_NAME mim_solvers)
set(PROJECT_DESCRIPTION "Numerical solvers of the Machines in Motion Laboratory.")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})
SET(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
SET(PROJECT_USE_CMAKE_EXPORT TRUE)
SET(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

add_compile_options(--warn-no-conversion)

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
option(BUILD_WITH_PROXSUITE "Build the ProxQP-based SQP solver" OFF)

# Project configuration
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\ncmake/base.cmake not found. Please run the following command first:\ngit submodule update --init\n")
ENDIF()

# JRL-cmakemodule setup
include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/apple.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ide.cmake)
include(CMakeDependentOption)

# If needed, fix CMake policy for APPLE systems
APPLY_DEFAULT_APPLE_CONFIGURATION()

# Project dependencies
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
add_project_dependency(crocoddyl 2.0.0 REQUIRED)
add_project_dependency(eigenpy 2.7.10 REQUIRED)
add_project_dependency(example-robot-data 4.0.7 REQUIRED PKG_CONFIG_REQUIRES
"example-robot-data >= 4.0.7")

if(BUILD_WITH_PROXSUITE)
  add_project_dependency(proxsuite REQUIRED)
  add_definitions(-DMIM_SOLVERS_WITH_PROXQP)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

if(BUILD_PYTHON_INTERFACE)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

# Main Library
set(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/kkt.hpp
  include/${PROJECT_NAME}/ddp.hpp
  include/${PROJECT_NAME}/fddp.hpp
  include/${PROJECT_NAME}/sqp.hpp
  include/${PROJECT_NAME}/csqp.hpp
  include/${PROJECT_NAME}/timings.hpp
  )

set(${PROJECT_NAME}_SOURCES
  src/kkt.cpp
  src/ddp.cpp
  src/fddp.cpp
  src/sqp.cpp
  src/csqp.cpp
  )

if(BUILD_WITH_PROXSUITE)
  list(APPEND ${PROJECT_NAME}_HEADERS include/${PROJECT_NAME}/csqp_proxqp.hpp)
  list(APPEND ${PROJECT_NAME}_SOURCES src/csqp_proxqp.cpp)
endif(BUILD_WITH_PROXSUITE)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# Main Executable
TARGET_LINK_LIBRARIES(${PROJECT_NAME} crocoddyl::crocoddyl)
if(BUILD_WITH_PROXSUITE)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} proxsuite::proxsuite)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

# Unit tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif(BUILD_TESTING)

# Installation
install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
