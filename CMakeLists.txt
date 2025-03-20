cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_NAMESPACE machines-in-motion)
set(PROJECT_NAME mim_solvers)
set(PROJECT_DESCRIPTION
    "Numerical solvers of the Machines in Motion Laboratory.")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)

add_compile_options(--warn-no-conversion)

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
option(BUILD_WITH_PROXSUITE "Build the ProxQP-based SQP solver" OFF)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(CHECK_RUNTIME_MALLOC "Check eigen mallocs" OFF)
option(ENABLE_VECTORIZATION
       "Enable vectorization and further processor-related optimizations" OFF)
option(BUILD_WITH_MULTITHREADS
       "Build the library with the Multithreading support (required OpenMP)"
       OFF)

# Project configuration
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})

if(ENABLE_VECTORIZATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# JRL-cmakemodule setup
include(${JRL_CMAKE_MODULES}/base.cmake)
# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${JRL_CMAKE_MODULES}/boost.cmake)
include(${JRL_CMAKE_MODULES}/apple.cmake)
include(${JRL_CMAKE_MODULES}/ide.cmake)
include(CMakeDependentOption)

# If needed, fix CMake policy for APPLE systems
apply_default_apple_configuration()

# Project dependencies
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
add_project_dependency(crocoddyl 3.0.0 REQUIRED)
if(BUILD_PYTHON_INTERFACE)
  add_project_dependency(eigenpy 2.7.10 REQUIRED)
endif()

if(BUILD_BENCHMARKS OR BUILD_TESTING)
  add_project_dependency(example-robot-data 4.0.7 REQUIRED PKG_CONFIG_REQUIRES
                         "example-robot-data >= 4.0.7")
endif()

if(BUILD_WITH_PROXSUITE)
  add_project_dependency(proxsuite REQUIRED)
  add_definitions(-DMIM_SOLVERS_WITH_PROXQP)
endif()

if(BUILD_WITH_MULTITHREADS)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if(BUILD_PYTHON_INTERFACE)
  set(PY_NAME ${PROJECT_NAME}_pywrap)
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_definitions(-DMIM_SOLVERS_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)

# Main Library
set(${PROJECT_NAME}_HEADERS
    include/${PROJECT_NAME}/kkt.hpp
    include/${PROJECT_NAME}/ddp.hpp
    include/${PROJECT_NAME}/fddp.hpp
    include/${PROJECT_NAME}/sqp.hpp
    include/${PROJECT_NAME}/csqp.hpp
    include/${PROJECT_NAME}/utils/callbacks.hpp)

set(${PROJECT_NAME}_SOURCES src/kkt.cpp src/ddp.cpp src/fddp.cpp src/sqp.cpp
                            src/csqp.cpp src/utils/callbacks.cpp)

if(BUILD_WITH_PROXSUITE)
  list(APPEND ${PROJECT_NAME}_HEADERS include/${PROJECT_NAME}/csqp_proxqp.hpp)
  list(APPEND ${PROJECT_NAME}_SOURCES src/csqp_proxqp.cpp)
endif(BUILD_WITH_PROXSUITE)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES}
                                   ${${PROJECT_NAME}_HEADERS})
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# Main Executable
target_link_libraries(${PROJECT_NAME} crocoddyl::crocoddyl)
if(BUILD_WITH_PROXSUITE)
  target_link_libraries(${PROJECT_NAME} proxsuite::proxsuite)
endif()

if(UNIX)
  get_relative_rpath(${CMAKE_INSTALL_LIBDIR} ${PROJECT_NAME}_INSTALL_RPATH)
  set_target_properties(
    ${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${${PROJECT_NAME}_INSTALL_RPATH}")
endif()

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

# Unit tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Installation
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
