cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_NAMESPACE machines-in-motion)
set(PROJECT_NAME mim_solvers)
set(PROJECT_DESCRIPTION "Numerical solvers of the Machines in Motion Laboratory.")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

add_compile_options(--warn-no-conversion)
add_compile_options("-Wno-deprecated-declarations")

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
option(BUILD_WITH_HPIPM "Build the HPIPM-based SQP solver" ON)
option(BUILD_WITH_PROXQP "Build the ProxQP-based SQP solver" ON)

# Project configuration
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  IF(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
  ELSE()
    include(FetchContent)
    FetchContent_Declare("jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  ENDIF()
ENDIF()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/python.cmake")

# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

# Project dependencies
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(crocoddyl REQUIRED)
find_package(eigenpy REQUIRED)

add_project_dependency(crocoddyl 2.0.0 REQUIRED)
add_project_dependency(eigenpy 2.7.10 REQUIRED)

if(BUILD_WITH_PROXQP)
  find_package(proxsuite REQUIRED)
  add_project_dependency(proxsuite REQUIRED)
endif()

if(BUILD_WITH_HPIPM)
  set(blasfeo_INCLUDE_DIRS /opt/blasfeo/include)
  set(blasfeo_LIBRARIES /opt/blasfeo/lib/libblasfeo.a)
  set(hpipm_INCLUDE_DIRS /opt/hpipm/include)
  set(hpipm_LIBRARIES /opt/hpipm/lib/libhpipm.a)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

if(BUILD_PYTHON_INTERFACE)
  FINDPYTHON()
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
  SEARCH_FOR_BOOST_PYTHON()
endif()

# Main Library
set(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/ddp.hpp
  include/${PROJECT_NAME}/fddp.hpp
  include/${PROJECT_NAME}/sqp.hpp
  include/${PROJECT_NAME}/csqp.hpp
  include/${PROJECT_NAME}/csqp_proxqp.hpp
  include/${PROJECT_NAME}/csqp_hpipm.hpp
  )

set(${PROJECT_NAME}_SOURCES
  src/ddp.cpp
  src/fddp.cpp
  src/sqp.cpp
  src/csqp.cpp
  src/csqp_proxqp.cpp
  src/csqp_hpipm.cpp
  )

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
if(BUILD_WITH_HPIPM)
  target_include_directories(${PROJECT_NAME} PUBLIC ${hpipm_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${blasfeo_INCLUDE_DIRS})
endif()

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# Main Executable
TARGET_LINK_LIBRARIES(${PROJECT_NAME} crocoddyl::crocoddyl)
if(BUILD_WITH_PROXQP)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} proxsuite::proxsuite)
endif()
if(BUILD_WITH_HPIPM)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${hpipm_LIBRARIES})
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${blasfeo_LIBRARIES})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

# Unit tests
add_subdirectory(tests)

# Installation
install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
