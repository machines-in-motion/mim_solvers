if(GENERATE_PYTHON_STUBS)
  include(${JRL_CMAKE_MODULES}/stubs.cmake)
endif(GENERATE_PYTHON_STUBS)

set(${PY_NAME}_HEADERS utils/callback-base.hpp)
set(${PY_NAME}_SOURCES
    ddp.cpp
    fddp.cpp
    sqp.cpp
    csqp.cpp
    main.cpp
    utils/callbacks.cpp
    utils/callback-base.cpp)

if(BUILD_WITH_PROXSUITE)
  list(APPEND ${PY_NAME}_SOURCES csqp_proxqp.cpp)
endif(BUILD_WITH_PROXSUITE)

add_library(${PY_NAME} MODULE ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)

set_target_properties(
  ${PY_NAME}
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_NAME}
             PREFIX ""
             SUFFIX ${PYTHON_EXT_SUFFIX})

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PY_NAME} ${PYTHON_SITELIB}
                 ${PY_NAME})
endif(GENERATE_PYTHON_STUBS)

set(${PY_NAME}_PYTHON __init__.py)

set(PYLIB_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})
if(UNIX)
  get_relative_rpath(${PYLIB_INSTALL_DIR} PYLIB_RPATH)
  set_target_properties(${PY_NAME} PROPERTIES INSTALL_RPATH ${PYLIB_RPATH})
endif()

install(TARGETS ${PY_NAME} LIBRARY DESTINATION ${PYLIB_INSTALL_DIR})
foreach(source ${${PY_NAME}_PYTHON})
  python_install_on_site(${PROJECT_NAME} ${source})
endforeach()
