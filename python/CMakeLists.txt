include("../cmake/python-helpers.cmake")

if(GENERATE_PYTHON_STUBS)
  include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)
endif(GENERATE_PYTHON_STUBS)

set(${PY_NAME}_SOURCES
  ddp.cpp
  fddp.cpp
  sqp.cpp
  csqp.cpp
  main.cpp
  )

if(BUILD_WITH_PROXSUITE)
  list(APPEND ${PY_NAME}_SOURCES csqp_proxqp.cpp)
endif(BUILD_WITH_PROXSUITE)

add_library(${PY_NAME}_pywrap MODULE ${${PY_NAME}_SOURCES})
MODERNIZE_TARGET_LINK_LIBRARIES(${PY_NAME}_pywrap SCOPE PUBLIC TARGETS eigenpy::eigenpy)
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME})
target_link_libraries(${PY_NAME}_pywrap PUBLIC crocoddyl::crocoddyl)

set_target_properties(${PY_NAME}_pywrap PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}_pywrap
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX})

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PY_NAME}_pywrap ${PYTHON_SITELIB} ${PY_NAME}_pywrap)
endif(GENERATE_PYTHON_STUBS)

set(${PY_NAME}_PYTHON mim_solvers/__init__.py)

SET(PYLIB_INSTALL_DIR ${PYTHON_SITELIB}/mim_solvers)
GET_RELATIVE_RPATH(${PYLIB_INSTALL_DIR} PYLIB_RPATH)
IF(UNIX)
  SET_TARGET_PROPERTIES(${${PY_NAME}_pywrap} PROPERTIES INSTALL_RPATH ${PYLIB_RPATH})
ENDIF()

install(TARGETS ${PY_NAME}_pywrap LIBRARY DESTINATION ${PYLIB_INSTALL_DIR})
install(FILES "${${PY_NAME}_PYTHON}" DESTINATION ${PYLIB_INSTALL_DIR})
