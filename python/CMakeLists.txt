set(${PY_NAME}_SOURCES
  sqp.cpp
  main.cpp
  )

add_library(${PY_NAME}_pywrap SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME})
target_include_directories(${PY_NAME}_pywrap SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(${PY_NAME}_pywrap PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}_pywrap
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX})
TARGET_LINK_BOOST_PYTHON(${PY_NAME}_pywrap PRIVATE)
install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON
  __init__.py
  )

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()
