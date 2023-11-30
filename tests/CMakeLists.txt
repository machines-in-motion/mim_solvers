add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

ADD_UNIT_TEST(boost_sqp test_sqp.cpp)
target_link_libraries(boost_sqp ${PROJECT_NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
