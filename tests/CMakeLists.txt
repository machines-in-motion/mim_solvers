set(${PROJECT_NAME}_FACTORY_TEST
  # factory/solver.hpp
  # factory/solver.cpp
  # factory/model.hpp
  # factory/model.cpp
  factory/point-mass.hpp
  factory/point-mass.cpp
  )

add_library(${PROJECT_NAME}_unittest SHARED ${${PROJECT_NAME}_FACTORY_TEST})
target_link_libraries(
  ${PROJECT_NAME}_unittest PUBLIC ${PROJECT_NAME}
                                  example-robot-data::example-robot-data)

add_unit_test(test_solvers test_solvers.cpp)
target_link_libraries(test_solvers PUBLIC ${PROJECT_NAME}_unittest)

add_executable(timings timings.cpp)
target_link_libraries(timings PUBLIC ${PROJECT_NAME})