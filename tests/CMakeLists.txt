set(${PROJECT_NAME}_FACTORY_TEST
  factory/crocoddyl/pinocchio_model.hpp
  factory/crocoddyl/state.hpp
  factory/crocoddyl/activation.hpp
  factory/crocoddyl/impulse.hpp
  factory/crocoddyl/cost.hpp
  factory/crocoddyl/action.hpp
  factory/solver.hpp
  # factory/diff-action.hpp
  factory/crocoddyl/pinocchio_model.cpp
  factory/crocoddyl/state.cpp
  factory/crocoddyl/activation.cpp
  factory/crocoddyl/impulse.cpp
  factory/crocoddyl/cost.cpp
  factory/crocoddyl/action.cpp
  factory/solver.cpp
  # factory/diff-action.cpp
  )

add_library(${PROJECT_NAME}_unittest SHARED ${${PROJECT_NAME}_FACTORY_TEST})
target_link_libraries(
  ${PROJECT_NAME}_unittest PUBLIC ${PROJECT_NAME}
                                  example-robot-data::example-robot-data)

# add_unit_test(test_states test_states.cpp)
# target_link_libraries(test_states PUBLIC ${PROJECT_NAME}_unittest)

# add_unit_test(test_actuations test_actuations.cpp)
# target_link_libraries(test_actuations PUBLIC ${PROJECT_NAME}_unittest)

add_unit_test(test_solvers test_solvers.cpp)
target_link_libraries(test_solvers PUBLIC ${PROJECT_NAME}_unittest)


# if(BUILD_PYTHON_INTERFACE)
#   add_subdirectory(python)
# endif()
