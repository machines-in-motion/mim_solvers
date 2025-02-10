set(${PROJECT_NAME}_FACTORY_TEST
    factory/constraints.hpp
    factory/constraints.cpp
    factory/point-mass.hpp
    factory/point-mass.cpp
    factory/model.hpp
    factory/model.cpp
    factory/problem.hpp
    factory/problem.cpp
    factory/solver.hpp
    factory/solver.cpp)

add_library(${PROJECT_NAME}_unittest SHARED ${${PROJECT_NAME}_FACTORY_TEST})
target_link_libraries(
  ${PROJECT_NAME}_unittest PUBLIC ${PROJECT_NAME}
                                  example-robot-data::example-robot-data)

add_unit_test(test_solvers test_solvers.cpp)
target_link_libraries(test_solvers PUBLIC ${PROJECT_NAME}_unittest)

if(BUILD_PYTHON_INTERFACE)
  add_python_unit_test("py-test-clqr-convergence-all"
                       "tests/python/test_clqr_convergence_all.py" bindings)
  add_python_unit_test("py-test-clqr-osqp" "tests/python/test_clqr_osqp.py"
                       bindings)
  add_python_unit_test("py-test-clqr-stagewise-admm"
                       "tests/python/test_clqr_stagewise_admm.py" bindings)
  add_python_unit_test("py-test-lqr-sqp" "tests/python/test_lqr_problem.py"
                       bindings)
  add_python_unit_test("py-test-sqp-no-reg" "tests/python/test_sqp.py" bindings)
  add_python_unit_test("py-test-sqp-taichi-convergence"
                       "tests/python/test_sqp_taichi_convergence.py" bindings)
  add_python_unit_test("py-test-sqp-ur5-convergence"
                       "tests/python/test_sqp_ur5_convergence.py" bindings)
  add_python_unit_test("py-test-csqp-ur5" "tests/python/test_ur5_csqp.py"
                       bindings)
  add_python_unit_test("py-test-qp-ur5-all" "tests/python/test_qp_ur5_all.py"
                       bindings)
  add_python_unit_test("py-test-qp-solo-all" "tests/python/test_qp_solo_all.py"
                       bindings)
endif()
