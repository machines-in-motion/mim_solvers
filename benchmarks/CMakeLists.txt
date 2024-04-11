set(${PROJECT_NAME}_BENCHMARK ur5 solo12)

set(${PROJECT_NAME}_BENCHMARK_TOOLS timings.hpp timings.cpp)

add_library(${PROJECT_NAME}_BENCHMARK_TOOLS SHARED
            ${${PROJECT_NAME}_BENCHMARK_TOOLS})

foreach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})
  add_executable(${BENCHMARK_NAME} ${BENCHMARK_NAME}.cpp)
  target_link_libraries(
    ${BENCHMARK_NAME} ${PROJECT_NAME} example-robot-data::example-robot-data
    ${PROJECT_NAME}_BENCHMARK_TOOLS)
  add_custom_target("benchmarks-cpp-${BENCHMARK_NAME}" ${BENCHMARK_NAME}
                                                       \${INPUT})
endforeach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})
