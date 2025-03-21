//
// Copyright (c) 2021-2023 INRIA
// https://github.com/humanoid-path-planner/hpp-fcl/blob/86d97039b3b49bd7256bce0a353c71d1286f5bf7/include/hpp/fcl/timings.h#L5
//

#ifndef MIM_SOLVERS_TIMINGS_FWD_H
#define MIM_SOLVERS_TIMINGS_FWD_H

#include <chrono>

namespace mim_solvers {

struct CPUTimes {
  double wall;
  double user;
  double system;

  CPUTimes() : wall(0), user(0), system(0) {}

  void clear() { wall = user = system = 0; }
};

///
/// @brief This class mimics the way "boost/timer/timer.hpp" operates while
/// using the modern std::chrono library.
///        Importantly, this class will only have an effect for C++11 and more.
///
struct Timer {
  typedef std::chrono::steady_clock clock_type;
  typedef clock_type::duration duration_type;

  /// \brief Default constructor for the timer
  ///
  /// \param[in] start_on_construction if true, the timer will be run just after
  /// the object is created
  Timer(const bool start_on_construction = true) : m_is_stopped(true) {
    if (start_on_construction) Timer::start();
  }

  CPUTimes elapsed() const {
    if (m_is_stopped) return m_times;

    CPUTimes current(m_times);
    std::chrono::time_point<std::chrono::steady_clock> current_clock =
        std::chrono::steady_clock::now();
    current.user += static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            current_clock - m_start)
                            .count()) *
                    1e-6;
    return current;
  }

  duration_type duration() const { return (m_end - m_start); }

  void start() {
    if (m_is_stopped) {
      m_is_stopped = false;
      m_times.clear();

      m_start = std::chrono::steady_clock::now();
    }
  }

  void stop() {
    if (m_is_stopped) return;
    m_is_stopped = true;

    m_end = std::chrono::steady_clock::now();
    m_times.user += static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            m_end - m_start)
                            .count()) *
                    1e-6;
  }

  void resume() {
    if (m_is_stopped) {
      m_start = std::chrono::steady_clock::now();
      m_is_stopped = false;
    }
  }

  bool is_stopped() const { return m_is_stopped; }

 protected:
  CPUTimes m_times;
  bool m_is_stopped;

  std::chrono::time_point<std::chrono::steady_clock> m_start, m_end;
};

}  // namespace mim_solvers

#endif  // ifndef MIM_SOLVERS_TIMINGS_FWD_H
