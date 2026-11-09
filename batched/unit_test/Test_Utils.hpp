#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace TestUtils {

/// \brief Records the labels of all Kokkos allocations while alive, through
/// the Kokkos Tools allocation callback. Used to check that execute never
/// allocates and that a plan only allocates its own labelled tables.
class AllocationRecorder {
 public:
  AllocationRecorder() {
    labels().clear();
    Kokkos::Tools::Experimental::set_allocate_data_callback(&record);
  }
  ~AllocationRecorder() {
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  }
  AllocationRecorder(const AllocationRecorder &)            = delete;
  AllocationRecorder &operator=(const AllocationRecorder &) = delete;

  const std::vector<std::string> &recorded() const { return labels(); }
  void clear() { labels().clear(); }

 private:
  static std::vector<std::string> &labels() {
    static std::vector<std::string> recorded_labels;
    return recorded_labels;
  }
  static void record(const Kokkos_Profiling_SpaceHandle, const char *label,
                     const void *, const std::uint64_t) {
    if (!is_backend_team_scratch(label)) labels().emplace_back(label);
  }

  /// The team scratch pool that the backend sets up when a TeamPolicy kernel
  /// is launched (in the parallel_for, before execute runs), e.g.
  /// "Kokkos::CudaSpace::TeamScratchMemory". On CUDA it is re-allocated with
  /// 0 bytes at every launch when no level-1 scratch is requested, so a
  /// warm-up launch does not absorb it. It is launch machinery, not an
  /// allocation by execute, so it is not recorded.
  static bool is_backend_team_scratch(const std::string &label) {
    const std::string suffix = "::TeamScratchMemory";
    return label.rfind("Kokkos::", 0) == 0 && label.size() >= suffix.size() &&
           label.compare(label.size() - suffix.size(), suffix.size(), suffix) ==
               0;
  }
};

// ---- Team plans ----

using execution_space  = Kokkos::DefaultExecutionSpace;
using team_policy_type = Kokkos::TeamPolicy<execution_space>;
using member_type      = typename team_policy_type::member_type;

/// \brief How the team tests launch their kernels. team_size 0 means
/// Kokkos::AUTO; explicit sizes are clamped to what the backend allows for
/// the kernel (e.g. 1 on the Serial backend). max_vector requests
/// vector_length_max().
struct TeamConfig {
  int team_size;
  bool max_vector;
};

inline TeamConfig &team_config() {
  static TeamConfig config{0, false};
  return config;
}

/// \brief The configurations the team tests run with
inline std::vector<TeamConfig> team_configs() {
  return {{0, false}, {1, false}, {4, false}, {4, true}};
}

/// \brief First argument of a Plan: an execution space (serial plan) or a
/// TeamPolicy (team plan; only its type and execution space are used)
template <bool Team>
auto plan_policy() {
  if constexpr (Team) {
    return team_policy_type(1, Kokkos::AUTO);
  } else {
    return execution_space();
  }
}

/// \brief TeamPolicy over `league` teams for kernel `f`, following
/// team_config()
template <typename FunctorType>
team_policy_type make_team_policy(std::size_t league, const FunctorType &f) {
  const auto &config = team_config();
  const int vector_length =
      config.max_vector ? team_policy_type::vector_length_max() : 1;
  if (config.team_size == 0) {
    return team_policy_type(static_cast<int>(league), Kokkos::AUTO,
                            vector_length);
  }
  const int team_size_max =
      team_policy_type(static_cast<int>(league), 1, vector_length)
          .team_size_max(f, Kokkos::ParallelForTag());
  const int team_size = std::min(config.team_size, team_size_max);
  return team_policy_type(static_cast<int>(league), team_size, vector_length);
}

/// \brief Slice of a 2D batched view: the FFT axis `Axis` is kept and the
/// other dimension is fixed to `ib`
template <int Axis, typename ViewType>
KOKKOS_INLINE_FUNCTION auto batch_slice(const ViewType &view, std::size_t ib) {
  if constexpr (Axis == 0) {
    return Kokkos::subview(view, Kokkos::ALL, ib);
  } else {
    return Kokkos::subview(view, ib, Kokkos::ALL);
  }
}

/// \brief Relative L2 error ||actual - expected|| / ||expected|| (host)
/// Views of any rank with LayoutLeft/Right: their host mirrors are
/// contiguous and have the same layout, so they are compared element-wise
/// in memory order.
template <typename ViewType>
double relative_l2_error(const ViewType &actual, const ViewType &expected) {
  auto h_actual =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, actual);
  auto h_expected =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
  const auto *a = h_actual.data();
  const auto *e = h_expected.data();
  double diff = 0, norm = 0;
  for (std::size_t i = 0; i < h_actual.size(); ++i) {
    const double d = Kokkos::abs(a[i] - e[i]);
    const double r = Kokkos::abs(e[i]);
    diff += d * d;
    norm += r * r;
  }
  return norm == 0 ? std::sqrt(diff) : std::sqrt(diff / norm);
}

/// \brief Tolerance of development-plan.md §7: 10 eps log2(n)
template <typename T>
double fft_tolerance(std::size_t n) {
  return 10.0 * std::numeric_limits<T>::epsilon() *
         std::max(1.0, std::log2(static_cast<double>(n)));
}

}  // namespace TestUtils

#endif
