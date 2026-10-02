#ifndef BATCHED_PERFTEST_UTILS_HPP
#define BATCHED_PERFTEST_UTILS_HPP

#include "KokkosFFT_Batched.hpp"
#include <KokkosFFT.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <sstream>
#include <string>
#include <type_traits>

namespace BatchedFFTBenchmark {

using execution_space  = Kokkos::DefaultExecutionSpace;
using team_policy_type = Kokkos::TeamPolicy<execution_space>;
using member_type      = typename team_policy_type::member_type;
using range_policy_type =
    Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>;

/// \brief What executes the batched transforms
///   Serial   : Batched plan, one batch per iteration of a RangePolicy
///   Team     : Batched plan, one batch per team of a TeamPolicy, for each
///              team size in `team_sizes` (AUTO and explicit sizes)
///   KokkosFFT: one KokkosFFT plan over the whole batched view (reference)
enum class Variant { Serial, Team, KokkosFFT };

/// \brief Points per batched view (along FFT and batch dimensions together):
/// every size moves the same amount of data, and small sizes get many batches
/// (n = 8: 131072 batches; n = 1024: 1024 batches)
inline constexpr std::size_t total_points = std::size_t(1) << 20;

/// \brief Number of batches for FFTs of `points_per_fft` points
inline std::size_t batch_count(std::size_t points_per_fft) {
  const std::size_t nbatch = total_points / points_per_fft;
  return nbatch > 0 ? nbatch : 1;
}

/// \brief First argument of a KokkosFFT::Batched::Plan for the variant
template <Variant V>
auto plan_policy() {
  if constexpr (V == Variant::Team) {
    return team_policy_type(1, Kokkos::AUTO);
  } else {
    return execution_space();
  }
}

/// \brief Where the batch dimension of a batched view is: the first dimension
/// for LayoutRight views, the last one for LayoutLeft views. Either way one
/// batch is a contiguous block of memory.
template <typename ViewType>
inline constexpr bool is_batch_first_v =
    std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>;

/// \brief Batched view of `nbatch` slices of extents `n...`
///   LayoutLeft : (n..., nbatch)
///   LayoutRight: (nbatch, n...)
template <typename ViewType, typename... Extents>
ViewType make_batched_view(const std::string &label, std::size_t nbatch,
                           Extents... n) {
  static_assert(sizeof...(Extents) + 1 == ViewType::rank(),
                "make_batched_view: one extent per FFT dimension");
  if constexpr (is_batch_first_v<ViewType>) {
    return ViewType(label, nbatch, n...);
  } else {
    return ViewType(label, n..., nbatch);
  }
}

/// \brief Slice `ib` of a batched view (see is_batch_first_v)
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto batch_slice(const ViewType &v, std::size_t ib) {
  constexpr std::size_t rank = ViewType::rank();
  static_assert(rank >= 2 && rank <= 4, "batch_slice: rank 2 to 4");
  if constexpr (is_batch_first_v<ViewType>) {
    if constexpr (rank == 2) {
      return Kokkos::subview(v, ib, Kokkos::ALL);
    } else if constexpr (rank == 3) {
      return Kokkos::subview(v, ib, Kokkos::ALL, Kokkos::ALL);
    } else {
      return Kokkos::subview(v, ib, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    }
  } else {
    if constexpr (rank == 2) {
      return Kokkos::subview(v, Kokkos::ALL, ib);
    } else if constexpr (rank == 3) {
      return Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, ib);
    } else {
      return Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ib);
    }
  }
}

/// \brief Team sizes of the team variant (second benchmark argument, `team`).
/// 0 stands for Kokkos::AUTO; a size the backend cannot provide is skipped.
///   team_sizes     : run for every FFT size
///   scan_team_sizes: further sizes, run only for the FFT sizes of the
///                    team-size scan (a few sizes per case, to limit run time)
inline constexpr int team_size_auto    = 0;
inline constexpr int team_sizes[]      = {team_size_auto, 1, 8, 32};
inline constexpr int scan_team_sizes[] = {2, 4, 16, 64, 128, 256, 512, 1024};

/// \brief One batch per iteration of a RangePolicy, with a serial plan.
/// `dir` is ignored for R2C/C2R.
template <typename PlanType, typename InViewType, typename OutViewType>
struct SerialExecute {
  PlanType plan;
  InViewType in;
  OutViewType out;
  KokkosFFT::Direction dir;

  KOKKOS_FUNCTION void operator()(const std::size_t ib) const {
    if constexpr (PlanType::kind == KokkosFFT::Batched::TransformKind::C2C) {
      KokkosFFT::Batched::execute(plan, batch_slice(in, ib),
                                  batch_slice(out, ib), dir);
    } else {
      KokkosFFT::Batched::execute(plan, batch_slice(in, ib),
                                  batch_slice(out, ib));
    }
  }
};

/// \brief One batch per team of a TeamPolicy, with a team plan.
/// `dir` is ignored for R2C/C2R.
template <typename PlanType, typename InViewType, typename OutViewType>
struct TeamExecute {
  PlanType plan;
  InViewType in;
  OutViewType out;
  KokkosFFT::Direction dir;

  KOKKOS_FUNCTION void operator()(const member_type &member) const {
    const std::size_t ib = member.league_rank();
    if constexpr (PlanType::kind == KokkosFFT::Batched::TransformKind::C2C) {
      KokkosFFT::Batched::execute(member, plan, batch_slice(in, ib),
                                  batch_slice(out, ib), dir);
    } else {
      KokkosFFT::Batched::execute(member, plan, batch_slice(in, ib),
                                  batch_slice(out, ib));
    }
  }
};

/// \brief Nominal flop count of one C2C FFT of `n` points: 5 n log2(n)
inline double c2c_flops(double n) { return n > 1 ? 5.0 * n * std::log2(n) : 0; }

/// \brief Nominal flop count of one R2C FFT of `n` real points: 2.5 n log2(n)
inline double r2c_flops(double n) { return n > 1 ? 2.5 * n * std::log2(n) : 0; }

/// \brief Record one timed iteration: bytes read + written, nominal flops
inline void report_results(benchmark::State &state, double bytes, double flops,
                           double seconds) {
  state.SetIterationTime(seconds);
  state.counters["GB/s"] = benchmark::Counter(
      bytes / 1.0e9, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["GFLOP/s"] = benchmark::Counter(
      flops / 1.0e9, benchmark::Counter::kIsIterationInvariantRate);
}

/// \brief Time `n_iter` repetitions of `transform`, restoring the input from
/// `x0` before each one (execute overwrites its input; the copy is not timed)
template <typename SrcViewType, typename DstViewType, typename TransformType>
void timed_loop(benchmark::State &state, const SrcViewType &x0,
                const DstViewType &x, double bytes, double flops,
                const TransformType &transform) {
  for (auto _ : state) {
    Kokkos::deep_copy(x, x0);
    Kokkos::fence();
    Kokkos::Timer timer;
    transform();
    Kokkos::fence();
    report_results(state, bytes, flops, timer.seconds());
  }
}

/// \brief Benchmark a Batched plan (forward transform) on every batch of
/// `x`, writing to `y`.
///
/// Serial plans: a RangePolicy over the batches.
/// Team plans: one team per batch. The team size is the second benchmark
/// argument (0: Kokkos::AUTO); the size actually used and the vector length
/// are reported as the counters `team_size` and `vector_length`. For AUTO,
/// the size is the one Kokkos resolves for this functor
/// (team_size_recommended), queried outside the timed region.
template <typename PlanType, typename InViewType, typename OutViewType>
void run_batched(benchmark::State &state, const PlanType &plan,
                 const InViewType &x0, const InViewType &x,
                 const OutViewType &y, std::size_t nbatch, double bytes,
                 double flops) {
  const auto dir = KokkosFFT::Direction::forward;
  if constexpr (PlanType::is_team) {
    const TeamExecute<PlanType, InViewType, OutViewType> functor{plan, x, y,
                                                                 dir};
    const int requested = static_cast<int>(state.range(1));
    const team_policy_type auto_policy(nbatch, Kokkos::AUTO);
    const int team_size_max =
        auto_policy.team_size_max(functor, Kokkos::ParallelForTag());
    if (requested > team_size_max) {
      state.SkipWithMessage("team size " + std::to_string(requested) +
                            " not available (max " +
                            std::to_string(team_size_max) + ")");
      return;
    }
    const bool is_auto  = requested == team_size_auto;
    const int team_size = is_auto ? auto_policy.team_size_recommended(
                                        functor, Kokkos::ParallelForTag())
                                  : requested;
    const team_policy_type policy =
        is_auto ? auto_policy : team_policy_type(nbatch, requested);
    timed_loop(state, x0, x, bytes, flops, [&]() {
      Kokkos::parallel_for("batched_fft_team", policy, functor);
    });
    state.counters["team_size"] = static_cast<double>(team_size);
    state.counters["vector_length"] =
        static_cast<double>(policy.impl_vector_length());
  } else {
    const SerialExecute<PlanType, InViewType, OutViewType> functor{plan, x, y,
                                                                   dir};
    const range_policy_type policy(0, nbatch);
    timed_loop(state, x0, x, bytes, flops, [&]() {
      Kokkos::parallel_for("batched_fft_serial", policy, functor);
    });
  }
}

/// \brief Register the FFT sizes of a benchmark (argument `n`). The team
/// variant also gets a team size (argument `team`, 0: Kokkos::AUTO): every
/// size of `team_sizes` for all FFT sizes, and the sizes of `scan_team_sizes`
/// for the FFT sizes in `scan_sizes`.
template <Variant V>
void apply_sizes(benchmark::Benchmark *b, std::initializer_list<int> sizes,
                 [[maybe_unused]] std::initializer_list<int> scan_sizes) {
  if constexpr (V == Variant::Team) {
    b->ArgNames({"n", "team"});
    for (int team_size : team_sizes) {
      for (int n : sizes) {
        b->Args({n, team_size});
      }
    }
    for (int team_size : scan_team_sizes) {
      for (int n : scan_sizes) {
        b->Args({n, team_size});
      }
    }
  } else {
    b->ArgName("n");
    for (int n : sizes) {
      b->Arg(n);
    }
  }
  b->UseManualTime();
}

/// \brief Kokkos configuration and OpenMP settings in the benchmark context
inline void add_benchmark_context() {
  std::ostringstream msg;
  Kokkos::print_configuration(msg, false);
  std::stringstream ss{msg.str()};
  for (std::string line; std::getline(ss, line, '\n');) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    auto trim = [](const std::string &s) {
      const auto b = s.find_first_not_of(" :");
      const auto e = s.find_last_not_of(" :");
      return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
    };
    const auto key   = trim(line.substr(0, colon));
    const auto value = trim(line.substr(colon + 1));
    if (!key.empty() && !value.empty()) benchmark::AddCustomContext(key, value);
  }
  for (const char *name : {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"}) {
    if (const char *value = std::getenv(name))
      benchmark::AddCustomContext(name, value);
  }
}

/// \brief Label of a variant in benchmark names
template <Variant V>
const char *variant_name() {
  if constexpr (V == Variant::Serial) {
    return "BatchedSerial";
  } else if constexpr (V == Variant::Team) {
    return "BatchedTeam";
  } else {
    return "KokkosFFT";
  }
}

}  // namespace BatchedFFTBenchmark

#endif
