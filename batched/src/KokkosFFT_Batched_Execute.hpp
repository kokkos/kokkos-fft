#ifndef KOKKOSFFT_BATCHED_EXECUTE_HPP
#define KOKKOSFFT_BATCHED_EXECUTE_HPP

#include "KokkosFFT_Batched_Accessors.hpp"
#include "KokkosFFT_Batched_Base_Types.hpp"
#include "KokkosFFT_Batched_Concepts.hpp"
#include "KokkosFFT_Batched_Kernels_1D.hpp"
#include "KokkosFFT_Batched_Kernels_Odd.hpp"
#include "KokkosFFT_Batched_Levels.hpp"
#include "KokkosFFT_Batched_Plan.hpp"
#include "KokkosFFT_Batched_Traits.hpp"
#include <KokkosFFT_Common_Types.hpp>
#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

/// \brief Which buffer currently holds the data. Threaded through the plan
/// tree, since the parity of the number of stages is only known at runtime.
struct BufferState {
  bool data_in_out = false;
};

// Leaf passes. A leaf transforms all the lines of the slice along dimension
// `SliceDim` (one line for a 1-D slice), using `in` and `out` as the only
// work buffers. How the lines are run depends on the level
// (KokkosFFT_Batched_Levels.hpp):
// - serial level (merges_lines = false): line by line, all the stages of a
//   line before the next one (`level.for_lines`);
// - team level (merges_lines = true), slices of rank 2 or more: stage by
//   stage, with the butterflies of all the lines of a stage distributed over
//   the team in one for_each (`c2c_lines`, `for_each_in_lines`), and a barrier
//   after each stage. Every thread then has work whatever the number of
//   lines, and the work items are ordered so that neighbouring threads access
//   neighbouring memory;
// - team level, 1-D slices (one line): the whole team on that line, through
//   `level.for_lines` and the per-line kernels (see merged_pass_v).
// The odd-length real kernels (KokkosFFT_Batched_Kernels_Odd.hpp) are always
// run line by line through `level.for_lines`.

/// \brief Whether a pass over the lines of `ViewType` runs merged (stage by
/// stage over all lines, see above): on levels that merge lines, for slices
/// of rank 2 or more. A 1-D slice is a single line: it runs line by line,
/// i.e. the whole level on that line with the per-line kernels, which need no
/// line map, no pair of line accessors and no index split. In the merged
/// path those added 2-10 registers per thread to the 1-D team kernels
/// (optimization/measurements/2026-10-02_c65ff7b_A100, G1e).
template <typename LevelType, typename ViewType>
inline constexpr bool merged_pass_v =
    LevelType::merges_lines && (ViewType::rank() > 1);

/// \brief Copy line `src` to line `dst` (n elements)
template <typename LevelType, typename SrcLineType, typename DstLineType>
KOKKOS_FUNCTION void copy_line(const LevelType &level, std::size_t n,
                               const SrcLineType &src, const DstLineType &dst) {
  level.for_each(n, [&](std::size_t i) { dst.store(i, src.load(i)); });
  level.barrier();
}

/// \brief C2C of all lines of the complex view `data` along SliceDim, with
/// the whole real view `work` as flat complex scratch. Each line ends back
/// in `data` (copied if the stage count is odd). Lines that run concurrently
/// use disjoint scratch regions (`slot`), so at most work.size() / (2 n) lines
/// run at once: they are processed in rounds (by for_lines, or here when the
/// level merges lines).
template <int SliceDim, KokkosFFT::Direction Dir, typename LevelType,
          typename PlanType, typename DataViewType, typename WorkViewType>
KOKKOS_FUNCTION void c2c_lines_with_scratch(const LevelType &level,
                                            const PlanType &node,
                                            const DataViewType &data,
                                            const WorkViewType &work) {
  const std::size_t n = node.length(0);
  // n >= 1 by construction (plans reject zero lengths); this makes it
  // explicit for the division below (and for static analysis)
  if (n == 0) return;
  KOKKOS_ASSERT(data.extent(SliceDim) == n);
  KOKKOS_ASSERT(work.size() >= 2 * n);
  const std::size_t nlines = line_count<SliceDim>(data);
  // work.size() / 2 complex values, n per line (no 2 * n: it could wrap)
  const std::size_t max_concurrent = work.size() / 2 / n;
  if constexpr (LevelType::merges_lines) {
    // Rounds of at most max_concurrent lines; line j of a round uses the
    // scratch region starting at complex index j * n
    if (max_concurrent == 0) return;
    const bool line_fastest = lines_are_inner<SliceDim>(data);
    const auto map =
        make_line_pair_map<SliceDim>(data, data, first_dim_is_inner(data));
    const auto scratch = make_flat_pair_scratch(work);
    for (std::size_t base = 0; base < nlines; base += max_concurrent) {
      const std::size_t count =
          max_concurrent < nlines - base ? max_concurrent : nlines - base;
      // Line j of the round: (data line base + j, scratch region j)
      const auto lines_of = [&](std::size_t j) {
        auto region   = scratch;
        region.m_base = j * n;
        return Kokkos::make_pair(
            line_from_offset<SliceDim>(data, map.offsets(base + j).first),
            region);
      };
      c2c_lines<Dir>(level, node.kernel_data(), count, line_fastest, lines_of);
      if (node.nstages() % 2 == 1) {
        for_each_in_lines(level, count, n, line_fastest,
                          [&](std::size_t j, std::size_t i) {
                            const auto lines = lines_of(j);
                            lines.first.store(i, lines.second.load(i));
                          });
        level.barrier();
      }
    }
  } else {
    level.for_lines(nlines, max_concurrent,
                    [&](const auto &lvl, std::size_t l, std::size_t slot) {
                      const auto x = line_at<SliceDim>(data, l);
                      const auto scratch =
                          make_flat_pair_scratch(work, slot * n);
                      c2c_line<Dir>(lvl, node.kernel_data(), x, scratch);
                      if (node.nstages() % 2 == 1)
                        copy_line(lvl, n, scratch, x);
                    });
  }
}

/// \brief C2C along slice dimension SliceDim. Where the work buffer comes
/// from depends on the kind of the root plan (development-plan.md §2.6):
/// - C2C root: each line ping-pongs with the same positions in the other
///   buffer; all lines share the stage count, so the buffer holding the data
///   flips once per pass (odd stage count). Lines never share memory, so any
///   number of them may run concurrently.
/// - R2C root (heads after the real pass): the data is in `out`; the whole
///   real `in` is free and serves as flat complex scratch. The result is
///   copied back to `out` if the stage count is odd.
/// - C2R root (heads before the real pass): the data is in `in`; the whole
///   real `out` is the scratch; the result is copied back to `in`.
template <int SliceDim, KokkosFFT::Direction Dir, TransformKind RootKind,
          typename LevelType, typename PlanType, typename InViewType,
          typename OutViewType>
KOKKOS_FUNCTION void pass_c2c(const LevelType &level, const PlanType &node,
                              const InViewType &in, const OutViewType &out,
                              BufferState &state) {
  if constexpr (RootKind == TransformKind::C2C) {
    KOKKOS_ASSERT(in.extent(SliceDim) == node.length(0));
    KOKKOS_ASSERT(out.extent(SliceDim) == node.length(0));
    const std::size_t nlines = line_count<SliceDim>(in);
    const bool data_in_out   = state.data_in_out;
    if constexpr (merged_pass_v<LevelType, InViewType>) {
      const bool line_fastest = lines_are_inner<SliceDim>(in);
      const auto map =
          make_line_pair_map<SliceDim>(in, out, first_dim_is_inner(in));
      // Line l of in and of out, found with one decode
      const auto xy_of = [&](std::size_t l) {
        const auto offsets = map.offsets(l);
        return Kokkos::make_pair(
            line_from_offset<SliceDim>(in, offsets.first),
            line_from_offset<SliceDim>(out, offsets.second));
      };
      if (data_in_out) {
        c2c_lines<Dir>(level, node.kernel_data(), nlines, line_fastest,
                       [&](std::size_t l) {
                         const auto xy = xy_of(l);
                         return Kokkos::make_pair(xy.second, xy.first);
                       });
      } else {
        c2c_lines<Dir>(level, node.kernel_data(), nlines, line_fastest, xy_of);
      }
    } else {
      level.for_lines(nlines, nlines,
                      [&](const auto &lvl, std::size_t l, std::size_t) {
                        const auto x = line_at<SliceDim>(in, l);
                        const auto y = line_at<SliceDim>(out, l);
                        if (data_in_out) {
                          c2c_line<Dir>(lvl, node.kernel_data(), y, x);
                        } else {
                          c2c_line<Dir>(lvl, node.kernel_data(), x, y);
                        }
                      });
    }
    if (node.nstages() % 2 == 1) state.data_in_out = !state.data_in_out;
  } else if constexpr (RootKind == TransformKind::R2C) {
    KOKKOS_ASSERT(state.data_in_out);
    c2c_lines_with_scratch<SliceDim, Dir>(level, node, out, in);
  } else {
    KOKKOS_ASSERT(!state.data_in_out);
    c2c_lines_with_scratch<SliceDim, Dir>(level, node, in, out);
  }
}

/// \brief R2C along slice dimension SliceDim (the last FFT axis), line by
/// line. Odd n: odd_r2c_line. Even n (see r2c_postprocess): the real `in`
/// line is read as h = n/2 complex pairs; the length-h complex FFT
/// ping-pongs between it and the first h entries of the `out` line; the
/// post-processing writes X[0..h] to `out`. The result always ends in `out`.
/// Lines never share memory.
template <int SliceDim, typename LevelType, typename PlanType,
          typename InViewType, typename OutViewType>
KOKKOS_FUNCTION void pass_r2c(const LevelType &level, const PlanType &node,
                              const InViewType &in, const OutViewType &out,
                              BufferState &state) {
  KOKKOS_ASSERT(in.extent(SliceDim) == node.in_extent(0));
  KOKKOS_ASSERT(out.extent(SliceDim) == node.out_extent(0));
  KOKKOS_ASSERT(!state.data_in_out);

  const bool is_odd        = node.length(0) % 2 == 1;
  const std::size_t nlines = line_count<SliceDim>(in);
  if constexpr (merged_pass_v<LevelType, InViewType>) {
    if (!is_odd) {
      const bool line_fastest = lines_are_inner<SliceDim>(in);
      const auto map =
          make_line_pair_map<SliceDim>(in, out, first_dim_is_inner(in));
      // Line l of in (as complex pairs) and of out, found with one decode
      const auto xy_of = [&](std::size_t l) {
        const auto offsets = map.offsets(l);
        return Kokkos::make_pair(
            pair_line_from_offset<SliceDim>(in, offsets.first),
            line_from_offset<SliceDim>(out, offsets.second));
      };
      c2c_lines<KokkosFFT::Direction::forward>(level, node.kernel_data(),
                                               nlines, line_fastest, xy_of);
      const std::size_t h = node.n_fft();
      const auto &tw      = node.kernel_data().real_twiddles();
      const bool in_place = node.nstages() % 2 == 1;  // Z is already in `out`
      for_each_in_lines(level, nlines, h / 2 + 1, line_fastest,
                        [&](std::size_t l, std::size_t k) {
                          const auto xy = xy_of(l);
                          if (in_place) {
                            r2c_postprocess_item(h, tw, k, xy.second,
                                                 xy.second);
                          } else {
                            r2c_postprocess_item(h, tw, k, xy.first, xy.second);
                          }
                        });
      level.barrier();
      state.data_in_out = true;
      return;
    }
  }
  level.for_lines(
      nlines, nlines, [&](const auto &lvl, std::size_t l, std::size_t) {
        const auto y = line_at<SliceDim>(out, l);
        if (is_odd) {
          // Odd n: real half-complex DIT stages
          // (KokkosFFT_Batched_Kernels_Odd.hpp)
          odd_r2c_line(lvl, node.kernel_data(), line_at<SliceDim>(in, l), y);
          return;
        }
        const auto x = pair_line_at<SliceDim>(in, l);
        c2c_line<KokkosFFT::Direction::forward>(lvl, node.kernel_data(), x, y);
        if (node.nstages() % 2 == 1) {
          r2c_postprocess(lvl, node.kernel_data(), y, y);  // in place
        } else {
          r2c_postprocess(lvl, node.kernel_data(), x, y);
        }
        lvl.barrier();
      });
  state.data_in_out = true;
}

/// \brief C2R along slice dimension SliceDim (the last FFT axis), line by
/// line. Odd n: odd_c2r_line. Even n (see c2r_preprocess): the `in` line
/// (X[0..h]) is pre-processed in place into 2 Z; the length-h complex FFT
/// ping-pongs between it and the real `out` line read as h complex pairs.
/// The result always ends in `out` (copied if the stage count is even).
/// Lines never share memory.
template <int SliceDim, typename LevelType, typename PlanType,
          typename InViewType, typename OutViewType>
KOKKOS_FUNCTION void pass_c2r(const LevelType &level, const PlanType &node,
                              const InViewType &in, const OutViewType &out,
                              BufferState &state) {
  KOKKOS_ASSERT(in.extent(SliceDim) == node.in_extent(0));
  KOKKOS_ASSERT(out.extent(SliceDim) == node.out_extent(0));
  KOKKOS_ASSERT(!state.data_in_out);

  const bool is_odd        = node.length(0) % 2 == 1;
  const std::size_t nlines = line_count<SliceDim>(in);
  if constexpr (merged_pass_v<LevelType, InViewType>) {
    if (!is_odd) {
      const bool line_fastest = lines_are_inner<SliceDim>(in);
      const auto map =
          make_line_pair_map<SliceDim>(in, out, first_dim_is_inner(in));
      // Line l of in and of out (as complex pairs), found with one decode
      const auto xy_of = [&](std::size_t l) {
        const auto offsets = map.offsets(l);
        return Kokkos::make_pair(
            line_from_offset<SliceDim>(in, offsets.first),
            pair_line_from_offset<SliceDim>(out, offsets.second));
      };
      const std::size_t h = node.n_fft();
      const auto &tw      = node.kernel_data().real_twiddles();
      for_each_in_lines(level, nlines, h / 2 + 1, line_fastest,
                        [&](std::size_t l, std::size_t k) {
                          c2r_preprocess_item(h, tw, k, xy_of(l).first);
                        });
      level.barrier();
      c2c_lines<KokkosFFT::Direction::backward>(level, node.kernel_data(),
                                                nlines, line_fastest, xy_of);
      if (node.nstages() % 2 == 0) {
        for_each_in_lines(level, nlines, h, line_fastest,
                          [&](std::size_t l, std::size_t i) {
                            const auto xy = xy_of(l);
                            xy.second.store(i, xy.first.load(i));
                          });
        level.barrier();
      }
      state.data_in_out = true;
      return;
    }
  }
  level.for_lines(
      nlines, nlines, [&](const auto &lvl, std::size_t l, std::size_t) {
        const auto x = line_at<SliceDim>(in, l);
        if (is_odd) {
          // Odd n: real half-complex DIT stages
          // (KokkosFFT_Batched_Kernels_Odd.hpp)
          odd_c2r_line(lvl, node.kernel_data(), x, line_at<SliceDim>(out, l));
          return;
        }
        const auto y = pair_line_at<SliceDim>(out, l);
        c2r_preprocess(lvl, node.kernel_data(), x);
        lvl.barrier();
        c2c_line<KokkosFFT::Direction::backward>(lvl, node.kernel_data(), x, y);
        if (node.nstages() % 2 == 0) copy_line(lvl, node.n_fft(), x, y);
      });
  state.data_in_out = true;
}

/// \brief Walk the plan tree by compile-time recursion. Every node type is a
/// distinct instantiation, so there is no runtime recursion.
/// R2C: last (real) axis first, then heads (C2C on out)
/// C2R: heads (C2C on in) first, then last (real) axis
/// C2C: last axis first, then heads
///
/// \tparam RootAxes The FFT axes of the root plan, used to map the axis of
/// each leaf to its dimension in the slice.
/// \tparam RootKind The kind of the root plan, which decides where the C2C
/// passes of the heads find their work buffer.
template <typename RootAxes, TransformKind RootKind, KokkosFFT::Direction Dir,
          typename LevelType, typename PlanType, typename InViewType,
          typename OutViewType>
KOKKOS_FUNCTION void execute_node(const LevelType &level, const PlanType &node,
                                  const InViewType &in, const OutViewType &out,
                                  BufferState &state) {
  if constexpr (PlanType::rank == 1) {
    constexpr int dim = slice_dim_v<RootAxes, PlanType::axis>;
    static_assert(dim >= 0,
                  "KokkosFFT::Batched::execute: axis not found in root axes");
    if constexpr (PlanType::kind == TransformKind::R2C) {
      pass_r2c<dim>(level, node, in, out, state);
    } else if constexpr (PlanType::kind == TransformKind::C2R) {
      pass_c2r<dim>(level, node, in, out, state);
    } else {
      pass_c2c<dim, Dir, RootKind>(level, node, in, out, state);
    }
    level.barrier();
  } else if constexpr (PlanType::kind == TransformKind::C2R) {
    execute_node<RootAxes, RootKind, Dir>(level, node.heads_plan(), in, out,
                                          state);
    execute_node<RootAxes, RootKind, Dir>(level, node.last_plan(), in, out,
                                          state);
  } else {
    execute_node<RootAxes, RootKind, Dir>(level, node.last_plan(), in, out,
                                          state);
    execute_node<RootAxes, RootKind, Dir>(level, node.heads_plan(), in, out,
                                          state);
  }
}

/// \brief Scale the result and make sure it ends in `out`: copy + scale if
/// it is still in `in` (C2C only), otherwise scale in place (skipped if the
/// factor is 1). R2C/C2R passes always leave the result in `out`.
template <KokkosFFT::Direction Dir, typename LevelType, typename PlanType,
          typename InViewType, typename OutViewType>
KOKKOS_FUNCTION void finalize(const LevelType &level, const PlanType &plan,
                              const InViewType &in, const OutViewType &out,
                              const BufferState &state,
                              KokkosFFT::Normalization norm) {
  using float_type = typename PlanType::float_type;
  const float_type coef =
      normalization_factor<float_type, Dir>(norm, plan.fft_size());
  auto *y                = out.data();
  const std::size_t size = out.size();
  if (state.data_in_out) {
    if (coef != float_type(1)) {
      level.for_each(size, [&](std::size_t i) {
        const std::size_t iy = element_offset(out, i);
        y[iy]                = y[iy] * coef;
      });
    }
  } else {
    if constexpr (PlanType::kind == TransformKind::C2C) {
      const auto *x = in.data();
      level.for_each(size, [&](std::size_t i) {
        y[element_offset(out, i)] = x[element_offset(in, i)] * coef;
      });
    } else {
      KOKKOS_ASSERT(false && "R2C/C2R passes leave the result in out");
    }
  }
  level.barrier();
}

template <KokkosFFT::Direction Dir, typename LevelType, typename PlanType,
          typename InViewType, typename OutViewType>
KOKKOS_FUNCTION void run(const LevelType &level, const PlanType &plan,
                         const InViewType &in, const OutViewType &out,
                         KokkosFFT::Normalization norm) {
  BufferState state;
  execute_node<typename PlanType::axes_type, PlanType::kind, Dir>(
      level, plan, in, out, state);
  finalize<Dir>(level, plan, in, out, state, norm);
}

template <typename LevelType, typename PlanType, typename InViewType,
          typename OutViewType>
KOKKOS_FUNCTION void run(const LevelType &level, const PlanType &plan,
                         const InViewType &in, const OutViewType &out,
                         KokkosFFT::Direction dir,
                         KokkosFFT::Normalization norm) {
  if (dir == KokkosFFT::Direction::forward) {
    run<KokkosFFT::Direction::forward>(level, plan, in, out, norm);
  } else {
    run<KokkosFFT::Direction::backward>(level, plan, in, out, norm);
  }
}

/// \brief Direction implied by the kind of a real transform
template <TransformKind Kind>
inline constexpr KokkosFFT::Direction real_direction_v =
    Kind == TransformKind::R2C ? KokkosFFT::Direction::forward
                               : KokkosFFT::Direction::backward;

}  // namespace Impl

// ---- Serial plans: executed entirely by the calling thread ----

/// \brief C2C transform of one slice with a serial plan.
/// The same plan can be used for both directions.
/// The contents of `in` are unspecified after the call.
template <Planable PlanType, InSliceView<PlanType> InViewType,
          OutSliceView<PlanType> OutViewType>
  requires(!PlanType::is_team && PlanType::kind == TransformKind::C2C)
KOKKOS_FUNCTION void execute(
    const PlanType &plan, const InViewType &in, const OutViewType &out,
    KokkosFFT::Direction dir,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  Impl::run(Impl::SerialLevel{}, plan, in, out, dir, norm);
}

/// \brief R2C (forward) or C2R (backward) transform of one slice with a
/// serial plan. The contents of `in` are unspecified after the call.
template <Planable PlanType, InSliceView<PlanType> InViewType,
          OutSliceView<PlanType> OutViewType>
  requires(!PlanType::is_team && PlanType::kind != TransformKind::C2C)
KOKKOS_FUNCTION void execute(
    const PlanType &plan, const InViewType &in, const OutViewType &out,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  Impl::run<Impl::real_direction_v<PlanType::kind>>(Impl::SerialLevel{}, plan,
                                                    in, out, norm);
}

// ---- Team plans: all threads of `member` cooperate on one slice ----

/// \brief C2C transform of one slice with a team plan.
/// Must be called by all threads of the team with the same arguments.
template <typename MemberType, Planable PlanType,
          InSliceView<PlanType> InViewType, OutSliceView<PlanType> OutViewType>
  requires(Kokkos::is_team_handle_v<MemberType> && PlanType::is_team &&
           PlanType::kind == TransformKind::C2C)
KOKKOS_FUNCTION void execute(
    const MemberType &member, const PlanType &plan, const InViewType &in,
    const OutViewType &out, KokkosFFT::Direction dir,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  Impl::run(Impl::TeamLevel<MemberType>(member), plan, in, out, dir, norm);
}

/// \brief R2C (forward) or C2R (backward) transform of one slice with a team
/// plan. Must be called by all threads of the team with the same arguments.
template <typename MemberType, Planable PlanType,
          InSliceView<PlanType> InViewType, OutSliceView<PlanType> OutViewType>
  requires(Kokkos::is_team_handle_v<MemberType> && PlanType::is_team &&
           PlanType::kind != TransformKind::C2C)
KOKKOS_FUNCTION void execute(
    const MemberType &member, const PlanType &plan, const InViewType &in,
    const OutViewType &out,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  Impl::run<Impl::real_direction_v<PlanType::kind>>(
      Impl::TeamLevel<MemberType>(member), plan, in, out, norm);
}

// ---- Level mismatches ----

/// \brief Error: a team plan (built from a TeamPolicy) must be executed with
/// the team member: execute(member, plan, in, out, ...)
template <Planable PlanType, typename InViewType, typename OutViewType,
          typename... Args>
  requires(PlanType::is_team)
void execute(const PlanType &, const InViewType &, const OutViewType &,
             Args...) = delete;

/// \brief Error: a serial plan (built from an execution space) must be
/// executed without the team member: execute(plan, in, out, ...)
template <typename MemberType, Planable PlanType, typename InViewType,
          typename OutViewType, typename... Args>
  requires(Kokkos::is_team_handle_v<MemberType> && !PlanType::is_team)
void execute(const MemberType &, const PlanType &, const InViewType &,
             const OutViewType &, Args...) = delete;

}  // namespace Batched
}  // namespace KokkosFFT

#endif
