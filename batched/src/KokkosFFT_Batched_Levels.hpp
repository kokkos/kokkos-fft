#ifndef KOKKOSFFT_BATCHED_LEVELS_HPP
#define KOKKOSFFT_BATCHED_LEVELS_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstdint>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

// A level tells the kernels how the work of one slice is distributed:
//   for_each(n, f)  : call f(i) for i in [0, n), each index exactly once
//   barrier()       : make the writes of for_each visible to the whole level
//   for_lines(nlines, max_concurrent, f):
//                     call f(level, l, slot) for each line l of a pass. `level`
//                     is the level the line work must use, and `slot` in
//                     [0, max_concurrent) identifies a scratch region that no
//                     concurrent line shares.

// A level also says how a pass over many lines is best run (merges_lines):
//   false: line by line, all the stages of one line before the next line
//          (one thread: the line stays in cache)
//   true : stage by stage, the butterflies of all the lines of a stage in one
//          for_each (a team: every thread has work whatever the number of
//          lines, and neighbouring threads access neighbouring memory)

/// \brief Serial level: the calling thread does all the work
struct SerialLevel {
  static constexpr bool merges_lines = false;

  template <typename FunctorType>
  KOKKOS_INLINE_FUNCTION void for_each(std::size_t n,
                                       const FunctorType &f) const {
    for (std::size_t i = 0; i < n; ++i) {
      f(i);
    }
  }

  KOKKOS_INLINE_FUNCTION void barrier() const {}

  template <typename FunctorType>
  KOKKOS_INLINE_FUNCTION void for_lines(std::size_t nlines, std::size_t,
                                        const FunctorType &f) const {
    for (std::size_t l = 0; l < nlines; ++l) {
      f(*this, l, std::size_t(0));
    }
  }
};

/// \brief Team level: all threads (and vector lanes) of a team cooperate
template <typename MemberType>
struct TeamLevel {
  static constexpr bool merges_lines = true;

  const MemberType &m_member;

  KOKKOS_INLINE_FUNCTION explicit TeamLevel(const MemberType &member)
      : m_member(member) {}

  /// Each index runs on exactly one (thread, vector lane): TeamThreadRange
  /// alone would run it redundantly on every vector lane, which breaks the
  /// in-place passes (a lane could read a value another lane already wrote).
  template <typename FunctorType>
  KOKKOS_INLINE_FUNCTION void for_each(std::size_t n,
                                       const FunctorType &f) const {
    Kokkos::parallel_for(Kokkos::TeamVectorRange(m_member, n), f);
  }

  KOKKOS_INLINE_FUNCTION void barrier() const { m_member.team_barrier(); }

  /// Lines are distributed over the threads of the team when there are at
  /// least as many lines as threads: each line is then transformed serially
  /// by one vector lane of one thread, in rounds of at most `max_concurrent`
  /// lines (the number of disjoint scratch regions). Otherwise (e.g. 1-D, a
  /// single line) the lines are processed one after another and the whole
  /// team cooperates on each of them.
  template <typename FunctorType>
  KOKKOS_INLINE_FUNCTION void for_lines(std::size_t nlines,
                                        std::size_t max_concurrent,
                                        const FunctorType &f) const {
    const std::size_t team_size = m_member.team_size();
    if (nlines < team_size || max_concurrent < 2) {
      for (std::size_t l = 0; l < nlines; ++l) {
        f(*this, l, std::size_t(0));
      }
      return;
    }
    const std::size_t per_round =
        max_concurrent < nlines ? max_concurrent : nlines;
    for (std::size_t base = 0; base < nlines; base += per_round) {
      const std::size_t count =
          per_round < nlines - base ? per_round : nlines - base;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(m_member, count), [&](std::size_t j) {
            Kokkos::single(Kokkos::PerThread(m_member),
                           [&]() { f(SerialLevel{}, base + j, j); });
          });
      m_member.team_barrier();
    }
  }
};

/// \brief a / b in 32-bit arithmetic when both operands fit, otherwise in
/// 64-bit arithmetic
KOKKOS_FORCEINLINE_FUNCTION std::size_t div_32bit_if_fits(std::size_t a,
                                                          std::size_t b) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    if (((a | b) >> 32) == 0) {
      return static_cast<std::uint32_t>(a) / static_cast<std::uint32_t>(b);
    }
  }
  return a / b;
}

/// \brief a / b for unsigned sizes, as cheap as the target allows. The
/// merged-line passes divide in every work item, and the operands are almost
/// always small (indices within one slice).
/// - Host: 32-bit division when the operands fit, which is cheaper than a
///   64-bit division on x86-64. It is part of G1b, which made one-thread
///   teams 6-65% faster on OpenMP (development-plan.md, G1b).
/// - Device: a plain division (G1d). With the explicit check, the 1-D team
///   plans were 5-20% slower on the A100, with more registers and
///   instructions (optimization/measurements/2026-10-02_3f254eb_A100). The
///   device compiler is expected to emit 64-bit division with a fast path
///   for small operands already.
KOKKOS_FORCEINLINE_FUNCTION std::size_t fast_div(std::size_t a, std::size_t b) {
  KOKKOS_IF_ON_DEVICE((return a / b;))
  KOKKOS_IF_ON_HOST((return div_32bit_if_fits(a, b);))
}

/// \brief Call f(l, i) for every line l in [0, nlines) and item i in
/// [0, per_line), each pair exactly once, in one for_each of the level.
/// `line_fastest` chooses which index varies fastest between consecutive
/// work items, i.e. between neighbouring threads of a team: the line index
/// when neighbouring lines are closer in memory than neighbouring items.
/// One (cheap, see fast_div) division per work item.
template <typename LevelType, typename FunctorType>
KOKKOS_INLINE_FUNCTION void for_each_in_lines(const LevelType &level,
                                              std::size_t nlines,
                                              std::size_t per_line,
                                              bool line_fastest,
                                              const FunctorType &f) {
  if (nlines == 0 || per_line == 0) return;
  level.for_each(nlines * per_line, [&](std::size_t idx) {
    if (line_fastest) {
      const std::size_t i = fast_div(idx, nlines);
      f(idx - i * nlines, i);
    } else {
      const std::size_t l = fast_div(idx, per_line);
      f(l, idx - l * per_line);
    }
  });
}

}  // namespace Impl
}  // namespace Batched
}  // namespace KokkosFFT

#endif
