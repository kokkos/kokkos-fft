#ifndef KOKKOSFFT_BATCHED_AXISDATA_HPP
#define KOKKOSFFT_BATCHED_AXISDATA_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

/// \brief Everything the 1D kernels need to transform one axis: lengths,
/// stage descriptors and twiddle tables (development-plan.md §2.3, §3.6).
///
/// Kernels are templated on this type instead of the leaf Plan type, so all
/// axes of all plans with the same complex type and memory space share one set
/// of kernel instantiations (§9, R2). The leaf Plan owns one AxisData and
/// exposes it through kernel_data().
///
/// Trivially cheap to copy: scalars and View handles.
template <typename ComplexType, typename MemorySpace> struct AxisData {
  using complex_type = ComplexType;
  using float_type = typename ComplexType::value_type;
  using memory_space = MemorySpace;
  using stage_view_type = Kokkos::View<std::size_t *, memory_space>;
  using twiddle_view_type = Kokkos::View<complex_type *, memory_space>;

  //! Logical length along this axis
  std::size_t m_n = 1;
  //! Length of the transform run by the stages (n, n/2 for even-n real, n for
  //! odd-n real)
  std::size_t m_n_fft = 1;
  //! Number of stages (== m_radix.extent(0))
  int m_nstages = 0;
  //! Radix of each stage
  stage_view_type m_radix;
  //! Start of each stage in m_twiddles
  stage_view_type m_tw_offset;
  //! Start of each generic-radix stage in m_roots (0 for other stages)
  stage_view_type m_root_offset;
  //! Stage twiddles (forward), n_fft - 1 entries
  twiddle_view_type m_twiddles;
  //! Roots of unity of the generic-radix stages (forward)
  twiddle_view_type m_roots;
  //! R2C/C2R twiddles (forward): n/4 + 1 entries for even n, n for odd n
  twiddle_view_type m_real_twiddles;

  /// \brief Logical length of the axis (the index is for interface
  /// compatibility with Plan::length and must be 0)
  KOKKOS_FUNCTION std::size_t length([[maybe_unused]] std::size_t i) const {
    KOKKOS_ASSERT(i == 0);
    return m_n;
  }
  KOKKOS_FUNCTION std::size_t n_fft() const { return m_n_fft; }
  KOKKOS_FUNCTION int nstages() const { return m_nstages; }

  // The accessors below read Views in memory_space: use them in kernels only.
  KOKKOS_FUNCTION std::size_t radix(int s) const { return m_radix(s); }
  KOKKOS_FUNCTION std::size_t tw_offset(int s) const { return m_tw_offset(s); }
  KOKKOS_FUNCTION std::size_t root_offset(int s) const {
    return m_root_offset(s);
  }
  KOKKOS_FUNCTION const stage_view_type &radices() const { return m_radix; }
  KOKKOS_FUNCTION const twiddle_view_type &twiddles() const {
    return m_twiddles;
  }
  KOKKOS_FUNCTION const twiddle_view_type &roots() const { return m_roots; }
  KOKKOS_FUNCTION const twiddle_view_type &real_twiddles() const {
    return m_real_twiddles;
  }
};

} // namespace Impl
} // namespace Batched
} // namespace KokkosFFT

#endif
