// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HELPERS_HPP
#define KOKKOSFFT_HELPERS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {

/// \brief Compute shift amounts for FFT shift operations.
/// Computes the shift for each axis so that the zero-frequency component is
/// moved to the center of the spectrum (or back to its original position for an
/// inverse shift). The provided axes (which can be negative) are first
/// converted to non-negative axes. The function then calculates the shift as
/// half of the extent of each selected axis, adjusted by the given direction (1
/// for forward FFT shift, -1 for inverse FFT shift).
///
/// \tparam ViewType The type of the Kokkos View.
/// \tparam DIM The number of axes to shift.
///
/// \param x[in,out] The input/output Kokkos View whose extents are used to
/// determine the shift.
/// \param axes[in] A container of axes indices (negative values allowed; they
/// are converted).
/// \param direction[in] The direction of the shift: 1 for fftshift and -1 for
/// ifftshift.
///
/// \return A Kokkos::Array with the computed shift amounts for each axis.
///
/// \throws If any duplicate axes are found ("Axes overlap") or if any axis
/// index is out of the valid range.
///
template <typename ViewType, std::size_t DIM = 1>
auto get_shifts(const ViewType& x, axis_type<DIM> axes, int direction = 1) {
  // Convert the input axes to be in the range of [0, rank-1]
  std::vector<int> non_negative_axes;
  for (std::size_t i = 0; i < DIM; i++) {
    int axis =
        KokkosFFT::Impl::convert_negative_axis(ViewType::rank(), axes.at(i));
    non_negative_axes.push_back(axis);
  }

  // Assert if the elements are overlapped
  constexpr int rank = ViewType::rank();
  KOKKOSFFT_THROW_IF(KokkosFFT::Impl::has_duplicate_values(non_negative_axes),
                     "Axes overlap");
  KOKKOSFFT_THROW_IF(
      KokkosFFT::Impl::is_out_of_range_value_included(non_negative_axes, rank),
      "Axes include an out-of-range index."
      "Axes must be in the range of [-rank, rank-1].");

  using ArrayType  = Kokkos::Array<std::size_t, rank>;
  ArrayType shifts = {};
  for (int i = 0; i < static_cast<int>(DIM); i++) {
    int axis   = non_negative_axes.at(i);
    int extent = x.extent(axis);
    int shift  = extent / 2 * direction;
    shift      = shift % extent;
    if (shift < 0) shift += extent;
    shifts[axis] = static_cast<std::size_t>(shift);
  }
  return shifts;
}

/// \brief Roll functor for in-place FFT shift operations.
/// This struct implements a functor that applies a circular shift
/// (roll) on a Kokkos view. It shifts the data so that the zero-frequency
/// component is moved to the center of the spectrum. The shift amount is
/// specified by m_shifts which is calculated using the get_shifts function. The
/// functor creates a temporary view to store the shifted data and then the
/// shifted values are copied back into the original view.
///
/// \tparam ExecutionSpace The type of Kokkos execution space.
/// \tparam ViewType The type of the Kokkos View.
/// \tparam iType The index type used for the view.
template <typename ExecutionSpace, typename ViewType, typename iType>
struct Roll {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(ViewType::rank(), std::size_t(6));

  using ArrayType          = Kokkos::Array<std::size_t, ViewType::rank()>;
  using LayoutType         = typename ViewType::array_layout;
  using ManageableViewType = typename manageable_view_type<ViewType>::type;

  ManageableViewType m_tmp;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] x The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const ViewType& x) const {
    if constexpr (ViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, x.extent(0));
    } else {
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, Kokkos::Iterate::Default,
                       Kokkos::Iterate::Default>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins = {};
      Kokkos::Array<std::size_t, m_rank_truncated> ends   = {};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = x.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructor for the Roll functor.
  ///
  /// \param x[in,out] The input/output Kokkos view to be shifted.
  /// \param shifts[in] The shift amounts for each axis.
  /// \param exec_space[in] The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Roll(const ViewType& x, const ArrayType& shifts,
       const ExecutionSpace exec_space = ExecutionSpace())
      : m_tmp("tmp", create_layout<LayoutType>(extract_extents(x))) {
    Kokkos::parallel_for("KokkosFFT::roll", get_policy(exec_space, x),
                         RollInternal(x, m_tmp, shifts));
    Kokkos::deep_copy(exec_space, x, m_tmp);
  }

  /// \brief Helper functor to perform the roll operation
  struct RollInternal {
    ViewType m_x;
    ManageableViewType m_buffer;
    ArrayType m_shifts;

    RollInternal(const ViewType& x, const ManageableViewType& buffer,
                 const ArrayType& shifts)
        : m_x(x), m_buffer(buffer), m_shifts(shifts) {}

    template <typename... IndicesType>
    KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
      auto get_dst = [&](iType idx_src, std::size_t axis) {
        return (idx_src + iType(m_shifts[axis])) % iType(m_x.extent(axis));
      };

      if constexpr (ViewType::rank() <= 6) {
        iType src_idx[ViewType::rank()] = {static_cast<iType>(indices)...};
        iType dst_idx[ViewType::rank()] = {};
        for (std::size_t i = 0; i < ViewType::rank(); ++i) {
          dst_idx[i] = get_dst(src_idx[i], i);
        }
        roll_internal(dst_idx, src_idx,
                      std::make_index_sequence<ViewType::rank()>{});
      } else if constexpr (ViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_x.extent(6)); i6++) {
          iType src_idx[ViewType::rank()] = {static_cast<iType>(indices)...,
                                             i6};
          iType dst_idx[ViewType::rank()] = {};

          for (std::size_t i = 0; i < ViewType::rank(); ++i) {
            dst_idx[i] = get_dst(src_idx[i], i);
          }
          roll_internal(dst_idx, src_idx,
                        std::make_index_sequence<ViewType::rank()>{});
        }
      } else if constexpr (ViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_x.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_x.extent(7)); i7++) {
            iType src_idx[ViewType::rank()] = {static_cast<iType>(indices)...,
                                               i6, i7};
            iType dst_idx[ViewType::rank()] = {};

            for (std::size_t i = 0; i < ViewType::rank(); ++i) {
              dst_idx[i] = get_dst(src_idx[i], i);
            }
            roll_internal(dst_idx, src_idx,
                          std::make_index_sequence<ViewType::rank()>{});
          }
        }
      }
    }

    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION void roll_internal(
        iType dst_idx[], iType src_idx[], std::index_sequence<Is...>) const {
      m_buffer(dst_idx[Is]...) = m_x(src_idx[Is]...);
    }
  };
};

/// \brief Implementation of FFT shift operations.
/// Computes the necessary shift amounts for each axis using the get_shifts
/// function and applies an in-place FFT shift on the input view `x`. Depending
/// on the memory span of `x`, it selects an appropriate index type (int64_t if
/// the span exceeds the maximum value representable by int, otherwise int).
///
/// \tparam ExecutionSpace The type of Kokkos execution space.
/// \tparam ViewType The type of the Kokkos View.
/// \tparam DIM The number of axes to shift.
///
/// \param exec_space [in] Kokkos execution space.
/// \param x [in,out] The input/output Kokkos View.
/// \param axes [in] A container of axes indices (negative values allowed; they
/// are converted).
/// \param direction [in] The direction of the shift: 1 for fftshift and -1 for
/// ifftshift.
///
/// \throws if any duplicate axes are found ("Axes overlap") or if any axis
/// index is out of the valid range.
template <typename ExecutionSpace, typename ViewType, std::size_t DIM = 1>
void fftshift_impl(const ExecutionSpace& exec_space, const ViewType& x,
                   axis_type<DIM> axes, int direction) {
  auto shifts = get_shifts(x, axes, direction);
  if (x.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Roll<ExecutionSpace, ViewType, int64_t>(x, shifts, exec_space);
  } else {
    Roll<ExecutionSpace, ViewType, int>(x, shifts, exec_space);
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

namespace KokkosFFT {
/// \brief Return the DFT sample frequencies
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam RealType: The floating point precision type to represent frequencies
///
/// \param exec_space [in] Kokkos execution space
/// \param n [in] Window length
/// \param d [in] Sample spacing (default, 1)
///
/// \return Sampling frequency
template <typename ExecutionSpace, typename RealType>
auto fftfreq(const ExecutionSpace&, const std::size_t n,
             const RealType d = 1.0) {
  static_assert(KokkosFFT::Impl::is_real_v<RealType>,
                "fftfreq: d must be float or double");
  using ViewType = Kokkos::View<RealType*, ExecutionSpace>;
  ViewType freq("freq", n);

  RealType val = 1.0 / (static_cast<RealType>(n) * d);
  int N1       = (n - 1) / 2 + 1;
  int N2       = n / 2;

  auto h_freq = Kokkos::create_mirror_view(freq);

  auto p1 = KokkosFFT::Impl::arange(0, N1);
  auto p2 = KokkosFFT::Impl::arange(-N2, 0);

  for (int i = 0; i < N1; i++) {
    h_freq(i) = static_cast<RealType>(p1.at(i)) * val;
  }
  for (int i = 0; i < N2; i++) {
    h_freq(i + N1) = static_cast<RealType>(p2.at(i)) * val;
  }
  Kokkos::deep_copy(freq, h_freq);

  return freq;
}

/// \brief Return the DFT sample frequencies for Real FFTs
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam RealType: The floating point precision type to represent frequencies
///
/// \param exec_space [in] Kokkos execution space
/// \param n [in] Window length
/// \param d [in] Sample spacing (default, 1)
///
/// \return Sampling frequency starting from zero
template <typename ExecutionSpace, typename RealType>
auto rfftfreq(const ExecutionSpace&, const std::size_t n,
              const RealType d = 1.0) {
  static_assert(KokkosFFT::Impl::is_real_v<RealType>,
                "fftfreq: d must be float or double");
  using ViewType = Kokkos::View<RealType*, ExecutionSpace>;

  RealType val = 1.0 / (static_cast<RealType>(n) * d);
  int N        = n / 2 + 1;
  ViewType freq("freq", N);

  auto h_freq = Kokkos::create_mirror_view(freq);
  auto p      = KokkosFFT::Impl::arange(0, N);

  for (int i = 0; i < N; i++) {
    h_freq(i) = static_cast<RealType>(p.at(i)) * val;
  }
  Kokkos::deep_copy(freq, h_freq);

  return freq;
}

/// \brief Shift the zero-frequency component to the center of the spectrum
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam ViewType: Input/Output View type for the shift
///
/// \param exec_space [in] Kokkos execution space
/// \param inout [in,out] Spectrum
/// \param axes [in] Axes over which to shift (default: nullopt, shifting over
/// all axes)
template <typename ExecutionSpace, typename ViewType>
void fftshift(const ExecutionSpace& exec_space, const ViewType& inout,
              std::optional<int> axes = std::nullopt) {
  static_assert(KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, ViewType>,
                "fftshift: View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "ExecutionSpace must be able to access data in ViewType");
  static_assert(ViewType::rank() >= 1,
                "fftshift: View rank must be larger than or equal to 1");

  if (axes) {
    axis_type<1> tmp_axes{axes.value()};
    KokkosFFT::Impl::fftshift_impl(exec_space, inout, tmp_axes, 1);
  } else {
    constexpr std::size_t rank = ViewType::rank();
    constexpr int start        = -static_cast<int>(rank);
    auto tmp_axes = KokkosFFT::Impl::index_sequence<int, rank, start>();
    KokkosFFT::Impl::fftshift_impl(exec_space, inout, tmp_axes, 1);
  }
}

/// \brief Shift the zero-frequency component to the center of the spectrum
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam ViewType: Input/Output View type for the shift
/// \tparam DIM: The dimensionality of the shift
///
/// \param exec_space [in] Kokkos execution space
/// \param inout [in,out] Spectrum
/// \param axes [in] Axes over which to shift
template <typename ExecutionSpace, typename ViewType, std::size_t DIM = 1>
void fftshift(const ExecutionSpace& exec_space, const ViewType& inout,
              axis_type<DIM> axes) {
  static_assert(KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, ViewType>,
                "fftshift: View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "ExecutionSpace must be able to access data in ViewType");
  static_assert(ViewType::rank() >= DIM,
                "fftshift: View rank must be larger than or equal to the Rank "
                "of FFT axes");
  KokkosFFT::Impl::fftshift_impl(exec_space, inout, axes, 1);
}

/// \brief The inverse of fftshift
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam ViewType: Input/Output View type for the shift
///
/// \param exec_space [in] Kokkos execution space
/// \param inout [in,out] Spectrum
/// \param axes [in] Axes over which to shift (default: nullopt, shifting over
/// all axes)
template <typename ExecutionSpace, typename ViewType>
void ifftshift(const ExecutionSpace& exec_space, const ViewType& inout,
               std::optional<int> axes = std::nullopt) {
  static_assert(KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, ViewType>,
                "ifftshift: View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "ExecutionSpace must be able to access data in ViewType");
  static_assert(ViewType::rank() >= 1,
                "ifftshift: View rank must be larger than or equal to 1");
  if (axes) {
    axis_type<1> tmp_axes{axes.value()};
    KokkosFFT::Impl::fftshift_impl(exec_space, inout, tmp_axes, -1);
  } else {
    constexpr std::size_t rank = ViewType::rank();
    constexpr int start        = -static_cast<int>(rank);
    auto tmp_axes = KokkosFFT::Impl::index_sequence<int, rank, start>();
    KokkosFFT::Impl::fftshift_impl(exec_space, inout, tmp_axes, -1);
  }
}

/// \brief The inverse of fftshift
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam ViewType: Input/Output View type for the shift
/// \tparam DIM: The dimensionality of the shift
///
/// \param exec_space [in] Kokkos execution space
/// \param inout [in,out] Spectrum
/// \param axes [in] Axes over which to shift
template <typename ExecutionSpace, typename ViewType, std::size_t DIM = 1>
void ifftshift(const ExecutionSpace& exec_space, const ViewType& inout,
               axis_type<DIM> axes) {
  static_assert(KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, ViewType>,
                "ifftshift: View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "ExecutionSpace must be able to access data in ViewType");
  static_assert(ViewType::rank() >= DIM,
                "ifftshift: View rank must be larger than or equal to the Rank "
                "of FFT axes");
  KokkosFFT::Impl::fftshift_impl(exec_space, inout, axes, -1);
}
}  // namespace KokkosFFT

#endif
