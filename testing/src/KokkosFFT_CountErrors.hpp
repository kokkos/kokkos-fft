// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_COUNT_ERRORS_HPP
#define KOKKOSFFT_COUNT_ERRORS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_Concepts.hpp"

namespace KokkosFFT {
namespace Testing {
namespace Impl {

template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, KokkosLayout Layout, std::size_t Rank,
          typename iType>
struct ViewErrors;

template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, KokkosLayout Layout, std::size_t Rank,
          typename iType>
struct FindErrors;

template <KokkosView ViewType>
Kokkos::Iterate get_iteration_order(const ViewType& view) {
  int64_t strides[ViewType::rank + 1];
  view.stride(strides);
  Kokkos::Iterate iterate;
  if (std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>) {
    iterate = Kokkos::Iterate::Right;
  } else if (std::is_same_v<typename ViewType::array_layout,
                            Kokkos::LayoutLeft>) {
    iterate = Kokkos::Iterate::Left;
  } else if (std::is_same_v<typename ViewType::array_layout,
                            Kokkos::LayoutStride>) {
    if (strides[0] > strides[ViewType::rank - 1])
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  } else {
    if (std::is_same_v<typename ViewType::execution_space::array_layout,
                       Kokkos::LayoutRight>)
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  }
  return iterate;
}

/// \brief Computes the number of error mismatches between two 1D Kokkos views.
/// This structure performs an element-by-element comparison of two 1D views. It
/// counts the number of elements where the difference exceeds a specified
/// tolerance defined by an absolute tolerance and a relative tolerance.
///
/// \tparam ExecutionSpace The Kokkos execution space to run the
/// parallel_reduce. \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \tparam Layout The layout type of the Kokkos views.
/// \tparam iType The integer type used for indexing the view elements.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, KokkosLayout Layout, typename iType>
struct ViewErrors<ExecutionSpace, AViewType, BViewType, Layout, 1, iType> {
  AViewType m_a;
  BViewType m_b;

  double m_rtol;
  double m_atol;

  std::size_t m_error = 0;

  using policy_type =
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;

  ///
  /// \brief Constructs the error counter and performs the error computation.
  ///
  /// \param a [in] First Kokkos view containing data to compare.
  /// \param b [in] Second Kokkos view containing data to compare against.
  /// \param rtol [in] Relative tolerance for comparing the view elements
  /// (default 1.e-5).
  /// \param atol [in] Absolute tolerance for comparing the view elements
  /// (default 1.e-8).
  /// \param space [in] The Kokkos execution space used to launch the parallel
  /// reduction.
  ViewErrors(const AViewType& a, const BViewType& b, double rtol = 1.e-5,
             double atol = 1.e-8, const ExecutionSpace space = ExecutionSpace())
      : m_a(a), m_b(b), m_rtol(rtol), m_atol(atol) {
    Kokkos::parallel_reduce(
        "ViewErrors-1D", policy_type(space, 0, m_a.extent(0)), *this, m_error);
  }

  /// \brief Operator called by Kokkos to perform the comparison of each
  /// element.
  ///
  /// \param i0 [in] The index of the element in the views.
  /// \param err [in,out] The error counter incremented if a mismatch is
  /// detected.
  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0, std::size_t& err) const {
    auto tmp_a = m_a(i0);
    auto tmp_b = m_b(i0);
    bool not_close =
        Kokkos::abs(tmp_a - tmp_b) > (m_atol + m_rtol * Kokkos::abs(tmp_b));
    err += static_cast<std::size_t>(not_close);
  }

  /// \brief Retrieves the computed error count.
  ///
  /// \return The total number of mismatches detected.
  const auto error() const { return m_error; }
};

/// \brief Computes the number of error mismatches between two 2D Kokkos views.
/// This structure performs an element-by-element comparison of two 2D views. It
/// counts the number of elements where the difference exceeds a specified
/// tolerance defined by an absolute tolerance and a relative tolerance.
///
/// \tparam ExecutionSpace The Kokkos execution space to run the
/// parallel_reduce. \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \tparam Layout The layout type of the Kokkos views.
/// \tparam iType The integer type used for indexing the view elements.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, KokkosLayout Layout, typename iType>
struct ViewErrors<ExecutionSpace, AViewType, BViewType, Layout, 2, iType> {
  AViewType m_a;
  BViewType m_b;

  double m_rtol;
  double m_atol;

  std::size_t m_error = 0;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type = Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                            Kokkos::IndexType<iType>>;

  /// \brief Constructs the error counter and performs the error computation.
  ///
  /// \param a [in] First Kokkos view containing data to compare.
  /// \param b [in] Second Kokkos view containing data to compare against.
  /// \param rtol [in] Relative tolerance for comparing the view elements
  /// (default 1.e-5).
  /// \param atol [in] Absolute tolerance for comparing the view elements
  /// (default 1.e-8).
  /// \param space [in] The Kokkos execution space used to launch the parallel
  /// reduction.
  ViewErrors(const AViewType& a, const BViewType& b, double rtol = 1.e-5,
             double atol = 1.e-8, const ExecutionSpace space = ExecutionSpace())
      : m_a(a), m_b(b), m_rtol(rtol), m_atol(atol) {
    Kokkos::parallel_reduce(
        "ViewErrors-2D", policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
        *this, m_error);
  }

  /// \brief Operator called by Kokkos to perform the comparison of each
  /// element.
  ///
  /// \param i0 [in] The index along the first dimension of the element in the
  /// views.
  /// \param i1 [in] The index along the second dimension of the element in the
  /// views.
  /// \param err [in,out] The error counter incremented if a mismatch is
  /// detected.
  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, std::size_t& err) const {
    auto tmp_a = m_a(i0, i1);
    auto tmp_b = m_b(i0, i1);
    bool not_close =
        Kokkos::abs(tmp_a - tmp_b) > (m_atol + m_rtol * Kokkos::abs(tmp_b));
    err += static_cast<std::size_t>(not_close);
  };

  /// \brief Retrieves the computed error count.
  ///
  /// \return The total number of mismatches detected.
  const auto error() const { return m_error; }
};

/// \brief Finds errors in 1D Kokkos views by comparing two views element-wise.
/// This structure compares corresponding elements from two 1D Kokkos views and
/// records errors when the difference exceeds a combined tolerance (absolute
/// and relative). The error values from the first and second views along with
/// their index information are stored in separate Kokkos views.
///
/// \tparam ExecutionSpace The Kokkos execution space where the parallel_for is
/// executed.
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \tparam Layout The memory layout type of the Kokkos views.
/// \tparam iType The integer type used for indexing the view elements.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, KokkosLayout Layout, typename iType>
struct FindErrors<ExecutionSpace, AViewType, BViewType, Layout, 1, iType> {
  using a_value_type = typename AViewType::non_const_value_type;
  using b_value_type = typename BViewType::non_const_value_type;

  using AErrorViewType = Kokkos::View<a_value_type*, ExecutionSpace>;
  using BErrorViewType = Kokkos::View<b_value_type*, ExecutionSpace>;
  using CountViewType  = Kokkos::View<std::size_t**, ExecutionSpace>;

  AViewType m_a;
  BViewType m_b;

  AErrorViewType m_a_error;
  BErrorViewType m_b_error;
  CountViewType m_loc_error;

  double m_rtol;
  double m_atol;

  using policy_type =
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;

  /// \brief Constructs a FindErrors object and performs the error detection.
  /// This constructor initializes the output error views and launches a
  /// parallel kernel to scan through the input view elements. For each element,
  /// if the difference between the two views exceeds the specified tolerance,
  /// the corresponding error values and their index are recorded.
  ///
  /// \param a [in] The first Kokkos view containing data.
  /// \param b [in] The second Kokkos view containing data to compare against.
  /// \param nb_errors [in] The maximum number of errors expected.
  /// \param rtol [in] Relative tolerance for the comparison (default 1.e-5).
  /// \param atol [in] Absolute tolerance for the comparison (default 1.e-8).
  /// \param space [in] The execution space used to launch the parallel kernel.
  FindErrors(const AViewType& a, const BViewType& b, const iType nb_errors,
             double rtol = 1.e-5, double atol = 1.e-8,
             const ExecutionSpace space = ExecutionSpace())
      : m_a(a),
        m_b(b),
        m_a_error("a_error", nb_errors),
        m_b_error("b_error", nb_errors),
        m_loc_error("loc_error", nb_errors, 2),
        m_rtol(rtol),
        m_atol(atol) {
    Kokkos::parallel_for("FindErrors-1D", policy_type(space, 0, m_a.extent(0)),
                         *this);
  }

  ///\brief Executes the element-wise comparison for the given index.
  /// This operator is invoked in parallel by Kokkos. For each index, it
  /// compares the corresponding elements from the two views. If the absolute
  /// difference exceeds the tolerance, it stores the error values and their
  /// index.
  ///
  ///\param i0 [in] The index of the element in the views.
  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0) const {
    iType err  = 0;
    auto tmp_a = m_a(i0);
    auto tmp_b = m_b(i0);
    bool not_close =
        Kokkos::abs(tmp_a - tmp_b) > (m_atol + m_rtol * Kokkos::abs(tmp_b));
    iType tmp_error = static_cast<iType>(not_close);
    Kokkos::atomic_add(&tmp_error, err);

    if (not_close) {
      m_a_error(err)      = tmp_a;
      m_b_error(err)      = tmp_b;
      m_loc_error(err, 0) = i0;
      m_loc_error(err, 1) = i0;
    }
  }

  ///\brief Retrieves the error information.
  ///
  ///\return A tuple containing the error view of the first input, the error
  /// view of the second input, and the error locations.
  const auto error_info() const {
    return std::make_tuple(m_a_error, m_b_error, m_loc_error);
  }
};

/// \brief Finds errors in 2D Kokkos views by comparing two views element-wise.
/// This structure compares corresponding elements from two 2D Kokkos views and
/// records errors when the difference exceeds a combined tolerance (absolute
/// and relative). The error values from the first and second views along with
/// their index information are stored in separate Kokkos views.
///
/// \tparam ExecutionSpace The Kokkos execution space where the parallel_for is
/// executed.
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \tparam Layout The memory layout type of the Kokkos views.
/// \tparam iType The integer type used for indexing the view elements.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, KokkosLayout Layout, typename iType>
struct FindErrors<ExecutionSpace, AViewType, BViewType, Layout, 2, iType> {
  using a_value_type = typename AViewType::non_const_value_type;
  using b_value_type = typename BViewType::non_const_value_type;

  using AErrorViewType = Kokkos::View<a_value_type*, ExecutionSpace>;
  using BErrorViewType = Kokkos::View<b_value_type*, ExecutionSpace>;
  using CountViewType  = Kokkos::View<std::size_t**, ExecutionSpace>;

  AViewType m_a;
  BViewType m_b;

  AErrorViewType m_a_error;
  BErrorViewType m_b_error;
  CountViewType m_loc_error;

  double m_rtol;
  double m_atol;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Impl::layout_iterate_type_selector<
          Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type = Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                            Kokkos::IndexType<iType>>;

  ///\brief Constructs a FindErrors object and performs the error detection.
  /// This constructor initializes the output error views and launches a
  /// parallel kernel to scan through the input view elements. For each element,
  /// if the difference between the two views exceeds the specified tolerance,
  /// the corresponding error values and their index are recorded.
  ///
  ///\param a [in] The first Kokkos view containing data.
  ///\param b [in] The second Kokkos view containing data to compare against.
  ///\param nb_errors [in] The maximum number of errors expected.
  ///\param rtol [in] Relative tolerance for the comparison (default 1.e-5).
  ///\param atol [in] Absolute tolerance for the comparison (default 1.e-8).
  ///\param space [in] The execution space used to launch the parallel kernel.
  FindErrors(const AViewType& a, const BViewType& b, const iType nb_errors,
             double rtol = 1.e-5, double atol = 1.e-8,
             const ExecutionSpace space = ExecutionSpace())
      : m_a(a),
        m_b(b),
        m_a_error("a_error", nb_errors),
        m_b_error("b_error", nb_errors),
        m_loc_error("loc_error", nb_errors, 3),
        m_rtol(rtol),
        m_atol(atol) {
    Kokkos::parallel_for("FindErrors-2D",
                         policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
                         *this);
  }

  /// \brief Executes the element-wise comparison for the given index.
  /// This operator is invoked in parallel by Kokkos. For each index, it
  /// compares the corresponding elements from the two views. If the absolute
  /// difference exceeds the tolerance, it stores the error values and their
  /// index.
  ///
  /// \param i0 [in] The index along the first dimension of the element in the
  /// views.
  /// \param i1 [in] The index along the second dimension of the element in the
  /// views.
  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0, const iType i1) const {
    iType err  = 0;
    auto tmp_a = m_a(i0, i1);
    auto tmp_b = m_b(i0, i1);
    bool not_close =
        Kokkos::abs(tmp_a - tmp_b) > (m_atol + m_rtol * Kokkos::abs(tmp_b));
    iType tmp_error = static_cast<iType>(not_close);
    Kokkos::atomic_add(&tmp_error, err);

    if (not_close) {
      m_a_error(err)      = tmp_a;
      m_b_error(err)      = tmp_b;
      m_loc_error(err, 0) = i0 + i1 * m_a.extent(0);
      m_loc_error(err, 1) = i0;
      m_loc_error(err, 2) = i1;
    }
  }

  /// \brief Retrieves the error information.
  ///
  /// \return A tuple containing the error view of the first input, the error
  /// view of the second input, and the error locations.
  const auto error_info() const {
    return std::make_tuple(m_a_error, m_b_error, m_loc_error);
  }
};

/// \brief Computes the number of mismatch errors between two Kokkos views.
/// This function performs an element-wise comparison between two Kokkos views
/// and counts the number of mismatches where the absolute difference exceeds
/// the combined absolute and relative tolerances.
///
/// \tparam ExecutionSpace The type of the Kokkos execution space.
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \param exec [in] The execution space used to launch the parallel reduction.
/// \param a [in] The first Kokkos view containing data to compare.
/// \param b [in] The second Kokkos view containing data to compare against.
/// \param rtol [in] Relative tolerance for comparing the view elements
/// (default 1.e-5).
/// \param atol [in] Absolute tolerance for comparing the view elements
/// (default 1.e-8).
/// \return The total number of mismatches detected.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType>
  requires KokkosViewAccesible<ExecutionSpace, AViewType> &&
           KokkosViewAccesible<ExecutionSpace, BViewType>
std::size_t count_errors(const ExecutionSpace& exec, const AViewType& a,
                         const BViewType& b, double rtol = 1.e-5,
                         double atol = 1.e-8) {
  // Figure out iteration order in case we need it
  Kokkos::Iterate iterate = get_iteration_order(a);

  if ((a.span() >= size_t(std::numeric_limits<int>::max())) ||
      (b.span() >= size_t(std::numeric_limits<int>::max()))) {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType,
                       Kokkos::LayoutRight, AViewType::rank(), std::size_t>
          view_errors(a, b, rtol, atol, exec);
      return view_errors.error();
    } else {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType, Kokkos::LayoutLeft,
                       AViewType::rank(), std::size_t>
          view_errors(a, b, rtol, atol, exec);
      return view_errors.error();
    }
  } else {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType,
                       Kokkos::LayoutRight, AViewType::rank(), int>
          view_errors(a, b, rtol, atol, exec);
      return view_errors.error();
    } else {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType, Kokkos::LayoutLeft,
                       AViewType::rank(), int>
          view_errors(a, b, rtol, atol, exec);
      return view_errors.error();
    }
  }
}

/// \brief Finds error information between two Kokkos views using element-wise
/// comparison. This function creates an instance of the appropriate FindErrors
/// structure (based on the rank of the input view) and launches a parallel
/// kernel to compare the elements of the two provided Kokkos views. It then
/// returns a tuple containing the error view for the first input, the error
/// view for the second input, and the locations of the errors.
///
/// \tparam ExecutionSpace The Kokkos execution space type used for the parallel
/// operation.
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \param exec [in] The execution space used to launch the parallel kernel.
/// \param a [in] The first Kokkos view containing data.
/// \param b [in] The second Kokkos view containing data to compare against.
/// \param nb_errors [in] The maximum number of errors expected.
/// \param rtol [in] Relative tolerance for the element-wise comparison
/// (default 1.e-5).
/// \param atol [in] Absolute tolerance for the element-wise comparison
/// (default 1.e-8).
/// \return A tuple containing:
///         - The Kokkos view of error values from the first view.
///         - The Kokkos view of error values from the second view.
///         - The Kokkos view of error location indices.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType>
  requires KokkosViewAccesible<ExecutionSpace, AViewType> &&
           KokkosViewAccesible<ExecutionSpace, BViewType>
auto find_errors(const ExecutionSpace& exec, const AViewType& a,
                 const BViewType& b, const std::size_t nb_errors,
                 double rtol = 1.e-5, double atol = 1.e-8) {
  // Figure out iteration order in case we need it
  Kokkos::Iterate iterate = get_iteration_order(a);

  if ((a.span() >= size_t(std::numeric_limits<int>::max())) ||
      (b.span() >= size_t(std::numeric_limits<int>::max()))) {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType,
                       Kokkos::LayoutRight, AViewType::rank(), std::size_t>
          view_errors(a, b, nb_errors, rtol, atol, exec);
      return view_errors.error_info();
    } else {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType, Kokkos::LayoutLeft,
                       AViewType::rank(), std::size_t>
          view_errors(a, b, nb_errors, rtol, atol, exec);
      return view_errors.error_info();
    }
  } else {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType,
                       Kokkos::LayoutRight, AViewType::rank(), int>
          view_errors(a, b, nb_errors, rtol, atol, exec);
      return view_errors.error_info();
    } else {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType, Kokkos::LayoutLeft,
                       AViewType::rank(), int>
          view_errors(a, b, nb_errors, rtol, atol, exec);
      return view_errors.error_info();
    }
  }
}

}  // namespace Impl
}  // namespace Testing
}  // namespace KokkosFFT

#endif
