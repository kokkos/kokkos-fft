// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_COUNT_ERRORS_HPP
#define KOKKOSFFT_COUNT_ERRORS_HPP

#include <utility>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Concepts.hpp"

namespace KokkosFFT {
namespace Testing {
namespace Impl {

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

/// \brief Computes the number of error mismatches between two Kokkos views.
/// This structure performs an element-by-element comparison of two views. It
/// counts the number of elements where the difference exceeds a specified
/// tolerance defined by an absolute tolerance and a relative tolerance.
///
/// \tparam ExecutionSpace The Kokkos execution space to run the
/// parallel_reduce.
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \tparam ComparisonOp The binary operation to apply to the elements of the
/// views.
/// \tparam Layout The layout type of the Kokkos views.
/// \tparam iType The integer type used for indexing the view elements.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, typename ComparisonOp, KokkosLayout Layout,
          typename iType>
struct ViewErrors {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(AViewType::rank(), std::size_t(6));

  //! The error counter
  std::size_t m_error = 0;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space  The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] a The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const AViewType a) const {
    if constexpr (AViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, a.extent(0));
    } else {
      static const Kokkos::Iterate outer_iteration_pattern =
          Kokkos::Impl::layout_iterate_type_selector<
              Layout>::outer_iteration_pattern;
      static const Kokkos::Iterate inner_iteration_pattern =
          Kokkos::Impl::layout_iterate_type_selector<
              Layout>::inner_iteration_pattern;
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, outer_iteration_pattern,
                       inner_iteration_pattern>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins = {};
      Kokkos::Array<std::size_t, m_rank_truncated> ends   = {};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = a.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructs the error counter and performs the error computation.
  ///
  /// \param[in] a First Kokkos view containing data to compare.
  /// \param[in] b Second Kokkos view containing data to compare against.
  /// \param[in] op The binary operation used for comparison
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  ViewErrors(const AViewType& a, const BViewType& b, ComparisonOp op,
             const ExecutionSpace space = ExecutionSpace()) {
    Kokkos::parallel_reduce(
        "ViewErrors", get_policy(space, a),
        CountErrors<std::make_index_sequence<m_rank_truncated>>(a, b, op),
        m_error);
  }

  /// \brief Helper functor to count the number of errors in the views.
  ///
  /// \tparam IndexSequence The index sequence used for the parallel execution.
  template <typename IndexSequence>
  struct CountErrors;

  template <std::size_t... Idx>
  struct CountErrors<std::index_sequence<Idx...>> {
    template <std::size_t I>
    using IndicesType = iType;

    AViewType m_a;
    BViewType m_b;

    ComparisonOp m_op;

    /// \brief Constructs the error counter and performs the error computation.
    ///
    /// \param[in] a First Kokkos view containing data to compare.
    /// \param[in] b Second Kokkos view containing data to compare against.
    /// \param[in] p The binary operation used for comparison
    CountErrors(const AViewType& a, const BViewType& b, ComparisonOp op)
        : m_a(a), m_b(b), m_op(op) {}

    /// \brief Operator called by Kokkos to perform the comparison of each
    /// element.
    ///
    /// \param[in] indices The indices of the element in the views up to 6D.
    /// \param[in,out] err  The error counter incremented if a mismatch is
    /// detected.
    KOKKOS_INLINE_FUNCTION
    void operator()(const IndicesType<Idx>... indices, std::size_t& err) const {
      if constexpr (AViewType::rank() <= 6) {
        bool close = m_op(m_a(indices...), m_b(indices...));
        err += static_cast<std::size_t>(!close);
      } else if constexpr (AViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_a.extent(6)); i6++) {
          auto tmp_a = m_a(indices..., i6);
          auto tmp_b = m_b(indices..., i6);
          bool close = m_op(tmp_a, tmp_b);
          err += static_cast<std::size_t>(!close);
        }
      } else if constexpr (AViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_a.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_a.extent(7)); i7++) {
            auto tmp_a = m_a(indices..., i6, i7);
            auto tmp_b = m_b(indices..., i6, i7);
            bool close = m_op(tmp_a, tmp_b);
            err += static_cast<std::size_t>(!close);
          }
        }
      }
    }
  };

  /// \brief Retrieves the computed error count.
  ///
  /// \return The total number of mismatches detected.
  auto error() const { return m_error; }
};

/// \brief Finds errors in Kokkos views by comparing two views element-wise.
/// This structure compares corresponding elements from two Kokkos views and
/// records errors when the difference exceeds a combined tolerance (absolute
/// and relative). The error values from the first and second views along with
/// their index information are stored in separate Kokkos views.
///
/// \tparam ExecutionSpace The Kokkos execution space where the parallel_for is
/// executed.
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
/// \tparam ComparisonOp The binary operation to apply to the elements of the
/// views.
/// \tparam Layout The memory layout type of the Kokkos views.
/// \tparam iType The integer type used for indexing the view elements.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, typename ComparisonOp, KokkosLayout Layout,
          typename iType>
struct FindErrors {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(AViewType::rank(), std::size_t(6));
  using a_value_type = typename AViewType::non_const_value_type;
  using b_value_type = typename BViewType::non_const_value_type;

  using AErrorViewType = Kokkos::View<a_value_type*, ExecutionSpace>;
  using BErrorViewType = Kokkos::View<b_value_type*, ExecutionSpace>;
  using CountViewType  = Kokkos::View<std::size_t**, ExecutionSpace>;
  using CountType      = Kokkos::View<std::size_t, ExecutionSpace>;

  AErrorViewType m_a_error_pub;
  BErrorViewType m_b_error_pub;
  CountViewType m_loc_error_pub;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] a The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const AViewType a) const {
    if constexpr (AViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, a.extent(0));
    } else {
      static const Kokkos::Iterate outer_iteration_pattern =
          Kokkos::Impl::layout_iterate_type_selector<
              Layout>::outer_iteration_pattern;
      static const Kokkos::Iterate inner_iteration_pattern =
          Kokkos::Impl::layout_iterate_type_selector<
              Layout>::inner_iteration_pattern;
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, outer_iteration_pattern,
                       inner_iteration_pattern>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins = {};
      Kokkos::Array<std::size_t, m_rank_truncated> ends   = {};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = a.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  ///\brief Constructs a FindErrors object and performs the error detection.
  /// This constructor initializes the output error views and launches a
  /// parallel kernel to scan through the input view elements. For each element,
  /// if the difference between the two views exceeds the specified tolerance,
  /// the corresponding error values and their index are recorded.
  ///
  ///\param[in] a The first Kokkos view containing data.
  ///\param[in] b The second Kokkos view containing data to compare against.
  ///\param[in] nb_errors The maximum number of errors expected.
  ///\param[in] op The binary operation used for comparison.
  ///\param[in] space The execution space used to launch the parallel kernel.
  FindErrors(const AViewType& a, const BViewType& b, const iType nb_errors,
             ComparisonOp op, const ExecutionSpace space = ExecutionSpace())
      : m_a_error_pub("a_error", nb_errors),
        m_b_error_pub("b_error", nb_errors),
        m_loc_error_pub("loc_error", nb_errors, AViewType::rank() + 1) {
    Kokkos::parallel_for(
        "FindErrors", get_policy(space, a),
        FindErrorsInternal<std::make_index_sequence<m_rank_truncated>>(
            a, b, m_a_error_pub, m_b_error_pub, m_loc_error_pub, op));
  }

  template <typename IndexSequence>
  struct FindErrorsInternal;

  template <std::size_t... Idx>
  struct FindErrorsInternal<std::index_sequence<Idx...>> {
    template <std::size_t I>
    using IndicesType = iType;

    AViewType m_a;
    BViewType m_b;

    AErrorViewType m_a_error;
    BErrorViewType m_b_error;
    CountViewType m_loc_error;

    //! The error counter
    CountType m_count;

    ComparisonOp m_op;

    ///\brief Constructs a FindErrorsInternal object and performs the error
    /// detection.
    /// This constructor initializes the output error views and launches a
    /// parallel kernel to scan through the input view elements. For each
    /// element, if the difference between the two views exceeds the specified
    /// tolerance, the corresponding error values and their index are recorded.
    ///
    ///\param[in] a The first Kokkos view containing data.
    ///\param[in] b The second Kokkos view containing data to compare against.
    ///\param[in] nb_errors The maximum number of errors expected.
    ///\param[in] op The binary operation used for comparison.
    FindErrorsInternal(const AViewType& a, const BViewType& b,
                       const AErrorViewType& a_error,
                       const BErrorViewType& b_error,
                       const CountViewType& loc_error, ComparisonOp op)
        : m_a(a),
          m_b(b),
          m_a_error(a_error),
          m_b_error(b_error),
          m_loc_error(loc_error),
          m_count("count"),
          m_op(op) {}

    /// \brief Executes the element-wise comparison for the given index.
    /// This operator is invoked in parallel by Kokkos. For each index, it
    /// compares the corresponding elements from the two views. If the absolute
    /// difference exceeds the tolerance, it stores the error values and their
    /// index.
    ///
    /// \param[in] indices The indices of the element in the views up to 6D.
    KOKKOS_INLINE_FUNCTION
    void operator()(const IndicesType<Idx>... indices) const {
      if constexpr (AViewType::rank() <= 6) {
        auto tmp_a = m_a(indices...);
        auto tmp_b = m_b(indices...);
        bool close = m_op(tmp_a, tmp_b);
        if (!close) {
          std::size_t count = Kokkos::atomic_fetch_add(m_count.data(), 1);
          m_a_error(count)  = tmp_a;
          m_b_error(count)  = tmp_b;
          iType error_indices[AViewType::rank()] = {indices...};
          m_loc_error(count, 0) = get_global_idx(error_indices);
          for (std::size_t i = 0; i < AViewType::rank(); i++) {
            m_loc_error(count, i + 1) = error_indices[i];
          }
        }
      } else if constexpr (AViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_a.extent(6)); i6++) {
          auto tmp_a = m_a(indices..., i6);
          auto tmp_b = m_b(indices..., i6);
          bool close = m_op(tmp_a, tmp_b);
          if (!close) {
            std::size_t count = Kokkos::atomic_fetch_add(m_count.data(), 1);
            m_a_error(count)  = tmp_a;
            m_b_error(count)  = tmp_b;
            iType error_indices[AViewType::rank()] = {indices..., i6};
            m_loc_error(count, 0) = get_global_idx(error_indices);
            for (std::size_t i = 0; i < AViewType::rank(); i++) {
              m_loc_error(count, i + 1) = error_indices[i];
            }
          }
        }
      } else if constexpr (AViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_a.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_a.extent(7)); i7++) {
            auto tmp_a = m_a(indices..., i6, i7);
            auto tmp_b = m_b(indices..., i6, i7);
            bool close = m_op(tmp_a, tmp_b);
            if (!close) {
              std::size_t count = Kokkos::atomic_fetch_add(m_count.data(), 1);
              m_a_error(count)  = tmp_a;
              m_b_error(count)  = tmp_b;
              iType error_indices[AViewType::rank()] = {indices..., i6, i7};
              m_loc_error(count, 0) = get_global_idx(error_indices);
              for (std::size_t i = 0; i < AViewType::rank(); i++) {
                m_loc_error(count, i + 1) = error_indices[i];
              }
            }
          }
        }
      }
    }

    /// \brief Get the flatten index in LayoutLeft order.
    ///
    /// \param[in] error_indices The indices of the element in Views
    KOKKOS_INLINE_FUNCTION
    std::size_t get_global_idx(const iType error_indices[]) const {
      std::size_t global_idx = 0;
      std::size_t stride     = 1;
      for (std::size_t d = 0; d < AViewType::rank(); ++d) {
        global_idx += error_indices[d] * stride;
        stride *= m_a.extent(d);
      }
      return global_idx;
    }
  };

  /// \brief Retrieves the error information.
  ///
  /// \return A tuple containing the error view of the first input, the error
  /// view of the second input, and the error locations.
  auto error_info() const {
    return std::make_tuple(m_a_error_pub, m_b_error_pub, m_loc_error_pub);
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
/// \tparam ComparisonOp The type of the binary operation used for comparison.
///
/// \param[in] exec The execution space used to launch the parallel reduction.
/// \param[in] a The first Kokkos view containing data to compare.
/// \param[in] b The second Kokkos view containing data to compare against.
/// \param[in] op The binary operation used for comparison.
/// \return The total number of mismatches detected.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, typename ComparisonOp>
  requires KokkosViewAccessible<ExecutionSpace, AViewType> &&
           KokkosViewAccessible<ExecutionSpace, BViewType> &&
           (AViewType::rank() == BViewType::rank())
std::size_t count_errors(const ExecutionSpace& exec, const AViewType& a,
                         const BViewType& b, ComparisonOp op) {
  // Figure out iteration order in case we need it
  Kokkos::Iterate iterate = get_iteration_order(a);

  if ((a.span() >= size_t(std::numeric_limits<int>::max())) ||
      (b.span() >= size_t(std::numeric_limits<int>::max()))) {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutRight, int64_t>
          view_errors(a, b, op, exec);
      return view_errors.error();
    } else {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutLeft, int64_t>
          view_errors(a, b, op, exec);
      return view_errors.error();
    }
  } else {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutRight, int>
          view_errors(a, b, op, exec);
      return view_errors.error();
    } else {
      Impl::ViewErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutLeft, int>
          view_errors(a, b, op, exec);
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
/// \tparam ComparisonOp The type of the binary operation used for comparison.
///
/// \param[in] exec The execution space used to launch the parallel kernel.
/// \param[in] a The first Kokkos view containing data.
/// \param[in] b The second Kokkos view containing data to compare against.
/// \param[in] nb_errors The maximum number of errors expected.
/// \param[in] op The binary operation used for comparison.
/// \return A tuple containing:
///         - The Kokkos view of error values from the first view.
///         - The Kokkos view of error values from the second view.
///         - The Kokkos view of error location indices.
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType, typename ComparisonOp>
  requires KokkosViewAccessible<ExecutionSpace, AViewType> &&
           KokkosViewAccessible<ExecutionSpace, BViewType> &&
           (AViewType::rank() == BViewType::rank())
auto find_errors(const ExecutionSpace& exec, const AViewType& a,
                 const BViewType& b, const std::size_t nb_errors,
                 ComparisonOp op) {
  // Figure out iteration order in case we need it
  Kokkos::Iterate iterate = get_iteration_order(a);

  if ((a.span() >= size_t(std::numeric_limits<int>::max())) ||
      (b.span() >= size_t(std::numeric_limits<int>::max()))) {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutRight, int64_t>
          find_errors(a, b, nb_errors, op, exec);
      return find_errors.error_info();
    } else {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutLeft, int64_t>
          find_errors(a, b, nb_errors, op, exec);
      return find_errors.error_info();
    }
  } else {
    if (iterate == Kokkos::Iterate::Right) {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutRight, int>
          find_errors(a, b, nb_errors, op, exec);
      return find_errors.error_info();
    } else {
      Impl::FindErrors<ExecutionSpace, AViewType, BViewType, ComparisonOp,
                       Kokkos::LayoutLeft, int>
          find_errors(a, b, nb_errors, op, exec);
      return find_errors.error_info();
    }
  }
}

}  // namespace Impl
}  // namespace Testing
}  // namespace KokkosFFT

#endif
