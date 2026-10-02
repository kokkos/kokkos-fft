#ifndef KOKKOSFFT_BATCHED_BASE_TYPES_HPP
#define KOKKOSFFT_BATCHED_BASE_TYPES_HPP

#include "KokkosFFT_Batched_fwd.hpp"
#include <cstddef>
#include <type_traits>

namespace KokkosFFT {
namespace Batched {

/// \brief Kind of a transform, deduced from the in/out value types.
/// C2C: complex -> complex, R2C: real -> complex, C2R: complex -> real
enum class TransformKind { C2C, R2C, C2R };

template <int Tag, typename Axes>
struct PrependAxisTag;

template <int Tag, int... Tags>
struct PrependAxisTag<Tag, AxisTag<Tags...>> {
  using type = AxisTag<Tag, Tags...>;
};

template <int Tag, typename Axes>
using prepend_axis_tag_t = typename PrependAxisTag<Tag, Axes>::type;

/// \brief Base case: represents an empty axis list.
/// It provides a tail type for the recursion and a rank of 0.
template <>
struct AxisTag<> {
  using heads                       = AxisTag<>;
  static constexpr std::size_t rank = 0;
};

/// \brief Recursive case: for one or more tags.
/// 'head' represents the current tag as an integral constant.
/// 'tail' recursively represents the remaining tags.
/// 'heads' represents all axes except the last axis.
/// 'last' represents the last tag as an integral constant.
/// The rank is computed as 1 (for the head) plus the rank of the tail.
/// Example usage:
/// using MyAxes = AxisTag<0, 2, 4>;
/// MyAxes::head is std::integral_constant<int, 0>
/// MyAxes::tail is AxisTag<2, 4>
/// MyAxes::heads is AxisTag<0, 2>
/// MyAxes::last is std::integral_constant<int, 4>
/// MyAxes::rank is 3
///
/// \tparam Tag The current axis tag (an integer)
/// \tparam Rest The remaining axis tags (a parameter pack of integers)
template <int Tag>
struct AxisTag<Tag> {
  using head  = std::integral_constant<int, Tag>;
  using tail  = AxisTag<>;
  using heads = AxisTag<>;
  using last  = head;

  static constexpr std::size_t rank = 1;
  static constexpr int head_v       = head::value;
  static constexpr int tail_rank    = tail::rank;
  static constexpr int heads_rank   = heads::rank;
  static constexpr int last_v       = last::value;
};

template <int Tag, int Next, int... Rest>
struct AxisTag<Tag, Next, Rest...> {
  using head  = std::integral_constant<int, Tag>;
  using tail  = AxisTag<Next, Rest...>;
  using heads = prepend_axis_tag_t<Tag, typename tail::heads>;
  using last  = typename tail::last;

  /// The rank is the current dimension (1) plus the rank of the tail.
  static constexpr std::size_t rank = 1 + tail::rank;
  static constexpr int head_v       = head::value;
  static constexpr int tail_rank    = tail::rank;
  static constexpr int heads_rank   = heads::rank;
  static constexpr int last_v       = last::value;
};

}  // namespace Batched
}  // namespace KokkosFFT

#endif
