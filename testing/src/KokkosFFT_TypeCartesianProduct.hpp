// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TYPE_CARTESIAN_PRODUCT_HPP
#define KOKKOSFFT_TYPE_CARTESIAN_PRODUCT_HPP

#include <tuple>
#include <type_traits>

namespace KokkosFFT {
namespace Testing {
namespace Impl {

/// Transform a sequence S to a tuple:
/// - a std::integer_sequence<T, Ints...> to a
/// std::tuple<std::integral_constant<T, Ints>...>
/// - a std::pair<T, U> to a std::tuple<T, U>
/// - identity otherwise (std::tuple)
template <class S>
struct to_tuple {
  using type = S;
};

template <class T, T... Ints>
struct to_tuple<std::integer_sequence<T, Ints...>> {
  using type = std::tuple<std::integral_constant<T, Ints>...>;
};

template <class T, class U>
struct to_tuple<std::pair<T, U>> {
  using type = std::tuple<T, U>;
};

template <class S>
using to_tuple_t = typename to_tuple<S>::type;

template <class TupleOfTuples, class Type>
struct for_each_tuple_cat;

template <class... Tuples, class Type>
struct for_each_tuple_cat<std::tuple<Tuples...>, Type> {
  using type = std::tuple<decltype(std::tuple_cat(
      std::declval<Tuples>(), std::declval<std::tuple<Type>>()))...>;
};

template <class... Tuples, class Type>
struct for_each_tuple_cat<std::tuple<Tuples...>, std::tuple<Type>> {
  using type = std::tuple<decltype(std::tuple_cat(
      std::declval<Tuples>(), std::declval<std::tuple<Type>>()))...>;
};

/// Construct a tuple of tuples that is the result of the concatenation of the
/// tuples in TupleOfTuples with Tuple.
template <class TupleOfTuples, class Type>
using for_each_tuple_cat_t =
    typename for_each_tuple_cat<TupleOfTuples, Type>::type;

template <class InTupleOfTuples, class OutTupleOfTuples>
struct cartesian_product_impl;

/// \brief Recursive case: The cartesian product is built in a recursive manner.
/// Each time, each element of the tuple in std::tuple<HeardArgs...> is added to
/// all the tuples in OutTupleOfTuples. Then the process is repeated with the
/// next tuple that is present in TailTuples...
///
/// \tparam HeadArgs The types of the first tuple in the input tuple of tuples
/// \tparam TailTuples The remaining tuples in the input tuple of tuples
/// \tparam OutTupleOfTuples The output tuple of tuples
template <class... HeadArgs, class... TailTuples, class OutTupleOfTuples>
struct cartesian_product_impl<
    std::tuple<std::tuple<HeadArgs...>, TailTuples...>, OutTupleOfTuples>
    : cartesian_product_impl<
          std::tuple<TailTuples...>,
          decltype(std::tuple_cat(
              std::declval<for_each_tuple_cat_t<OutTupleOfTuples,
                                                std::tuple<HeadArgs>>>()...))> {
};

template <class OutTupleOfTuples>
struct cartesian_product_impl<std::tuple<>, OutTupleOfTuples> {
  using type = OutTupleOfTuples;
};

/// Generate a std::tuple cartesian product from multiple tuple-like structures
/// (std::tuple, std::integer_sequence and std::pair) Do not rely on the
/// ordering result.
template <class... InTuplesLike>
using cartesian_product_t =
    typename cartesian_product_impl<std::tuple<to_tuple_t<InTuplesLike>...>,
                                    std::tuple<std::tuple<>>>::type;

/// Transform a std::tuple<Args...> to a testing::Types<Args...>, identity
/// otherwise
template <class T>
struct tuple_to_types {
  using type = T;
};

template <class... Args>
struct tuple_to_types<std::tuple<Args...>> {
  using type = testing::Types<Args...>;
};

template <class T>
using tuple_to_types_t = typename tuple_to_types<T>::type;

}  // namespace Impl

// API to generate caresian product of input types
template <class... InTuplesLike>
using make_cartesian_types =
    Impl::tuple_to_types_t<Impl::cartesian_product_t<InTuplesLike...>>;

}  // namespace Testing
}  // namespace KokkosFFT

#endif
