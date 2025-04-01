// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <tuple>
#include <type_traits>
#include <Kokkos_Core.hpp>

template <typename ExecutionSpace, typename AViewType, typename BViewType>
bool allclose(const ExecutionSpace& exec, const AViewType& a,
              const BViewType& b, double rtol = 1.e-5, double atol = 1.e-8) {
  constexpr std::size_t rank = AViewType::rank;
  for (std::size_t i = 0; i < rank; i++) {
    assert(a.extent(i) == b.extent(i));
  }
  const auto n = a.size();

  auto* ptr_a = a.data();
  auto* ptr_b = b.data();

  int error = 0;
  Kokkos::parallel_reduce(
      "KokkosFFT::Test::allclose",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(exec,
                                                                          0, n),
      KOKKOS_LAMBDA(const std::size_t& i, int& err) {
        auto tmp_a = ptr_a[i];
        auto tmp_b = ptr_b[i];
        bool not_close =
            Kokkos::abs(tmp_a - tmp_b) > (atol + rtol * Kokkos::abs(tmp_b));
        err += static_cast<int>(not_close);
      },
      error);

  return error == 0;
}

template <typename ExecutionSpace, typename ViewType, typename T>
void multiply(const ExecutionSpace& exec, ViewType& x, T a) {
  const auto n = x.size();
  auto* ptr_x  = x.data();

  Kokkos::parallel_for(
      "KokkosFFT::Test::multiply",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(exec,
                                                                          0, n),
      KOKKOS_LAMBDA(const std::size_t& i) { ptr_x[i] = ptr_x[i] * a; });
}

template <typename T>
void display(std::string name, std::vector<T>& values) {
  std::cout << name << std::endl;
  for (auto value : values) {
    std::cout << value << std::endl;
  }
}

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

template <class TupleOfTuples, class Tuple>
struct for_each_tuple_cat;

template <class... Tuples, class Tuple>
struct for_each_tuple_cat<std::tuple<Tuples...>, Tuple> {
  using type = std::tuple<decltype(std::tuple_cat(std::declval<Tuples>(),
                                                  std::declval<Tuple>()))...>;
};

/// Construct a tuple of tuples that is the result of the concatenation of the
/// tuples in TupleOfTuples with Tuple.
template <class TupleOfTuples, class Tuple>
using for_each_tuple_cat_t =
    typename for_each_tuple_cat<TupleOfTuples, Tuple>::type;

static_assert(
    std::is_same_v<for_each_tuple_cat_t<std::tuple<std::tuple<double, double>,
                                                   std::tuple<int, double>>,
                                        std::tuple<int>>,
                   std::tuple<std::tuple<double, double, int>,
                              std::tuple<int, double, int>>>);

static_assert(
    std::is_same_v<for_each_tuple_cat_t<std::tuple<std::tuple<double, double>>,
                                        std::tuple<int>>,
                   std::tuple<std::tuple<double, double, int>>>);

template <class InTupleOfTuples, class OutTupleOfTuples>
struct cartesian_product_impl;

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

#endif
