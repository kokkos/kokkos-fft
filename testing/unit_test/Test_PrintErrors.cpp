// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_PrintErrors.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestSortErrors : public ::testing::Test {
  using float_type = T;
};

template <typename T>
struct TestPrintErrors : public ::testing::Test {
  using float_type = T;
};

template <typename T>
void test_sort_errors_1D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 1;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 2);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_loc_error(0, 0) = 0;  // global idx
  h_loc_error(0, 1) = 0;  // idx of dimension 0

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon                = std::numeric_limits<T>::epsilon();
  auto [loc, a_val, b_val] = error_map[0];
  EXPECT_LT(Kokkos::abs(a_val - h_a_error(0)), epsilon);
  EXPECT_LT(Kokkos::abs(b_val - h_b_error(0)), epsilon);
  EXPECT_EQ(loc.at(0), h_loc_error(0, 1));
}

template <typename T>
void test_sort_errors_2D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 3);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 3;  // global idx
  h_loc_error(0, 1) = 0;  // idx of dimension 0
  h_loc_error(0, 2) = 1;  // idx of dimension 1
  h_loc_error(1, 0) = 4;  // global idx
  h_loc_error(1, 1) = 1;  // idx of dimension 0
  h_loc_error(1, 2) = 1;  // idx of dimension 1

  std::vector<std::size_t> global_indices = {h_loc_error(0, 0),
                                             h_loc_error(1, 0)};

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i_error = 0; i_error < nb_errors; ++i_error) {
    auto [loc, a_val, b_val] = error_map[global_indices.at(i_error)];
    EXPECT_LT(Kokkos::abs(a_val - h_a_error(i_error)), epsilon);
    EXPECT_LT(Kokkos::abs(b_val - h_b_error(i_error)), epsilon);
    EXPECT_EQ(loc.at(0), h_loc_error(i_error, 1));
    EXPECT_EQ(loc.at(1), h_loc_error(i_error, 2));
  }
}

template <typename T>
void test_sort_errors_3D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 4);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 9;   // global idx
  h_loc_error(0, 1) = 0;   // idx of dimension 0
  h_loc_error(0, 2) = 1;   // idx of dimension 1
  h_loc_error(0, 3) = 1;   // idx of dimension 2
  h_loc_error(1, 0) = 16;  // global idx
  h_loc_error(1, 1) = 1;   // idx of dimension 0
  h_loc_error(1, 2) = 1;   // idx of dimension 1
  h_loc_error(1, 3) = 2;   // idx of dimension 2

  std::vector<std::size_t> global_indices = {h_loc_error(0, 0),
                                             h_loc_error(1, 0)};

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i_error = 0; i_error < nb_errors; ++i_error) {
    auto [loc, a_val, b_val] = error_map[global_indices.at(i_error)];
    EXPECT_LT(Kokkos::abs(a_val - h_a_error(i_error)), epsilon);
    EXPECT_LT(Kokkos::abs(b_val - h_b_error(i_error)), epsilon);
    EXPECT_EQ(loc.at(0), h_loc_error(i_error, 1));
    EXPECT_EQ(loc.at(1), h_loc_error(i_error, 2));
    EXPECT_EQ(loc.at(2), h_loc_error(i_error, 3));
  }
}

template <typename T>
void test_sort_errors_4D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 5);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 81;  // global idx
  h_loc_error(0, 1) = 0;   // idx of dimension 0
  h_loc_error(0, 2) = 1;   // idx of dimension 1
  h_loc_error(0, 3) = 1;   // idx of dimension 2
  h_loc_error(0, 4) = 3;   // idx of dimension 3
  h_loc_error(1, 0) = 64;  // global idx
  h_loc_error(1, 1) = 1;   // idx of dimension 0
  h_loc_error(1, 2) = 1;   // idx of dimension 1
  h_loc_error(1, 3) = 2;   // idx of dimension 2
  h_loc_error(1, 4) = 2;   // idx of dimension 3

  std::vector<std::size_t> global_indices = {h_loc_error(0, 0),
                                             h_loc_error(1, 0)};

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i_error = 0; i_error < nb_errors; ++i_error) {
    auto [loc, a_val, b_val] = error_map[global_indices.at(i_error)];
    EXPECT_LT(Kokkos::abs(a_val - h_a_error(i_error)), epsilon);
    EXPECT_LT(Kokkos::abs(b_val - h_b_error(i_error)), epsilon);
    EXPECT_EQ(loc.at(0), h_loc_error(i_error, 1));
    EXPECT_EQ(loc.at(1), h_loc_error(i_error, 2));
    EXPECT_EQ(loc.at(2), h_loc_error(i_error, 3));
    EXPECT_EQ(loc.at(3), h_loc_error(i_error, 4));
  }
}

template <typename T>
void test_sort_errors_5D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 6);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 201;  // global idx
  h_loc_error(0, 1) = 0;    // idx of dimension 0
  h_loc_error(0, 2) = 1;    // idx of dimension 1
  h_loc_error(0, 3) = 1;    // idx of dimension 2
  h_loc_error(0, 4) = 3;    // idx of dimension 3
  h_loc_error(0, 5) = 1;    // idx of dimension 4
  h_loc_error(1, 0) = 298;  // global idx
  h_loc_error(1, 1) = 1;    // idx of dimension 0
  h_loc_error(1, 2) = 1;    // idx of dimension 1
  h_loc_error(1, 3) = 2;    // idx of dimension 2
  h_loc_error(1, 4) = 2;    // idx of dimension 3
  h_loc_error(1, 5) = 1;    // idx of dimension 4

  std::vector<std::size_t> global_indices = {h_loc_error(0, 0),
                                             h_loc_error(1, 0)};

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i_error = 0; i_error < nb_errors; ++i_error) {
    auto [loc, a_val, b_val] = error_map[global_indices.at(i_error)];
    EXPECT_LT(Kokkos::abs(a_val - h_a_error(i_error)), epsilon);
    EXPECT_LT(Kokkos::abs(b_val - h_b_error(i_error)), epsilon);
    EXPECT_EQ(loc.at(0), h_loc_error(i_error, 1));
    EXPECT_EQ(loc.at(1), h_loc_error(i_error, 2));
    EXPECT_EQ(loc.at(2), h_loc_error(i_error, 3));
    EXPECT_EQ(loc.at(3), h_loc_error(i_error, 4));
    EXPECT_EQ(loc.at(4), h_loc_error(i_error, 5));
  }
}

template <typename T>
void test_sort_errors_6D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 7);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 561;  // global idx
  h_loc_error(0, 1) = 0;    // idx of dimension 0
  h_loc_error(0, 2) = 1;    // idx of dimension 1
  h_loc_error(0, 3) = 1;    // idx of dimension 2
  h_loc_error(0, 4) = 3;    // idx of dimension 3
  h_loc_error(0, 5) = 1;    // idx of dimension 4
  h_loc_error(0, 6) = 1;    // idx of dimension 5
  h_loc_error(1, 0) = 658;  // global idx
  h_loc_error(1, 1) = 1;    // idx of dimension 0
  h_loc_error(1, 2) = 1;    // idx of dimension 1
  h_loc_error(1, 3) = 2;    // idx of dimension 2
  h_loc_error(1, 4) = 2;    // idx of dimension 3
  h_loc_error(1, 5) = 1;    // idx of dimension 4
  h_loc_error(1, 6) = 1;    // idx of dimension 5

  std::vector<std::size_t> global_indices = {h_loc_error(0, 0),
                                             h_loc_error(1, 0)};

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i_error = 0; i_error < nb_errors; ++i_error) {
    auto [loc, a_val, b_val] = error_map[global_indices.at(i_error)];
    EXPECT_LT(Kokkos::abs(a_val - h_a_error(i_error)), epsilon);
    EXPECT_LT(Kokkos::abs(b_val - h_b_error(i_error)), epsilon);
    EXPECT_EQ(loc.at(0), h_loc_error(i_error, 1));
    EXPECT_EQ(loc.at(1), h_loc_error(i_error, 2));
    EXPECT_EQ(loc.at(2), h_loc_error(i_error, 3));
    EXPECT_EQ(loc.at(3), h_loc_error(i_error, 4));
    EXPECT_EQ(loc.at(4), h_loc_error(i_error, 5));
    EXPECT_EQ(loc.at(5), h_loc_error(i_error, 6));
  }
}

template <typename T>
void test_sort_errors_7D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 8);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 2001;  // global idx
  h_loc_error(0, 1) = 0;     // idx of dimension 0
  h_loc_error(0, 2) = 1;     // idx of dimension 1
  h_loc_error(0, 3) = 1;     // idx of dimension 2
  h_loc_error(0, 4) = 3;     // idx of dimension 3
  h_loc_error(0, 5) = 1;     // idx of dimension 4
  h_loc_error(0, 6) = 1;     // idx of dimension 5
  h_loc_error(0, 7) = 1;     // idx of dimension 6
  h_loc_error(1, 0) = 3538;  // global idx
  h_loc_error(1, 1) = 1;     // idx of dimension 0
  h_loc_error(1, 2) = 1;     // idx of dimension 1
  h_loc_error(1, 3) = 2;     // idx of dimension 2
  h_loc_error(1, 4) = 2;     // idx of dimension 3
  h_loc_error(1, 5) = 1;     // idx of dimension 4
  h_loc_error(1, 6) = 1;     // idx of dimension 5
  h_loc_error(1, 7) = 2;     // idx of dimension 6

  std::vector<std::size_t> global_indices = {h_loc_error(0, 0),
                                             h_loc_error(1, 0)};

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i_error = 0; i_error < nb_errors; ++i_error) {
    auto [loc, a_val, b_val] = error_map[global_indices.at(i_error)];
    EXPECT_LT(Kokkos::abs(a_val - h_a_error(i_error)), epsilon);
    EXPECT_LT(Kokkos::abs(b_val - h_b_error(i_error)), epsilon);
    EXPECT_EQ(loc.at(0), h_loc_error(i_error, 1));
    EXPECT_EQ(loc.at(1), h_loc_error(i_error, 2));
    EXPECT_EQ(loc.at(2), h_loc_error(i_error, 3));
    EXPECT_EQ(loc.at(3), h_loc_error(i_error, 4));
    EXPECT_EQ(loc.at(4), h_loc_error(i_error, 5));
    EXPECT_EQ(loc.at(5), h_loc_error(i_error, 6));
    EXPECT_EQ(loc.at(6), h_loc_error(i_error, 7));
  }
}

template <typename T>
void test_sort_errors_8D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 9);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 2001;  // global idx
  h_loc_error(0, 1) = 0;     // idx of dimension 0
  h_loc_error(0, 2) = 1;     // idx of dimension 1
  h_loc_error(0, 3) = 1;     // idx of dimension 2
  h_loc_error(0, 4) = 3;     // idx of dimension 3
  h_loc_error(0, 5) = 1;     // idx of dimension 4
  h_loc_error(0, 6) = 1;     // idx of dimension 5
  h_loc_error(0, 7) = 1;     // idx of dimension 6
  h_loc_error(0, 8) = 0;     // idx of dimension 7
  h_loc_error(1, 0) = 3538;  // global idx
  h_loc_error(1, 1) = 1;     // idx of dimension 0
  h_loc_error(1, 2) = 1;     // idx of dimension 1
  h_loc_error(1, 3) = 2;     // idx of dimension 2
  h_loc_error(1, 4) = 2;     // idx of dimension 3
  h_loc_error(1, 5) = 1;     // idx of dimension 4
  h_loc_error(1, 6) = 1;     // idx of dimension 5
  h_loc_error(1, 7) = 2;     // idx of dimension 6
  h_loc_error(1, 8) = 0;     // idx of dimension 7

  std::vector<std::size_t> global_indices = {h_loc_error(0, 0),
                                             h_loc_error(1, 0)};

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_error, b_error, loc_error);

  T epsilon = std::numeric_limits<T>::epsilon();
  for (std::size_t i_error = 0; i_error < nb_errors; ++i_error) {
    auto [loc, a_val, b_val] = error_map[global_indices.at(i_error)];
    EXPECT_LT(Kokkos::abs(a_val - h_a_error(i_error)), epsilon);
    EXPECT_LT(Kokkos::abs(b_val - h_b_error(i_error)), epsilon);
    EXPECT_EQ(loc.at(0), h_loc_error(i_error, 1));
    EXPECT_EQ(loc.at(1), h_loc_error(i_error, 2));
    EXPECT_EQ(loc.at(2), h_loc_error(i_error, 3));
    EXPECT_EQ(loc.at(3), h_loc_error(i_error, 4));
    EXPECT_EQ(loc.at(4), h_loc_error(i_error, 5));
    EXPECT_EQ(loc.at(5), h_loc_error(i_error, 6));
    EXPECT_EQ(loc.at(6), h_loc_error(i_error, 7));
    EXPECT_EQ(loc.at(7), h_loc_error(i_error, 8));
  }
}

template <typename T>
void test_print_errors_1D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 1;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 2);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_loc_error(0, 0) = 0;  // global idx
  h_loc_error(0, 1) = 0;  // idx of dimension 0

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // "Mismatched elements (by indices):\n"
  // "  Index (0): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(0\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}

template <typename T>
void test_print_errors_2D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 3);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 3;  // global idx
  h_loc_error(0, 1) = 0;  // idx of dimension 0
  h_loc_error(0, 2) = 1;  // idx of dimension 1
  h_loc_error(1, 0) = 4;  // global idx
  h_loc_error(1, 1) = 1;  // idx of dimension 0
  h_loc_error(1, 2) = 1;  // idx of dimension 1

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // "Mismatched elements (by indices):\n"
  // "  Index (0, 1): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)\n"
  // "  Index (1, 1): actual 4.0000000000000000 vs expected 3.0000000000000000 (diff=1.0000000000000000)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(0, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))"
      "\n"
      R"(\s+Index \(1, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}

template <typename T>
void test_print_errors_3D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 4);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 9;   // global idx
  h_loc_error(0, 1) = 0;   // idx of dimension 0
  h_loc_error(0, 2) = 1;   // idx of dimension 1
  h_loc_error(0, 3) = 1;   // idx of dimension 2
  h_loc_error(1, 0) = 16;  // global idx
  h_loc_error(1, 1) = 1;   // idx of dimension 0
  h_loc_error(1, 2) = 1;   // idx of dimension 1
  h_loc_error(1, 3) = 2;   // idx of dimension 2

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // "Mismatched elements (by indices):\n"
  // "  Index (0, 1, 1): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)\n"
  // "  Index (1, 1, 2): actual 4.0000000000000000 vs expected 3.0000000000000000 (diff=1.0000000000000000)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(0, 1, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))"
      "\n"
      R"(\s+Index \(1, 1, 2\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}

template <typename T>
void test_print_errors_4D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 5);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 81;  // global idx
  h_loc_error(0, 1) = 0;   // idx of dimension 0
  h_loc_error(0, 2) = 1;   // idx of dimension 1
  h_loc_error(0, 3) = 1;   // idx of dimension 2
  h_loc_error(0, 4) = 3;   // idx of dimension 3
  h_loc_error(1, 0) = 64;  // global idx
  h_loc_error(1, 1) = 1;   // idx of dimension 0
  h_loc_error(1, 2) = 1;   // idx of dimension 1
  h_loc_error(1, 3) = 2;   // idx of dimension 2
  h_loc_error(1, 4) = 2;   // idx of dimension 3

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // [NOTE] Order reversed because of sorting by global idx
  // "Mismatched elements (by indices):\n"
  // "  Index (1, 1, 2, 2): actual 4.0000000000000000 vs expected 3.0000000000000000 (diff=1.0000000000000000)\n"
  // "  Index (0, 1, 1, 3): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(1, 1, 2, 2\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))"
      "\n"
      R"(\s+Index \(0, 1, 1, 3\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}

template <typename T>
void test_print_errors_5D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 6);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 201;  // global idx
  h_loc_error(0, 1) = 0;    // idx of dimension 0
  h_loc_error(0, 2) = 1;    // idx of dimension 1
  h_loc_error(0, 3) = 1;    // idx of dimension 2
  h_loc_error(0, 4) = 3;    // idx of dimension 3
  h_loc_error(0, 5) = 1;    // idx of dimension 4
  h_loc_error(1, 0) = 298;  // global idx
  h_loc_error(1, 1) = 1;    // idx of dimension 0
  h_loc_error(1, 2) = 1;    // idx of dimension 1
  h_loc_error(1, 3) = 2;    // idx of dimension 2
  h_loc_error(1, 4) = 2;    // idx of dimension 3
  h_loc_error(1, 5) = 1;    // idx of dimension 4

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // "Mismatched elements (by indices):\n"
  // "  Index (0, 1, 1, 3, 1): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)\n"
  // "  Index (1, 1, 2, 2, 1): actual 4.0000000000000000 vs expected 3.0000000000000000 (diff=1.0000000000000000)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(0, 1, 1, 3, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))"
      "\n"
      R"(\s+Index \(1, 1, 2, 2, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}

template <typename T>
void test_print_errors_6D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 7);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 561;  // global idx
  h_loc_error(0, 1) = 0;    // idx of dimension 0
  h_loc_error(0, 2) = 1;    // idx of dimension 1
  h_loc_error(0, 3) = 1;    // idx of dimension 2
  h_loc_error(0, 4) = 3;    // idx of dimension 3
  h_loc_error(0, 5) = 1;    // idx of dimension 4
  h_loc_error(0, 6) = 1;    // idx of dimension 5
  h_loc_error(1, 0) = 658;  // global idx
  h_loc_error(1, 1) = 1;    // idx of dimension 0
  h_loc_error(1, 2) = 1;    // idx of dimension 1
  h_loc_error(1, 3) = 2;    // idx of dimension 2
  h_loc_error(1, 4) = 2;    // idx of dimension 3
  h_loc_error(1, 5) = 1;    // idx of dimension 4
  h_loc_error(1, 6) = 1;    // idx of dimension 5

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // "Mismatched elements (by indices):\n"
  // "  Index (0, 1, 1, 3, 1, 1): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)\n"
  // "  Index (1, 1, 2, 2, 1, 1): actual 4.0000000000000000 vs expected 3.0000000000000000 (diff=1.0000000000000000)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(0, 1, 1, 3, 1, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))"
      "\n"
      R"(\s+Index \(1, 1, 2, 2, 1, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}

template <typename T>
void test_print_errors_7D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 8);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 2001;  // global idx
  h_loc_error(0, 1) = 0;     // idx of dimension 0
  h_loc_error(0, 2) = 1;     // idx of dimension 1
  h_loc_error(0, 3) = 1;     // idx of dimension 2
  h_loc_error(0, 4) = 3;     // idx of dimension 3
  h_loc_error(0, 5) = 1;     // idx of dimension 4
  h_loc_error(0, 6) = 1;     // idx of dimension 5
  h_loc_error(0, 7) = 1;     // idx of dimension 6
  h_loc_error(1, 0) = 3538;  // global idx
  h_loc_error(1, 1) = 1;     // idx of dimension 0
  h_loc_error(1, 2) = 1;     // idx of dimension 1
  h_loc_error(1, 3) = 2;     // idx of dimension 2
  h_loc_error(1, 4) = 2;     // idx of dimension 3
  h_loc_error(1, 5) = 1;     // idx of dimension 4
  h_loc_error(1, 6) = 1;     // idx of dimension 5
  h_loc_error(1, 7) = 2;     // idx of dimension 6

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // "Mismatched elements (by indices):\n"
  // "  Index (0, 1, 1, 3, 1, 1, 1): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)\n"
  // "  Index (1, 1, 2, 2, 1, 1, 2): actual 4.0000000000000000 vs expected 3.0000000000000000 (diff=1.0000000000000000)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(0, 1, 1, 3, 1, 1, 1\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))"
      "\n"
      R"(\s+Index \(1, 1, 2, 2, 1, 1, 2\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}

template <typename T>
void test_print_errors_8D_analytical() {
  using View1DType            = Kokkos::View<T*>;
  using CountViewType         = Kokkos::View<std::size_t**>;
  const std::size_t nb_errors = 2;
  View1DType a_error("ref_a_error", nb_errors),
      b_error("ref_b_error", nb_errors);
  CountViewType loc_error("ref_loc_error", nb_errors, 9);

  auto h_a_error   = Kokkos::create_mirror_view(a_error);
  auto h_b_error   = Kokkos::create_mirror_view(b_error);
  auto h_loc_error = Kokkos::create_mirror_view(loc_error);

  // Initialization and prepare reference at host
  h_a_error(0)      = 3.0;
  h_a_error(1)      = 3.0;
  h_b_error(0)      = 3.0 + 2.0 * (3.0 * 1.0e-5);
  h_b_error(1)      = 4.0;
  h_loc_error(0, 0) = 2001;  // global idx
  h_loc_error(0, 1) = 0;     // idx of dimension 0
  h_loc_error(0, 2) = 1;     // idx of dimension 1
  h_loc_error(0, 3) = 1;     // idx of dimension 2
  h_loc_error(0, 4) = 3;     // idx of dimension 3
  h_loc_error(0, 5) = 1;     // idx of dimension 4
  h_loc_error(0, 6) = 1;     // idx of dimension 5
  h_loc_error(0, 7) = 1;     // idx of dimension 6
  h_loc_error(0, 8) = 0;     // idx of dimension 7
  h_loc_error(1, 0) = 3538;  // global idx
  h_loc_error(1, 1) = 1;     // idx of dimension 0
  h_loc_error(1, 2) = 1;     // idx of dimension 1
  h_loc_error(1, 3) = 2;     // idx of dimension 2
  h_loc_error(1, 4) = 2;     // idx of dimension 3
  h_loc_error(1, 5) = 1;     // idx of dimension 4
  h_loc_error(1, 6) = 1;     // idx of dimension 5
  h_loc_error(1, 7) = 2;     // idx of dimension 6
  h_loc_error(1, 8) = 0;     // idx of dimension 7

  Kokkos::deep_copy(a_error, h_a_error);
  Kokkos::deep_copy(b_error, h_b_error);
  Kokkos::deep_copy(loc_error, h_loc_error);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(b_error, a_error, loc_error);

  // clang-format off
  // NOLINTBEGIN(*)
  // Example error message. The floating point numbers may vary.
  // "Mismatched elements (by indices):\n"
  // "  Index (0, 1, 1, 3, 1, 1, 1, 0): actual 3.0000599999999999 vs expected 3.0000000000000000 (diff=0.0000599999999999)\n"
  // "  Index (1, 1, 2, 2, 1, 1, 2, 0): actual 4.0000000000000000 vs expected 3.0000000000000000 (diff=1.0000000000000000)";
  // NOLINTEND(*)
  // clang-format on
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  // Regular expression pattern matching the expected format.
  // \d+\.\d+ matches one or more digits, a literal dot, and one or more digits.
  // \s+ matches one or more whitespace characters.
  // Updated regex pattern: Replace \d with [0-9]
  std::string pattern =
      R"(Mismatched elements \(by indices\):)"
      "\n"
      R"(\s+Index \(0, 1, 1, 3, 1, 1, 1, 0\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))"
      "\n"
      R"(\s+Index \(1, 1, 2, 2, 1, 1, 2, 0\): actual [0-9]+\.[0-9]+ vs expected [0-9]+\.[0-9]+ \(diff=[0-9]+\.[0-9]+\))";

  // Using GoogleMock's MatchesRegex matcher to compare the string with the
  // pattern.
  EXPECT_THAT(error_str, ::testing::MatchesRegex(pattern));
}
}  // namespace

TYPED_TEST_SUITE(TestSortErrors, float_types);
TYPED_TEST_SUITE(TestPrintErrors, float_types);

TYPED_TEST(TestSortErrors, View1D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_1D_analytical<float_type>();
}

TYPED_TEST(TestSortErrors, View2D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_2D_analytical<float_type>();
}

TYPED_TEST(TestSortErrors, View3D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_3D_analytical<float_type>();
}

TYPED_TEST(TestSortErrors, View4D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_4D_analytical<float_type>();
}

TYPED_TEST(TestSortErrors, View5D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_5D_analytical<float_type>();
}

TYPED_TEST(TestSortErrors, View6D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_6D_analytical<float_type>();
}

TYPED_TEST(TestSortErrors, View7D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_7D_analytical<float_type>();
}

TYPED_TEST(TestSortErrors, View8D) {
  using float_type = typename TestFixture::float_type;
  test_sort_errors_8D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View1D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_1D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View2D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_2D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View3D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_3D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View4D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_4D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View5D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_5D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View6D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_6D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View7D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_7D_analytical<float_type>();
}

TYPED_TEST(TestPrintErrors, View8D) {
  using float_type = typename TestFixture::float_type;
  test_print_errors_8D_analytical<float_type>();
}
