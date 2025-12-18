// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_All2All.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestAll2All : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T, typename LayoutType>
void test_all2all_view2D(int rank, int nprocs) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;

  const std::size_t n0 = 16, n1 = 15;
  const std::size_t n0_local = ((n0 - 1) / nprocs) + 1;
  const std::size_t n1_local = ((n1 - 1) / nprocs) + 1;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_buffer = n0_local;
    n1_buffer = n1_local;
    n2_buffer = nprocs;
  } else {
    n0_buffer = nprocs;
    n1_buffer = n0_local;
    n2_buffer = n1_local;
  }

  View3DType send("send", n0_buffer, n1_buffer, n2_buffer),
      recv("recv", n0_buffer, n1_buffer, n2_buffer),
      ref("ref", n0_buffer, n1_buffer, n2_buffer);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);

  for (std::size_t i2 = 0; i2 < send.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < send.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < send.extent(0); i0++) {
        if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
          T value =
              static_cast<T>(rank * send.size() + i0 + i1 * send.extent(0) +
                             i2 * send.extent(0) * send.extent(1));
          T value_T =
              static_cast<T>(i2 * send.size() + i0 + i1 * send.extent(0) +
                             rank * send.extent(0) * send.extent(1));
          h_send(i0, i1, i2) = value;
          h_ref(i0, i1, i2)  = value_T;
        } else {
          T value =
              static_cast<T>(rank * send.size() + i2 + i1 * send.extent(2) +
                             i0 * send.extent(2) * send.extent(1));
          T value_T =
              static_cast<T>(i0 * send.size() + i2 + i1 * send.extent(2) +
                             rank * send.extent(2) * send.extent(1));
          h_send(i0, i1, i2) = value;
          h_ref(i0, i1, i2)  = value_T;
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  execution_space exec;
  KokkosFFT::Distributed::Impl::all2all(exec, send, recv, MPI_COMM_WORLD);
  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (std::size_t i2 = 0; i2 < send.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < send.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < send.extent(0); i0++) {
        auto diff = Kokkos::abs(h_recv(i0, i1, i2) - h_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_all2all_view2D_incorrect_proc_size(int nprocs) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;

  const std::size_t n0 = 16, n1 = 15;
  const std::size_t n0_local = ((n0 - 1) / nprocs) + 1;
  const std::size_t n1_local = ((n1 - 1) / nprocs) + 1;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  // Set incorrect proc size deliberately
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_buffer = n0_local;
    n1_buffer = n1_local;
    n2_buffer = nprocs + 1;
  } else {
    n0_buffer = nprocs + 1;
    n1_buffer = n0_local;
    n2_buffer = n1_local;
  }

  View3DType send("send", n0_buffer, n1_buffer, n2_buffer),
      recv("recv", n0_buffer, n1_buffer, n2_buffer);

  execution_space exec;
  EXPECT_THROW(
      KokkosFFT::Distributed::Impl::all2all(exec, send, recv, MPI_COMM_WORLD),
      std::runtime_error);
}

}  // namespace

TYPED_TEST_SUITE(TestAll2All, test_types);

TYPED_TEST(TestAll2All, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_all2all_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestAll2All, View2D_incorrect_proc_size) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_all2all_view2D_incorrect_proc_size<float_type, layout_type>(
      this->m_nprocs);
}
