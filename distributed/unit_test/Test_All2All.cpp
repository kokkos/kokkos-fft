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
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
    m_npx = std::sqrt(m_nprocs);
  }
};

/// \brief Test all2all communication for 2D Views
/// For 8x8 matrix.
/// Before MPI
/// Rank 0
/// [[A00, A01, A02, A03, A04, A05, A06, A07],
///  [A10, A11, A12, A13, A14, A15, A16, A17]]
/// Rank 1
/// [[A20, A21, A22, A23, A24, A25, A26, A27],
///  [A30, A31, A32, A33, A34, A35, A36, A37]]
/// Rank 2
/// [[A40, A41, A42, A43, A44, A45, A46, A47],
///  [A50, A51, A52, A53, A54, A55, A56, A57]]
/// Rank 3
/// [[A60, A61, A62, A63, A64, A65, A66, A67],
///  [A70, A71, A72, A73, A74, A75, A76, A77]]
///
/// Send buffer (Each rank has 4x2x2 matrix)
/// Rank 0
/// [[[A00, A01], [A10, A11]],
///  [[A02, A03], [A12, A13]],
///  [[A04, A05], [A14, A15]],
///  [[A06, A07], [A16, A17]]]
/// Rank 1
/// [[[A20, A21], [A30, A31]],
///  [[A22, A23], [A32, A33]],
///  [[A24, A25], [A34, A35]],
///  [[A26, A27], [A36, A37]]]
/// Rank 2
/// [[[A40, A41], [A50, A51]],
///  [[A42, A43], [A52, A53]],
///  [[A44, A45], [A54, A55]],
///  [[A46, A47], [A56, A57]]]
/// Rank 3
/// [[[A60, A61], [A70, A71]],
///  [[A62, A63], [A72, A73]],
///  [[A64, A65], [A74, A75]],
///  [[A66, A67], [A76, A77]]]
///
/// Receive buffer (Each rank has 2x4x2 matrix)
/// Rank 0
/// [[[A00, A01], [A10, A11]],
///  [[A20, A21], [A30, A31]],
///  [[A40, A41], [A50, A51]],
///  [[A60, A61], [A70, A71]]]
/// Rank 1
/// [[[A02, A03], [A12, A13]],
///  [[A22, A23], [A32, A33]],
///  [[A42, A43], [A52, A53]],
///  [[A62, A63], [A72, A73]]]
/// Rank 2
/// [[[A04, A05], [A14, A15]],
///  [[A24, A25], [A34, A35]],
///  [[A44, A45], [A54, A55]],
///  [[A64, A65], [A74, A75]]]
/// Rank 3
/// [[[A06, A07], [A16, A17]],
///  [[A26, A27], [A36, A37]],
///  [[A46, A47], [A56, A57]],
///  [[A66, A67], [A76, A77]]]
///
/// After all2all communication with transpose
/// Rank 0
/// [[A00, A10, A20, A30, A40, A50, A60, A70],
///  [A01, A11, A21, A31, A41, A51, A61, A71]]
/// Rank 1
/// [[A02, A12, A22, A32, A42, A52, A62, A72],
///  [A03, A13, A23, A33, A43, A53, A63, A73]]
/// Rank 2
/// [[A04, A14, A24, A34, A44, A54, A64, A74],
///  [A05, A15, A25, A35, A45, A55, A65, A75]]
/// Rank 3
/// [[A06, A16, A26, A36, A46, A56, A66, A76],
///  [A07, A17, A27, A37, A47, A57, A67, A77]]
/// \tparam T Data type
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] rank MPI rank
/// \param[in] nprocs Number of MPI ranks
/// \param[in] use_tpl_wrapper Whether to use template wrapper
template <typename T, typename LayoutType>
void test_all2all_view2D(int rank, int nprocs, bool use_tpl_wrapper) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
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
          float_type value = static_cast<float_type>(
              rank * send.size() + i0 + i1 * send.extent(0) +
              i2 * send.extent(0) * send.extent(1));
          float_type value_T = static_cast<float_type>(
              i2 * send.size() + i0 + i1 * send.extent(0) +
              rank * send.extent(0) * send.extent(1));
          if constexpr (KokkosFFT::Impl::is_complex_v<T>) {
            h_send(i0, i1, i2) = T(value, value);
            h_ref(i0, i1, i2)  = T(value_T, value_T);
          } else {
            h_send(i0, i1, i2) = value;
            h_ref(i0, i1, i2)  = value_T;
          }
        } else {
          float_type value = static_cast<float_type>(
              rank * send.size() + i2 + i1 * send.extent(2) +
              i0 * send.extent(2) * send.extent(1));
          float_type value_T = static_cast<float_type>(
              i0 * send.size() + i2 + i1 * send.extent(2) +
              rank * send.extent(2) * send.extent(1));
          if constexpr (KokkosFFT::Impl::is_complex_v<T>) {
            h_send(i0, i1, i2) = T(value, value);
            h_ref(i0, i1, i2)  = T(value_T, value_T);
          } else {
            h_send(i0, i1, i2) = value;
            h_ref(i0, i1, i2)  = value_T;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  execution_space exec;
  if (use_tpl_wrapper) {
    KokkosFFT::Distributed::Impl::TplComm<execution_space> comm(MPI_COMM_WORLD,
                                                                exec);
    KokkosFFT::Distributed::Impl::all2all(send, recv, comm);
  } else {
    KokkosFFT::Distributed::Impl::all2all(send, recv, MPI_COMM_WORLD);
  }
  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  auto epsilon = std::numeric_limits<float_type>::epsilon() * 100;
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
void test_all2all_view2D_incorrect_proc_size(int nprocs, bool use_tpl_wrapper) {
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
  if (use_tpl_wrapper) {
    KokkosFFT::Distributed::Impl::TplComm<execution_space> comm(MPI_COMM_WORLD,
                                                                exec);
    EXPECT_THROW(KokkosFFT::Distributed::Impl::all2all(send, recv, comm),
                 std::runtime_error);
  } else {
    EXPECT_THROW(
        KokkosFFT::Distributed::Impl::all2all(send, recv, MPI_COMM_WORLD),
        std::runtime_error);
  }
}

/// \brief Test all2all communication for 2D Views
/// For 8x8 matrix.
/// Before MPI
/// Rank 0
/// [[A00, A01, A02, A03],
///  [A10, A11, A12, A13],
///  [A20, A21, A22, A23],
///  [A30, A31, A32, A33]]
/// Rank 1
/// [[A04, A05, A06, A07],
///  [A14, A15, A16, A17],
///  [A24, A25, A26, A27],
///  [A34, A35, A36, A37]]
/// Rank 2
/// [[A40, A41, A42, A43],
///  [A50, A51, A52, A53],
///  [A60, A61, A62, A63],
///  [A70, A71, A72, A73]]
/// Rank 3
/// [[A44, A45, A46, A47],
///  [A54, A55, A56, A57],
///  [A64, A65, A66, A67],
///  [A74, A75, A76, A77]]
///
/// Send buffer (Each rank has 2x4x2 matrix)
/// Rank 0 <-> Rank 2
/// [[[A00, A01], [A10, A11], [A20, A21], [A30, A31]],
///  [[A02, A03], [A12, A13], [A22, A23], [A32, A33]]]
/// Rank 1 <-> Rank 3
/// [[[A04, A05], [A14, A15], [A24, A25], [A34, A35]],
///  [[A06, A07], [A16, A17], [A26, A27], [A36, A37]]]
/// Rank 2 <-> Rank 0
/// [[[A40, A41], [A50, A51], [A60, A61], [A70, A71]],
///  [[A42, A43], [A52, A53], [A62, A63], [A72, A73]]]
/// Rank 3 <-> Rank 1
/// [[[A44, A45], [A54, A55], [A64, A65], [A74, A75]],
///  [[A46, A47], [A56, A57], [A66, A67], [A76, A77]]]
///
/// Receive buffer (Each rank has 2x4x2 matrix)
/// Rank 0 <-> Rank 2
/// [[[A00, A01], [A10, A11], [A20, A21], [A30, A31]],
///  [[A40, A41], [A50, A51], [A60, A61], [A70, A71]]]
/// Rank 1 <-> Rank 3
/// [[[A04, A05], [A14, A15], [A24, A25], [A34, A35]],
///  [[A44, A45], [A54, A55], [A64, A65], [A74, A75]]]
/// Rank 2 <-> Rank 0
/// [[[A02, A03], [A12, A13], [A22, A23], [A32, A33]],
///  [[A42, A43], [A52, A53], [A62, A63], [A72, A73]]]
/// Rank 3 <-> Rank 1
/// [[[A06, A07], [A16, A17], [A26, A27], [A36, A37]],
///  [[A46, A47], [A56, A57], [A66, A67], [A76, A77]]]
///
/// After all2all communication with transpose
/// Rank 0
/// [[[A00, A01], [A10, A11], [A20, A21], [A30, A31]],
///  [[A40, A41], [A50, A51], [A60, A61], [A70, A71]]]
/// Rank 1
/// [[[A04, A05], [A14, A15], [A24, A25], [A34, A35]],
///  [[A44, A45], [A54, A55], [A64, A65], [A74, A75]]]
/// Rank 2
/// [[[A02, A03], [A12, A13], [A22, A23], [A32, A33]],
///  [[A42, A43], [A52, A53], [A62, A63], [A72, A73]]]
/// Rank 3
/// [[[A06, A07], [A16, A17], [A26, A27], [A36, A37]],
///  [[A46, A47], [A56, A57], [A66, A67], [A76, A77]]]
/// \tparam T Data type
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] rank MPI rank
/// \param[in] nprocs Number of MPI ranks
/// \param[in] use_tpl_wrapper Whether to use template wrapper
template <typename T, typename LayoutType>
void test_all2all_view2D_row(int rank, int npx, int npy, bool use_tpl_wrapper) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;

  const std::size_t n0 = 8, n1 = 8;
  const std::size_t n0_local = n0 / npx;
  const std::size_t n1_local = n1 / (npx * npy);
  int rx = rank / npy, ry = rank % npy;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_buffer = n0_local;
    n1_buffer = n1_local;
    n2_buffer = npx;
  } else {
    n0_buffer = npx;
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
          float_type value = static_cast<float_type>(
              (ry * (n1 / npy) + i1) + (rx * (n0 / npx) + i0) * n1 +
              i2 * send.extent(2));
          float_type value_T = static_cast<float_type>(
              i1 + rx * send.extent(1) + ry * (n1 / npy) + i0 * n1 +
              i2 * (n0 / npx) * n1);
          if constexpr (KokkosFFT::Impl::is_complex_v<T>) {
            h_send(i0, i1, i2) = T(value, value);
            h_ref(i0, i1, i2)  = T(value_T, value_T);
          } else {
            h_send(i0, i1, i2) = value;
            h_ref(i0, i1, i2)  = value_T;
          }
        } else {
          float_type value = static_cast<float_type>(
              (ry * (n1 / npy) + i2) + (rx * (n0 / npx) + i1) * n1 +
              i0 * send.extent(0));
          float_type value_T = static_cast<float_type>(
              i2 + rx * send.extent(0) + ry * (n1 / npy) + i1 * n1 +
              i0 * (n0 / npx) * n1);
          if constexpr (KokkosFFT::Impl::is_complex_v<T>) {
            h_send(i0, i1, i2) = T(value, value);
            h_ref(i0, i1, i2)  = T(value_T, value_T);
          } else {
            h_send(i0, i1, i2) = value;
            h_ref(i0, i1, i2)  = value_T;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  // Create a cartesian communicator
  int dims[2]    = {npx, npy};
  int periods[2] = {1, 1};
  MPI_Comm cart_comm;
  ::MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  ::MPI_Comm row_comm;

  int remain_dims[2];

  // keep Y‐axis for row_comm (all procs with same px)
  remain_dims[0] = 1;
  remain_dims[1] = 0;
  ::MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

  execution_space exec;
  if (use_tpl_wrapper) {
    KokkosFFT::Distributed::Impl::TplComm<execution_space> comm(row_comm, exec);
    KokkosFFT::Distributed::Impl::all2all(send, recv, comm);
  } else {
    KokkosFFT::Distributed::Impl::all2all(send, recv, row_comm);
  }

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  auto epsilon = std::numeric_limits<float_type>::epsilon() * 100;
  for (std::size_t i2 = 0; i2 < send.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < send.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < send.extent(0); i0++) {
        auto diff = Kokkos::abs(h_recv(i0, i1, i2) - h_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }

  ::MPI_Comm_free(&row_comm);
  ::MPI_Comm_free(&cart_comm);
}

/// \brief Test all2all communication for 2D Views
/// For 8x8 matrix.
/// Before MPI
/// Rank 0
/// [[A00, A01, A02, A03],
///  [A10, A11, A12, A13],
///  [A20, A21, A22, A23],
///  [A30, A31, A32, A33]]
/// Rank 1
/// [[A04, A05, A06, A07],
///  [A14, A15, A16, A17],
///  [A24, A25, A26, A27],
///  [A34, A35, A36, A37]]
/// Rank 2
/// [[A40, A41, A42, A43],
///  [A50, A51, A52, A53],
///  [A60, A61, A62, A63],
///  [A70, A71, A72, A73]]
/// Rank 3
/// [[A44, A45, A46, A47],
///  [A54, A55, A56, A57],
///  [A64, A65, A66, A67],
///  [A74, A75, A76, A77]]
///
/// Send buffer (Each rank has 2x2x4 matrix)
/// Rank 0 <-> Rank 1
/// [[[A00, A01, A02, A03],
///   [A10, A11, A12, A13]],
///  [[A20, A21, A22, A23],
///   [A30, A31, A32, A33]]]
/// Rank 1 <-> Rank 0
/// [[[A04, A05, A06, A07],
///   [A14, A15, A16, A17]],
///  [[A24, A25, A26, A27],
///   [A34, A35, A36, A37]]]
/// Rank 2 <-> Rank 3
/// [[[A40, A41, A42, A43],
///   [A50, A51, A52, A53]],
///  [[A60, A61, A62, A63],
///   [A70, A71, A72, A73]]]
/// Rank 3 <-> Rank 2
/// [[[A44, A45, A46, A47],
///   [A54, A55, A56, A57]],
///  [[A64, A65, A66, A67],
///   [A74, A75, A76, A77]]]
///
/// Receive buffer (Each rank has 2x2x4 matrix)
/// Rank 0 <-> Rank 1
/// [[[A00, A01, A02, A03],
///   [A10, A11, A12, A13]],
///  [[A04, A05, A06, A07],
///   [A14, A15, A16, A17]]]
/// Rank 1 <-> Rank 0
/// [[[A20, A21, A22, A23],
///   [A30, A31, A32, A33]],
///  [[A24, A25, A26, A27],
///   [A34, A35, A36, A37]]]
/// Rank 2 <-> Rank 3
/// [[[A40, A41, A42, A43],
///   [A50, A51, A52, A53]],
///  [[A44, A45, A46, A47],
///   [A54, A55, A56, A57]]]
/// Rank 3 <-> Rank 2
/// [[[A60, A61, A62, A63],
///   [A70, A71, A72, A73]],
///  [[A64, A65, A66, A67],
///   [A74, A75, A76, A77]]]
///
/// After all2all communication with transpose
/// Rank 0
/// [[A00, A01, A02, A03, A04, A05, A06, A07],
///  [A10, A11, A12, A13, A14, A15, A16, A17]]
/// Rank 1
/// [[A40, A41, A42, A43, A44, A45, A46, A47],
///  [A50, A51, A52, A53, A54, A55, A56, A57]]
/// Rank 2
/// [[A20, A21, A22, A23, A24, A25, A26, A27],
///  [A30, A31, A32, A33, A34, A35, A36, A37]]
/// Rank 3
/// [[A60, A61, A62, A63, A64, A65, A66, A67],
///  [A70, A71, A72, A73, A74, A75, A76, A77]]
/// \tparam T Data type
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] rank MPI rank
/// \param[in] nprocs Number of MPI ranks
/// \param[in] use_tpl_wrapper Whether to use template wrapper
template <typename T, typename LayoutType>
void test_all2all_view2D_col(int rank, int npx, int npy, bool use_tpl_wrapper) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;

  const std::size_t n0 = 8, n1 = 8;
  const std::size_t n0_local = n0 / (npx * npy);
  const std::size_t n1_local = ((n1 - 1) / npy) + 1;
  int rx = rank / npy, ry = rank % npy;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_buffer = n0_local;
    n1_buffer = n1_local;
    n2_buffer = npy;
  } else {
    n0_buffer = npy;
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
          float_type value = static_cast<float_type>(
              (ry * send.extent(1) + i1) + (rx * (n0 / npx) + i0) * n1 +
              i2 * send.extent(2) * n1);
          float_type value_T =
              static_cast<float_type>(i1 + (i0 + ry * send.extent(0)) * n1 +
                                      (i2 + rx * n1) * (n0 / npx));
          if constexpr (KokkosFFT::Impl::is_complex_v<T>) {
            h_send(i0, i1, i2) = T(value, value);
            h_ref(i0, i1, i2)  = T(value_T, value_T);
          } else {
            h_send(i0, i1, i2) = value;
            h_ref(i0, i1, i2)  = value_T;
          }
        } else {
          float_type value = static_cast<float_type>(
              (ry * send.extent(2) + i2) + (rx * (n0 / npx) + i1) * n1 +
              i0 * send.extent(0) * n1);
          float_type value_T =
              static_cast<float_type>(i2 + (i1 + ry * send.extent(0)) * n1 +
                                      (i0 + rx * n1) * (n0 / npx));
          if constexpr (KokkosFFT::Impl::is_complex_v<T>) {
            h_send(i0, i1, i2) = T(value, value);
            h_ref(i0, i1, i2)  = T(value_T, value_T);
          } else {
            h_send(i0, i1, i2) = value;
            h_ref(i0, i1, i2)  = value_T;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  // Create a cartesian communicator
  int dims[2]    = {npx, npy};
  int periods[2] = {1, 1};
  MPI_Comm cart_comm;
  ::MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  ::MPI_Comm col_comm;

  int remain_dims[2];

  // keep X‐axis for col_comm (all procs with same py)
  remain_dims[0] = 0;
  remain_dims[1] = 1;
  ::MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

  execution_space exec;
  if (use_tpl_wrapper) {
    KokkosFFT::Distributed::Impl::TplComm<execution_space> comm(col_comm, exec);
    KokkosFFT::Distributed::Impl::all2all(send, recv, comm);
  } else {
    KokkosFFT::Distributed::Impl::all2all(send, recv, col_comm);
  }

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  auto epsilon = std::numeric_limits<float_type>::epsilon() * 100;
  for (std::size_t i2 = 0; i2 < send.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < send.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < send.extent(0); i0++) {
        auto diff = Kokkos::abs(h_recv(i0, i1, i2) - h_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }

  ::MPI_Comm_free(&col_comm);
  ::MPI_Comm_free(&cart_comm);
}

}  // namespace

TYPED_TEST_SUITE(TestAll2All, test_types);

TYPED_TEST(TestAll2All, RView2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_all2all_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs,
                                               false);
}

TYPED_TEST(TestAll2All, CView2D) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  test_all2all_view2D<complex_type, layout_type>(this->m_rank, this->m_nprocs,
                                                 false);
}

TYPED_TEST(TestAll2All, RView2D_with_wrapper) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_all2all_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs,
                                               true);
}

TYPED_TEST(TestAll2All, CView2D_with_wrapper) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  test_all2all_view2D<complex_type, layout_type>(this->m_rank, this->m_nprocs,
                                                 true);
}

TYPED_TEST(TestAll2All, View2D_incorrect_proc_size) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_all2all_view2D_incorrect_proc_size<float_type, layout_type>(
      this->m_nprocs, false);
}

TYPED_TEST(TestAll2All, View2D_incorrect_proc_size_with_wrapper) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_all2all_view2D_incorrect_proc_size<float_type, layout_type>(
      this->m_nprocs, true);
}

TYPED_TEST(TestAll2All, RView2D_row) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_row<float_type, layout_type>(this->m_rank, this->m_npx,
                                                   this->m_npx, false);
}

TYPED_TEST(TestAll2All, CView2D_row) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_row<complex_type, layout_type>(this->m_rank, this->m_npx,
                                                     this->m_npx, false);
}

TYPED_TEST(TestAll2All, RView2D_row_with_wrapper) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_row<float_type, layout_type>(this->m_rank, this->m_npx,
                                                   this->m_npx, true);
}

TYPED_TEST(TestAll2All, CView2D_row_with_wrapper) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_row<complex_type, layout_type>(this->m_rank, this->m_npx,
                                                     this->m_npx, true);
}

TYPED_TEST(TestAll2All, RView2D_col) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_col<float_type, layout_type>(this->m_rank, this->m_npx,
                                                   this->m_npx, false);
}

TYPED_TEST(TestAll2All, CView2D_col) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_col<complex_type, layout_type>(this->m_rank, this->m_npx,
                                                     this->m_npx, false);
}

TYPED_TEST(TestAll2All, RView2D_col_with_wrapper) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_col<float_type, layout_type>(this->m_rank, this->m_npx,
                                                   this->m_npx, true);
}

TYPED_TEST(TestAll2All, CView2D_col_with_wrapper) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_all2all_view2D_col<complex_type, layout_type>(this->m_rank, this->m_npx,
                                                     this->m_npx, true);
}
