// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace {
using test_types =
    ::testing::Types<int, std::size_t, float, double, Kokkos::complex<float>,
                     Kokkos::complex<double>>;

template <typename T>
struct TestMPIType : public ::testing::Test {
  using value_type = T;
};

template <typename T>
void test_mpi_data_type() {
  MPI_Datatype mpi_type = KokkosFFT::Distributed::Impl::MPIDataType<T>::type();

  if constexpr (std::is_same_v<T, int>) {
    ASSERT_EQ(mpi_type, MPI_INT32_T);
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    ASSERT_EQ(mpi_type, MPI_UINT64_T);
  } else if constexpr (std::is_same_v<T, float>) {
    ASSERT_EQ(mpi_type, MPI_FLOAT);
  } else if constexpr (std::is_same_v<T, double>) {
    ASSERT_EQ(mpi_type, MPI_DOUBLE);
  } else if constexpr (std::is_same_v<T, Kokkos::complex<float>>) {
    ASSERT_EQ(mpi_type, MPI_CXX_FLOAT_COMPLEX);
  } else if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
    ASSERT_EQ(mpi_type, MPI_CXX_DOUBLE_COMPLEX);
  }
}

}  // namespace

TYPED_TEST_SUITE(TestMPIType, test_types);

TYPED_TEST(TestMPIType, test_convert_scalar_type) {
  using value_type = typename TestFixture::value_type;
  test_mpi_data_type<value_type>();
}
