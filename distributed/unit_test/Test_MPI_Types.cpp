// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace {
using test_types = ::testing::Types<
    char, unsigned char, short, unsigned short, int, unsigned int, long,
    unsigned long, long long, unsigned long long, std::int8_t, std::uint8_t,
    std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t,
    std::uint64_t, std::size_t, std::ptrdiff_t, float, double,
    Kokkos::complex<float>, Kokkos::complex<double>>;

template <typename T>
struct TestMPIType : public ::testing::Test {
  using value_type = T;
};

template <typename T>
void test_mpi_data_type() {
  MPI_Datatype mpi_type = KokkosFFT::Distributed::Impl::mpi_datatype_v<T>;

  if constexpr (std::is_same_v<T, char>) {
    ASSERT_EQ(mpi_type, MPI_CHAR);
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    ASSERT_EQ(mpi_type, MPI_UNSIGNED_CHAR);
  } else if constexpr (std::is_same_v<T, short>) {
    ASSERT_EQ(mpi_type, MPI_SHORT);
  } else if constexpr (std::is_same_v<T, unsigned short>) {
    ASSERT_EQ(mpi_type, MPI_UNSIGNED_SHORT);
  } else if constexpr (std::is_same_v<T, int>) {
    ASSERT_EQ(mpi_type, MPI_INT32_T);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    ASSERT_EQ(mpi_type, MPI_UINT32_T);
  } else if constexpr (std::is_same_v<T, long>) {
    ASSERT_EQ(mpi_type, MPI_LONG);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    ASSERT_EQ(mpi_type, MPI_UNSIGNED_LONG);
  } else if constexpr (std::is_same_v<T, long long>) {
    ASSERT_EQ(mpi_type, MPI_LONG_LONG);
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    ASSERT_EQ(mpi_type, MPI_UNSIGNED_LONG_LONG);
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    ASSERT_EQ(mpi_type, MPI_INT8_T);
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    ASSERT_EQ(mpi_type, MPI_UINT8_T);
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    ASSERT_EQ(mpi_type, MPI_INT16_T);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    ASSERT_EQ(mpi_type, MPI_UINT16_T);
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    ASSERT_EQ(mpi_type, MPI_INT32_T);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    ASSERT_EQ(mpi_type, MPI_UINT32_T);
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    ASSERT_EQ(mpi_type, MPI_INT64_T);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    ASSERT_EQ(mpi_type, MPI_UINT64_T);
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == sizeof(unsigned int)) {
      ASSERT_EQ(mpi_type, MPI_UNSIGNED);
    } else if constexpr (sizeof(std::size_t) == sizeof(unsigned long)) {
      ASSERT_EQ(mpi_type, MPI_UNSIGNED_LONG);
    } else if constexpr (sizeof(std::size_t) == sizeof(unsigned long long)) {
      ASSERT_EQ(mpi_type, MPI_UNSIGNED_LONG_LONG);
    }
  } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == sizeof(int)) {
      ASSERT_EQ(mpi_type, MPI_INT);
    } else if constexpr (sizeof(std::ptrdiff_t) == sizeof(long)) {
      ASSERT_EQ(mpi_type, MPI_LONG);
    } else if constexpr (sizeof(std::ptrdiff_t) == sizeof(long long)) {
      ASSERT_EQ(mpi_type, MPI_LONG_LONG);
    }
  } else if constexpr (std::is_same_v<T, float>) {
    ASSERT_EQ(mpi_type, MPI_FLOAT);
  } else if constexpr (std::is_same_v<T, double>) {
    ASSERT_EQ(mpi_type, MPI_DOUBLE);
  } else if constexpr (std::is_same_v<T, Kokkos::complex<float>>) {
    ASSERT_EQ(mpi_type, MPI_COMPLEX);
  } else if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
    ASSERT_EQ(mpi_type, MPI_DOUBLE_COMPLEX);
  }
}

}  // namespace

TYPED_TEST_SUITE(TestMPIType, test_types);

TYPED_TEST(TestMPIType, test_convert_scalar_type) {
  using value_type = typename TestFixture::value_type;
  test_mpi_data_type<value_type>();
}
