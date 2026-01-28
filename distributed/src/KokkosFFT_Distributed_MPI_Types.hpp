// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_MPI_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_TYPES_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ValueType>
auto mpi_datatype() -> MPI_Datatype {
  using T = std::decay_t<ValueType>;

  if constexpr (std::is_same_v<T, char>) {
    return MPI_CHAR;
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same_v<T, short>) {
    return MPI_SHORT;
  } else if constexpr (std::is_same_v<T, unsigned short>) {
    return MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same_v<T, int>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<T, long>) {
    return MPI_LONG;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same_v<T, long long>) {
    return MPI_LONG_LONG;
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return MPI_INT8_T;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return MPI_UINT8_T;
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return MPI_INT16_T;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return MPI_UINT16_T;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return MPI_INT64_T;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return MPI_UINT64_T;
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == sizeof(unsigned int))
      return MPI_UNSIGNED;
    else if constexpr (sizeof(std::size_t) == sizeof(unsigned long))
      return MPI_UNSIGNED_LONG;
    else if constexpr (sizeof(std::size_t) == sizeof(unsigned long long))
      return MPI_UNSIGNED_LONG_LONG;
    else {
      static_assert(std::is_void_v<T>,
                    "Unsupported std::size_t size for MPI mapping");
      return MPI_UNSIGNED;  // unreachable
    }
  } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == sizeof(int))
      return MPI_INT;
    else if constexpr (sizeof(std::ptrdiff_t) == sizeof(long))
      return MPI_LONG;
    else if constexpr (sizeof(std::ptrdiff_t) == sizeof(long long))
      return MPI_LONG_LONG;
    else {
      static_assert(std::is_void_v<T>,
                    "Unsupported std::ptrdiff_t size for MPI mapping");
      return MPI_INT;  // unreachable
    }
  } else if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, long double>) {
    return MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same_v<T, Kokkos::complex<float>>) {
    return MPI_COMPLEX;
  } else if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
    return MPI_DOUBLE_COMPLEX;
  } else {
    static_assert(std::is_void_v<T>,
                  "Unsupported type for MPI datatype mapping");
    return MPI_CHAR;  // unreachable
  }
}

template <typename ValueType>
inline MPI_Datatype mpi_datatype_v = mpi_datatype<ValueType>();

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
