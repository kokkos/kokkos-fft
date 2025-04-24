// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CONCEPTS_HPP
#define KOKKOSFFT_CONCEPTS_HPP

#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Impl {
template <class>
struct is_kokkos_array : public std::false_type {};

template <class T, std::size_t N>
struct is_kokkos_array<Kokkos::Array<T, N>> : public std::true_type {};

template <class T, std::size_t N>
struct is_kokkos_array<const Kokkos::Array<T, N>> : public std::true_type {};
}  // namespace Impl

template <typename T>
concept KokkosArray = Impl::is_kokkos_array<T>::value;

template <typename T>
concept KokkosLayout = Kokkos::is_array_layout_v<T>;

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename ExecutionSpace, typename ViewType>
concept KokkosViewAccessible = (bool)Kokkos::SpaceAccessibility<
    ExecutionSpace, typename ViewType::memory_space>::accessible;
}  // namespace KokkosFFT

#endif
