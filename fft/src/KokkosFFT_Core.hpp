// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CORE_HPP
#define KOKKOSFFT_CORE_HPP

namespace KokkosFFT {
void initialize();

[[nodiscard]] bool is_initialized() noexcept;
[[nodiscard]] bool is_finalized() noexcept;

void finalize();
}  // namespace KokkosFFT

#endif
