// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include "KokkosFFT_Core.hpp"
#include "KokkosFFT_default_types.hpp"

namespace {
bool g_is_initialized = false;
bool g_is_finalized   = false;

bool kokkosfft_initialize_was_called() {
  return KokkosFFT::is_initialized() || KokkosFFT::is_finalized();
}
bool kokkosfft_finalize_was_called() { return KokkosFFT::is_finalized(); }

void initialize_internal() {
  KokkosFFT::Impl::initialize_host();
  KokkosFFT::Impl::initialize_device();
}

void finalize_internal() {
  KokkosFFT::Impl::finalize_host();
  KokkosFFT::Impl::finalize_device();
}
}  // namespace

[[nodiscard]] bool KokkosFFT::is_initialized() noexcept {
  return g_is_initialized;
}

[[nodiscard]] bool KokkosFFT::is_finalized() noexcept { return g_is_finalized; }

void KokkosFFT::initialize() {
  if (!(Kokkos::is_initialized() || Kokkos::is_finalized())) {
    Kokkos::abort(
        "Error: KokkosFFT::initialize() must not be called before initializing "
        "Kokkos.\n");
  }
  if (Kokkos::is_finalized()) {
    Kokkos::abort(
        "Error: KokkosFFT::initialize() must not be called after finalizing "
        "Kokkos.\n");
  }
  if (kokkosfft_initialize_was_called()) {
    Kokkos::abort(
        "Error: KokkosFFT::initialize() has already been called."
        " KokkosFFT can be initialized at most once.\n");
  }
  initialize_internal();
  g_is_initialized = true;
}

void KokkosFFT::finalize() {
  if (!(Kokkos::is_initialized() || Kokkos::is_finalized())) {
    Kokkos::abort(
        "Error: KokkosFFT::finalize() may only be called after Kokkos has been "
        "initialized.\n");
  }
  if (Kokkos::is_finalized()) {
    Kokkos::abort(
        "Error: KokkosFFT::finalize() must be called before finalizing "
        "Kokkos.\n");
  }

  if (!kokkosfft_initialize_was_called()) {
    Kokkos::abort(
        "Error: KokkosFFT::finalize() may only be called after KokkosFFT has "
        "been "
        "initialized.\n");
  }
  if (kokkosfft_finalize_was_called()) {
    Kokkos::abort("Error: KokkosFFT::finalize() has already been called.\n");
  }
  finalize_internal();
  g_is_initialized = false;
  g_is_finalized   = true;
}
