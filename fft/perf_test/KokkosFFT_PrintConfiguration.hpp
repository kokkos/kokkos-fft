//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef KOKKOSFFT_PRINT_CONFIGURATION_HPP
#define KOKKOSFFT_PRINT_CONFIGURATION_HPP

#include "KokkosFFT_config.h"
#include "KokkosFFT_TplsVersion.hpp"
#include <iostream>

namespace KokkosFFT {
namespace Impl {

inline void print_cufft_version_if_enabled(std::ostream& os) {
#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_CUFFT: " << cufft_version_string() << "\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_CUFFT: no\n";
#endif
}

inline void print_rocfft_version_if_enabled(std::ostream& os) {
#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ROCFFT: " << rocfft_version_string() << "\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ROCFFT: no\n";
#endif
}

inline void print_hipfft_version_if_enabled(std::ostream& os) {
#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: " << hipfft_version_string() << "\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: no\n";
#endif
}

inline void print_enabled_tpls(std::ostream& os) {
#ifdef KOKKOSFFT_ENABLE_TPL_FFTW
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_FFTW: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_FFTW: no\n";
#endif

  print_cufft_version_if_enabled(os);

#ifdef KOKKOSFFT_ENABLE_TPL_HIPFFT
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: no\n";
#endif

#ifdef KOKKOSFFT_ENABLE_TPL_ONEMKL
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ONEMKL: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ONEMKL: no\n";
#endif
}

inline void print_version(std::ostream& os) {
  // KOKKOSFFT_VERSION is used because MAJOR, MINOR and PATCH macros
  // are not available in FFT
  os << "  "
     << "KokkosFFT Version: " << KOKKOSFFT_VERSION_MAJOR << "."
     << KOKKOSFFT_VERSION_MINOR << "." << KOKKOSFFT_VERSION_PATCH << '\n';
}
}  // namespace Impl

inline void print_configuration(std::ostream& os) {
  Impl::print_version(os);

  os << "TPLs: \n";
  Impl::print_enabled_tpls(os);
}

}  // namespace KokkosFFT

#endif