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
#ifndef KOKKOSFFT_TPLS_VERSIONS_HPP
#define KOKKOSFFT_TPLS_VERSIONS_HPP

#include "KokkosFFT_config.h"
#include <sstream>
#include <iostream>

#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
#include "cufft.h"
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#include <rocfft/rocfft.h>
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
#include <hipfft/hipfft.h>
#endif

namespace KokkosFFT {
#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
inline std::string cufft_version_string() {
  // Print version
  std::stringstream ss;

  ss << CUFFT_VER_MAJOR << "." << CUFFT_VER_MINOR << "." << CUFFT_VER_PATCH;

  return ss.str();
}
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
inline std::string rocfft_version_string() {
  // Print version
  std::stringstream ss;
  constexpr std::size_t len = 50;
  char version_string[len];
  rocfft_get_version_string(version_string, len);

  ss << version_string;

  return ss.str();
}
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
inline std::string hipfft_version_string() {
  // Print version
  std::stringstream ss;

  ss << hipfftVersionMajor << "." << hipfftVersionMinor << "."
     << hipfftVersionPatch;

  return ss.str();
}
#endif

}  // namespace KokkosFFT
#endif