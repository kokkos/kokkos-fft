# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

if(KOKKOSFFT_HAS_DEFAULT_BACKEND)
  add_subdirectory(01_1DFFT)
  add_subdirectory(02_2DFFT)
  add_subdirectory(03_NDFFT)
  add_subdirectory(04_batchedFFT)
  add_subdirectory(06_1DFFT_reuse_plans)
  add_subdirectory(07_unmanaged_views)
  add_subdirectory(08_inplace_FFT)
  add_subdirectory(09_derivative)
  add_subdirectory(10_HasegawaWakatani)
endif()
add_subdirectory(05_1DFFT_HOST_DEVICE)
