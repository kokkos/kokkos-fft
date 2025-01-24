# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(09_hasegawa_wakatani 09_hasegawa_wakatani.cpp)
target_link_libraries(09_hasegawa_wakatani PUBLIC KokkosFFT::fft)
