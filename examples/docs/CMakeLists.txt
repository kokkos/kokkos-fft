# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# Standard FFT examples
kokkosfft_add_example(docs_fft SOURCES standard/docs_fft.cpp)

kokkosfft_add_example(docs_ifft SOURCES standard/docs_ifft.cpp)

kokkosfft_add_example(docs_fft2 SOURCES standard/docs_fft2.cpp)

kokkosfft_add_example(docs_ifft2 SOURCES standard/docs_ifft2.cpp)

kokkosfft_add_example(docs_fftn SOURCES standard/docs_fftn.cpp)

kokkosfft_add_example(docs_ifftn SOURCES standard/docs_ifftn.cpp)

# Real FFT examples
kokkosfft_add_example(docs_rfft SOURCES real/docs_rfft.cpp)

kokkosfft_add_example(docs_irfft SOURCES real/docs_irfft.cpp)

kokkosfft_add_example(docs_rfft2 SOURCES real/docs_rfft2.cpp)

kokkosfft_add_example(docs_irfft2 SOURCES real/docs_irfft2.cpp)

kokkosfft_add_example(docs_rfftn SOURCES real/docs_rfftn.cpp)

kokkosfft_add_example(docs_irfftn SOURCES real/docs_irfftn.cpp)

# Hermitian FFT examples
kokkosfft_add_example(docs_hfft SOURCES hermitian/docs_hfft.cpp)

kokkosfft_add_example(docs_ihfft SOURCES hermitian/docs_ihfft.cpp)

# Helper routines
kokkosfft_add_example(docs_fftfreq SOURCES helper/docs_fftfreq.cpp)

kokkosfft_add_example(docs_rfftfreq SOURCES helper/docs_rfftfreq.cpp)

kokkosfft_add_example(docs_fftshift SOURCES helper/docs_fftshift.cpp)

kokkosfft_add_example(docs_ifftshift SOURCES helper/docs_ifftshift.cpp)

# Plan and execute examples
kokkosfft_add_example(docs_plan_execute SOURCES plan/docs_plan_execute.cpp)
