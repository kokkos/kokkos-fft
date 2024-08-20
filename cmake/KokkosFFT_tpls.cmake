# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

function(get_tpls_list tpls_list backend_list)
    # In order to display Tpls in the Kokkos order
    # Device Parallel:
    # Host Parallel:
    # Host Serial:
    if(Kokkos_ENABLE_CUDA)
        list(APPEND tpls_list CUFFT)
        list(APPEND backend_list "Device Parallel")
    elseif(Kokkos_ENABLE_HIP)
        if(KokkosFFT_ENABLE_ROCFFT)
            list(APPEND tpls_list ROCFFT)
        else()
            list(APPEND tpls_list HIPFFT)
        endif()
        list(APPEND backend_list "Device Parallel")
    elseif(Kokkos_ENABLE_SYCL)
        list(APPEND tpls_list ONEMKL)
        list(APPEND backend_list "Device Parallel")
    elseif(Kokkos_ENABLE_OPENMP)
        list(APPEND tpls_list FFTW_OPENMP)
        list(APPEND backend_list "Host Parallel")
    elseif(Kokkos_ENABLE_THREADS)
        list(APPEND tpls_list FFTW_THREADS)
        list(APPEND backend_list "Host Parallel")
    elseif(Kokkos_ENABLE_SERIAL)
        list(APPEND tpls_list FFTW_SERIAL)
        list(APPEND backend_list "  Host Serial")
    endif()

    if(KokkosFFT_ENABLE_HOST_AND_DEVICE)
        # Check if TPL is already included
        list(FIND tpls_list "FFTW_OPENMP" idx_FFTW_OPENMP)
        list(FIND tpls_list "FFTW_THREADS" idx_FFTW_THREADS)
        list(FIND tpls_list "FFTW_SERIAL" idx_FFTW_SERIAL)
        if(Kokkos_ENABLE_OPENMP AND (${idx_FFTW_OPENMP} EQUAL -1))
            list(APPEND tpls_list FFTW_OPENMP)
            list(APPEND backend_list "Host Parallel")
        elseif(Kokkos_ENABLE_THREADS AND (${idx_FFTW_THREADS} EQUAL -1))
            list(APPEND tpls_list FFTW_THREADS)
            list(APPEND backend_list "Host Parallel")
        endif()

        if(Kokkos_ENABLE_SERIAL AND (${idx_FFTW_SERIAL} EQUAL -1))
            list(APPEND tpls_list FFTW_SERIAL)
            list(APPEND backend_list "  Host Serial")
        endif()
    endif()

    set(${tpls_list} ${${tpls_list}} PARENT_SCOPE)
    set(${backend_list} ${${backend_list}} PARENT_SCOPE)
endfunction()

