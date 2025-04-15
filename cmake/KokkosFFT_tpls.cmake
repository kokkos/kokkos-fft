# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# \brief Get the list of required FFT tpls based on the
#        CMake configuration
#        Display Tpls in the Kokkos order
#        Device Parallel:
#        Host Parallel:
#        Host Serial:
# \param tpls_list[inout] A list of tpls that are needed
# \param backend_list[inout] A list of backends corresponding to tpls
#        backend_list is used to display
# \param default_backend[inout] Whether default backend is available or not
function(get_tpls_list tpls_list backend_list default_backend)
  set(KOKKOSFFT_HAS_DEVICE FALSE)
  set(has_default_backend TRUE)
  if(Kokkos_ENABLE_CUDA)
    set(KOKKOSFFT_HAS_DEVICE TRUE)
    if(KokkosFFT_ENABLE_CUFFT)
      list(APPEND tpls_list CUFFT)
      list(APPEND backend_list "Device Parallel")
    else()
      set(has_default_backend FALSE)
    endif()
  elseif(Kokkos_ENABLE_HIP)
    set(KOKKOSFFT_HAS_DEVICE TRUE)
    if(KokkosFFT_ENABLE_ROCFFT)
      list(APPEND tpls_list ROCFFT)
      list(APPEND backend_list "Device Parallel")
    elseif(KokkosFFT_ENABLE_HIPFFT)
      list(APPEND tpls_list HIPFFT)
      list(APPEND backend_list "Device Parallel")
    else()
      set(has_default_backend FALSE)
    endif()
  elseif(Kokkos_ENABLE_SYCL)
    set(KOKKOSFFT_HAS_DEVICE TRUE)
    if(KokkosFFT_ENABLE_ONEMKL)
      list(APPEND tpls_list ONEMKL)
      list(APPEND backend_list "Device Parallel")
    else()
      set(has_default_backend FALSE)
    endif()
  elseif(Kokkos_ENABLE_OPENMP AND KokkosFFT_ENABLE_FFTW)
    list(APPEND tpls_list FFTW_OPENMP)
    list(APPEND backend_list "Host Parallel")
  elseif(Kokkos_ENABLE_THREADS AND KokkosFFT_ENABLE_FFTW)
    list(APPEND tpls_list FFTW_THREADS)
    list(APPEND backend_list "Host Parallel")
  elseif(Kokkos_ENABLE_SERIAL AND KokkosFFT_ENABLE_FFTW)
    list(APPEND tpls_list FFTW_SERIAL)
    list(APPEND backend_list "Host Serial")
  endif()
  if(KokkosFFT_ENABLE_FFTW)
    # Check if TPL is already included
    if(Kokkos_ENABLE_OPENMP AND NOT ("FFTW_OPENMP" IN_LIST tpls_list))
      list(APPEND tpls_list FFTW_OPENMP)
      list(APPEND backend_list "Host Parallel")
    elseif(Kokkos_ENABLE_THREADS AND NOT ("FFTW_THREADS" IN_LIST tpls_list))
      list(APPEND tpls_list FFTW_THREADS)
      list(APPEND backend_list "Host Parallel")
    endif()
    if(Kokkos_ENABLE_SERIAL AND NOT ("FFTW_SERIAL" IN_LIST tpls_list))
      list(APPEND tpls_list FFTW_SERIAL)
      list(APPEND backend_list "Host Serial")
    endif()
  endif()
  # Enable Serial if Device is not used
  if(NOT ${KOKKOSFFT_HAS_DEVICE} AND (Kokkos_ENABLE_SERIAL AND NOT ("FFTW_SERIAL" IN_LIST tpls_list)))
    list(APPEND tpls_list FFTW_SERIAL)
    list(APPEND backend_list "Host Serial")
  endif()
  # Get the size of the lists
  list(LENGTH tpls_list tpls_list_len)
  list(LENGTH backend_list backend_list_len)
  if(tpls_list_len EQUAL 1)
    message(FATAL_ERROR "tpls_list is empty! You have to enable at least one backend library")
  endif()
  if(NOT tpls_list_len EQUAL backend_list_len)
    message(FATAL_ERROR "Lists have different lengths, cannot zip")
  endif()
  set(${tpls_list} ${${tpls_list}} PARENT_SCOPE)
  set(${backend_list} ${${backend_list}} PARENT_SCOPE)
  set(${default_backend} ${has_default_backend} PARENT_SCOPE)
endfunction()
