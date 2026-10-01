# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

function(kokkosfft_add_example EXAMPLE_NAME)
  cmake_parse_arguments(PARSE_ARGV 1 PARSE "" "" "SOURCES")

  add_executable(${EXAMPLE_NAME} ${PARSE_SOURCES})
  target_link_libraries(${EXAMPLE_NAME} PUBLIC KokkosFFT::fft)

  if(Kokkos_COMPILE_LANGUAGE)
    set_source_files_properties(${PARSE_SOURCES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
  endif()
endfunction()

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

# Documentations
add_subdirectory(docs)

# if(Kokkos_COMPILE_LANGUAGE)
#   file(GLOB_RECURSE EXAMPLE_SOURCES "*.cpp")
#   foreach(source ${EXAMPLE_SOURCES})
#     cmake_path(GET source PARENT_PATH source_dir)
#     message(STATUS "source ${source}")
#     message(STATUS "source_dir ${source_dir}")
#     set_source_files_properties(${source} DIRECTORY ${source_dir} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
#   endforeach()
# endif()
