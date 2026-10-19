# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# \brief Left padding function for better alignment of a string
#        https://gist.github.com/jtanx/96ded5e050d5ee5b19804195ee5cf5f9
# \param output[out] Output string that may be paded
# \param str[in] Input string
# \param length[in] The length of an output string
function(pad_string output_string input_string length)
  # Get the length of the input string
  string(LENGTH "${input_string}" input_length)
  # Check if padding is necessary
  if(${input_length} LESS ${length})
    # Calculate the number of spaces needed for padding
    math(EXPR padding_length "${length} - ${input_length}")

    # Create a string of spaces for padding
    string(REPEAT " " ${padding_length} padding)

    # Append the padding to the input string
    set(padded_string "${padding}${input_string}")
  else()
    set(padded_string "${input_string}")
  endif()
  # Set the output variable
  set(${output_string} "${padded_string}" PARENT_SCOPE)
endfunction()

# \brief Set the architectures for the matching KOKKOSFFT_COMPILE_LANGUAGE.
# \param target[in] Target to configure
function(kokkosfft_set_architectures target)
  if(NOT KOKKOSFFT_COMPILE_LANGUAGE STREQUAL CXX)
    set_property(
      TARGET ${target} PROPERTY ${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES
                                ${KOKKOSFFT_${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES}
    )
    if(KOKKOSFFT_COMPILE_LANGUAGE STREQUAL CUDA AND Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
      set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
  else()
    return()
  endif()
endfunction()

# \brief Set the sources languages to KOKKOSFFT_COMPILE_LANGUAGE
# \param target[in] Target to configure
function(kokkosfft_set_language)
  if(${ARGC} LESS 1)
    message(WARNING "kokkosfft_set_language() need at least one argument")
  endif()
  if(NOT KOKKOSFFT_COMPILE_LANGUAGE STREQUAL CXX)
    set_source_files_properties(${ARGV} PROPERTIES LANGUAGE ${KOKKOSFFT_COMPILE_LANGUAGE})
  else()
    return()
  endif()
endfunction()
