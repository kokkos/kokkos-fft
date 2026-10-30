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

# \brief Set the architectures and language for the matching KOKKOSFFT_COMPILE_LANGUAGE.
# \param target[in] Target to configure
function(kokkosfft_set_language kokkosfft_target)
  if(NOT KOKKOSFFT_COMPILE_LANGUAGE STREQUAL CXX)
    get_target_property(TARGET_SOURCES ${kokkosfft_target} SOURCES)
    set_source_files_properties(${TARGET_SOURCES} PROPERTIES LANGUAGE ${KOKKOSFFT_COMPILE_LANGUAGE})
    set_target_properties(
      ${kokkosfft_target} PROPERTIES ${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES
                                     ${KOKKOSFFT_${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES}
    )
  endif()
endfunction()
