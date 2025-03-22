# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" unit-test for functions in check_versions.py
"""

from check_versions import (
    extract_versions,
    extract_cmake_versions,
    is_subdictionary,
)

def test_extract_versions_normal(tmp_path):
    """
    Test extract_versions with a markdown file containing version details.

    This test creates a temporary file with version information,
    then calls extract_versions with a modules list (including one missing module 'nvcc'),
    and asserts that only the found versions for 'CMake' and 'gcc' are returned.
    """
    # Create temporary markdown file with version info
    content = (
        "This document holds version details.\n"
        "CMake 3.20.1 extra text...\n"
        "gcc   4.8.5 additional info\n"
        "Some other random line.\n"
    )
    file_path = tmp_path / "test.md"
    file_path.write_text(content, encoding="utf-8")

    # Call extract_versions with modules list including one missing module 'nvcc'
    modules = ["CMake", "gcc", "nvcc"]
    result = extract_versions(str(file_path), modules)

    # Expect only keys for CMake and gcc
    expected = {
        "CMake": (3, 20, 1),
        "gcc": (4, 8, 5)
    }
    assert result == expected

def test_file_not_found():
    """
    Test extract_versions handling when the input file is not found.

    This test uses a non-existent file name and verifies that extract_versions
    returns an empty dictionary.
    """
    # Provide a file name that does not exist.
    non_existent_file = "non_existent_test_file.md"
    result = extract_versions(non_existent_file, ["CMake"])
    # Check that the result is empty.
    assert not result

def test_extract_cmake_versions_normal(tmp_path):
    """
    Test extract_cmake_versions with a valid CMakeLists.txt file.

    This test creates a temporary CMakeLists.txt file containing the minimum required 
    version for CMake and version information for modules 'gcc' and 'nvcc'. It then verifies
    that extract_cmake_versions correctly parses and returns the expected versions.
    """
    content = (
        "cmake_minimum_required(VERSION 3.18.5)\n"
        "set(GCC_REQUIRED_VERSION 9.3.0)\n"
        "set(NVCC_REQUIRED_VERSION 10.1)\n"
    )
    file_path = tmp_path / "CMakeLists.txt"
    file_path.write_text(content, encoding="utf-8")

    modules = ["gcc", "nvcc"]
    result = extract_cmake_versions(str(file_path), modules)

    expected = {
        "CMake": (3, 18, 5),
        "gcc": (9, 3, 0),
        "nvcc": (10, 1, 0)
    }
    assert result == expected

def test_extract_cmake_versions_file_not_found():
    """
    Test extract_cmake_versions when the CMakeLists.txt file is not found.

    This test verifies that when a non-existent file is provided,
    extract_cmake_versions returns an empty dictionary.
    """
    non_existent_file = "non_existent_CMakeLists.txt"
    result = extract_cmake_versions(non_existent_file, ["gcc"])
    assert not result

def test_is_subdictionary():
    """
    Test is_subdictionary using two cases:
    
    1. A valid sub-dictionary where all keys (with matching case) and values from 'sub'
       are present in 'main'. The function should return True.
       
    2. A dictionary 'not_sub' which differs in at least one corresponding value from 'main'.
       The function should return False for this case.
    """
    sub = {
        "CMake": (3, 18, 5),
        "gcc": (4, 8, 5)
    }
    not_sub = {
        "CMake": (3, 20, 5),
        "gcc": (4, 8, 5)
    }
    main = {
        "CMake": (3, 18, 5),
        "gcc": (4, 8, 5),
        "nvcc": (10, 1, 0)
    }
    assert is_subdictionary(sub, main)
    assert not is_subdictionary(not_sub, main)

def test_is_subdictionary_case_insensitivity():
    """
    Test that is_subdictionary correctly handles case differences in keys.

    This test verifies that the function considers keys equal regardless of their case,
    so that a sub-dictionary with keys in different cases is still recognized as a valid
    sub-dictionary of the main dictionary.
    """
    sub = {
        "cmake": (3, 18, 5),
        "Gcc": (4, 8, 5)
    }
    main = {
        "CMake": (3, 18, 5),
        "gcc": (4, 8, 5),
        "nvcc": (10, 1, 0)
    }
    assert is_subdictionary(sub, main)
