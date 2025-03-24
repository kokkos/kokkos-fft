# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" 
A module to check the versions in documents are identical
"""

import re
import pathlib
from typing import List

def extract_versions(filename: str, modules: List[str]) -> dict:
    """
    Extracts the version information for a list of modules from a markdown file.

    Parameters
    ----------
    filename : str
        Path to the markdown file.
    modules : List[str]
        List of module names to search for (e.g., ['CMake', 'gcc', 'nvcc']).

    Returns
    -------
    dict
        A dictionary where each key is a module name and the value is a tuple (major, minor, patch).
    """
    results = {}

    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return results

    for module in modules:
        # Regex explanation:
        # - re.escape(module): Escapes any special characters in the module name.
        # - \s+: one or more whitespace characters.
        # - (\d+): captures the major version.
        # - \.: literal dot.
        # - (\d+): captures the minor version.
        # - (?:\.(\d+))?: optionally captures the patch version if present.
        # - [^\d]*: ignores any trailing non-digit characters (like '+' or letters in parentheses).
        pattern = rf"{re.escape(module)}\s+(\d+)\.(\d+)(?:\.(\d+))?[^\d]*"
        match = re.search(pattern, content)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            # If patch version is not found, default it to 0.
            patch = int(match.group(3)) if match.group(3) is not None else 0
            results[module] = (major, minor, patch)
        else:
            print(f"Version information for module '{module}' not found in the file.")
    return results

def extract_cmake_versions(filename: str, modules: List[str]) -> dict:
    """
    Extracts version information from a CMakeLists.txt file.

    It extracts:
      1. The CMake minimum required version from the 'cmake_minimum_required' command.
      2. The version for each module from a line like:
         set(<MODULE_UPPER>_REQUIRED_VERSION x.y[.z])

    Parameters
    ----------
    filename : str
        Path to the CMakeLists.txt file.
    modules : List[str]
        List of module names (e.g., ['gcc', 'nvcc']).

    Returns
    -------
    dict
        A dictionary with keys as module names (plus key 'cmake') and values as
        a tuple (major, minor, patch). If patch is not provided, defaults to 0.
    """
    results = {}

    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return results

    # Extract CMake minimum required version.
    cmake_pattern = r"cmake_minimum_required\s*\(\s*VERSION\s+(\d+)\.(\d+)(?:\.(\d+))?"
    cmake_match = re.search(cmake_pattern, content, re.IGNORECASE)
    if cmake_match:
        major = int(cmake_match.group(1))
        minor = int(cmake_match.group(2))
        patch = int(cmake_match.group(3)) if cmake_match.group(3) is not None else 0
        results["CMake"] = (major, minor, patch)
    else:
        print("Warning: CMake minimum required version not found.")

    # Extract versions for each module from the list.
    for module in modules:
        module_upper = module.upper()
        # Build regex to find a line like: set(MODULE_REQUIRED_VERSION x.y.z)
        pattern = rf"set\s*\(\s*{module_upper}_REQUIRED_VERSION\s+(\d+)\.(\d+)(?:\.(\d+))?\s*\)"
        match = re.search(pattern, content)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            patch = int(match.group(3)) if match.group(3) is not None else 0
            results[module] = (major, minor, patch)
        else:
            print(f"Warning: Version information for module '{module}' not found.")

    return results

def is_subdictionary(sub: dict, main: dict) -> bool:
    """
    Check whether the first dictionary (sub) is a sub-dictionary of the second (main),
    ignoring differences in the cases of the keys.

    Parameters
    ----------
    sub : dict
        The dictionary to check as a potential sub-dictionary.
    main : dict
        The dictionary to be checked against.

    Returns
    -------
    bool
        True if 'sub' is a sub-dictionary of 'main', otherwise False.
    """
    # Create lower-case key versions of both dictionaries.
    sub_lower = {key.lower(): value for key, value in sub.items()}
    main_lower = {key.lower(): value for key, value in main.items()}

    # Check that each lower-case key from sub exists in main with the same value.
    return all(key in main_lower and main_lower[key] == sub_lower[key] for key in sub_lower)

if __name__ == "__main__":
    # Get versions from the README.md as a reference
    module_list = ["CMake", "Kokkos", "gcc", "IntelLLVM", "nvcc", "rocm"]
    reference_readme = "README.md"
    reference_versions = extract_versions(reference_readme, module_list)

    # Get versions from multiple files
    readme_in_example = pathlib.Path("examples/10_HasegawaWakatani") / "README.md"
    docs_building = pathlib.Path("docs/intro") / "building.rst"
    docs_quick_start = pathlib.Path("docs/intro") / "quick_start.rst"
    cmake_file = pathlib.Path(".") / "CMakeLists.txt"

    error_message = ""
    inspected_files = [readme_in_example, docs_building, docs_quick_start, cmake_file]
    for inspected_file in inspected_files:
        versions_in_file = extract_cmake_versions(str(inspected_file), ["Kokkos"]) \
            if inspected_file == cmake_file else extract_versions(str(inspected_file), module_list)

        if is_subdictionary(sub=versions_in_file, main=reference_versions):
            print(f"All the versions are consistent between {inspected_file} and README.md.")
        else:
            # Create a dictionary to store the mismatches.
            mismatches = {}

            # Get the union of keys from both dictionaries.
            all_keys = set(reference_versions.keys()).union(versions_in_file.keys())

            for key in all_keys:
                val1 = reference_versions.get(key)
                val2 = versions_in_file.get(key)

                # Check only if the key exists in both dictionaries.
                if val2 is not None and val1 != val2:
                    mismatches[key] = (
                        f"{str(reference_readme)}: {val1}",
                        f"{str(inspected_file)}: {val2}",
                    )

            error_message += (
                f"Versions are not identical between {inspected_file} "
                "and README.md\n"
            )
            for key, values in mismatches.items():
                error_message += f"  {key}:\n"
                error_message += f"    {values[0]}\n"
                error_message += f"    {values[1]}\n"

    if error_message:
        raise ValueError(error_message)
