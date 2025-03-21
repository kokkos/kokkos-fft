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
        dict: A dictionary where each key is a module name and the value is a tuple (major, minor, patch).
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

if __name__ == "__main__":    
    # Get versions from the README.md as a reference
    module_list = ['CMake', 'Kokkos', 'gcc', 'IntelLLVM', 'nvcc', 'rocm']
    reference_readme = pathlib.Path('..') / 'README.md'
    reference_versions = extract_versions(str(reference_readme), module_list)
    
    # Get versions from the README.md in an example
    readme_in_example = pathlib.Path('../examples/10_HasegawaWakatani') / 'README.md'
    versions_in_examples = extract_versions(str(readme_in_example), module_list)
    
    if reference_versions == versions_in_examples:
        print('All the versions are consistent')
    else:
        # Create a dictionary to store the mismatches.
        mismatches = {}

        # Get the union of keys from both dictionaries.
        all_keys = set(reference_versions.keys()).union(versions_in_examples.keys())

        for key in all_keys:
            val1 = reference_versions.get(key)
            val2 = versions_in_examples.get(key)
            if val1 != val2:
                mismatches[key] = (f'{str(reference_readme)}: {val1}', f'{str(readme_in_example)}: {val2}')
        
        raise ValueError(f"Versions are not identical. Mismatches: {mismatches}")
    