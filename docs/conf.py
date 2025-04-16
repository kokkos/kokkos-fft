# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import subprocess, os
import re
from datetime import datetime

def configureDoxyfile(src_dir, input_dirs, output_dir, doxyfile_in, doxyfile_out):
    
    with open(doxyfile_in, 'r') as file :
        filedata = file.read()
        
    filedata = filedata.replace('@CMAKE_SOURCE_DIR@', src_dir)
    filedata = filedata.replace('@DOXYGEN_INPUT_DIR1@', input_dirs[0])
    filedata = filedata.replace('@DOXYGEN_INPUT_DIR2@', input_dirs[1])
    filedata = filedata.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)
    
    with open(doxyfile_out, 'w') as file:
        file.write(filedata)
		
def get_version(src_dir: str) -> str:
    """
    Extracts the version string from a CMakeLists.txt file.

    The function looks for a line formatted like:
        project(NAME VERSION 1.2.0 LANGUAGES CXX)
    and returns the version (e.g., "1.2.0").

    Parameters
    ----------
    src_dir : str
        The directory including CMakeLists.txt

    Returns
    -------
    str
        The version string if found.

    Raises
    ------
    ValueError
        If the version cannot be found in the file.
    """
    cmake_file = src_dir + 'CMakeLists.txt'
    
    # Define a regex pattern to capture the version string after 'VERSION'.
    # Explanation:
    #   - project\(: matches the literal "project(".
    #   - [^)]*?: lazily matches any characters except the closing parenthesis.
    #   - \bVERSION\s+: matches the word "VERSION" followed by one or more spaces.
    #   - ([\d\.]+): capture group for the version number (digits and dots).
    pattern = re.compile(r'project\([^)]*\bVERSION\s+([\d\.]+)', re.IGNORECASE)
    with open(cmake_file, 'r', encoding='utf-8') as f:
        content = f.read()

    match = pattern.search(content)
    if match:
        return match.group(1)
    else:
        raise ValueError("Version information not found in the CMakeLists.txt file.")

# -- Project information -----------------------------------------------------
author = 'Yuuichi Asahi'
project = 'KokkosFFT'
copyright = f"2023-{datetime.now().year}, {author}"

version = get_version('../')
release = 'release'

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

breathe_projects = {}

if read_the_docs_build:
    cwd = os.getcwd()
    print(cwd)

    src_dir = f'{cwd}/..'
    input_dirs = [f'{cwd}/../common/src/', f'{cwd}/../fft/src/']
    output_dir = f'{cwd}/doxygen/'
    doxyfile_in = f'{cwd}/Doxyfile.in'
    doxyfile_out = f'{cwd}/Doxyfile'
    configureDoxyfile(src_dir, input_dirs, output_dir, doxyfile_in, doxyfile_out)
    subprocess.call('pwd; ls -lat; doxygen Doxyfile; ls -lat doxygen/xml', shell=True)
    breathe_projects[project] = output_dir + '/xml'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#...

extensions = [ "breathe" ]

#...

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Breathe Configuration
breathe_default_project = project
