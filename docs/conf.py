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

def configureDoxyfile(input_dir, output_dir):

	with open('Doxyfile.in', 'r') as file :
		filedata = file.read()

	filedata = filedata.replace('@CMAKE_SOURCE_DIR@', input_dir)
	filedata = filedata.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)
	
	with open('Doxyfile', 'w') as file:
		file.write(filedata)

# -- Project information -----------------------------------------------------

project = 'KokkosFFT'
copyright = '2024, Yuuichi Asahi'
author = 'Yuuichi Asahi'

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

breathe_projects = {}

if read_the_docs_build:
    cwd = os.getcwd()
    print(cwd)
    
    cmake_commands = 'cmake -DBUILD_TESTING=OFF \
                            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                            -DKokkos_ENABLE_OPENMP=ON \
                            -DKokkosFFT_INTERNAL_Kokkos=ON \
                            -DKokkosFFT_ENABLE_DOCS=ON ..'
    build = 'cmake --build . -j 2'
                            
    subprocess.call(f'cd ../; mkdir build; cd build; {cmake_commands}; {build}', shell=True)
    #input_dir = cwd + '/..'
    #output_dir = cwd +'/doxygen/'
    #doxyfile_in = cwd + '/Doxyfile.in'
    #doxyfile_out = cwd + '/Doxyfile'
    #configureDoxyfile(input_dir, output_dir, doxyfile_in, doxyfile_out)
    #subprocess.call('pwd; ls -lat; doxygen Doxyfile; ls -lat doxygen/xml', shell=True)
    #breathe_projects[project] = output_dir + '/xml'

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
