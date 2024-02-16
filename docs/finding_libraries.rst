.. _finding_libraries:

Finding FFT libraries by CMake
==============================

Some tips to find FFT libraries for each backend. 

`fftw <http://www.fftw.org>`_
-----------------------------

If fftw is offered as a module, our cmake helper would likely find fftw.
Assuming fftw is installed in `<fftw_install_dir>`, it is expected that `<fftw_install_dir>` would be found under `LIBRARY_PATH`, `LD_LIBRARY_PATH`, and `PATH`.
It would look like

.. code-block:: bash

    LIBRARY_PATH=...:<fftw_install_dir>/lib
    LD_LIBRARY_PATH=...:<fftw_install_dir>/lib
    PATH=...:<fftw_install_dir>/bin

If CMake fails to find `fftw`, please try to set `FFTWDIR` in the following way. 

.. code-block:: bash

    export FFTWDIR=<fftw_install_dir>

`cufft <https://developer.nvidia.com/cufft>`_
---------------------------------------------

`hipfft <https://github.com/ROCm/hipFFT>`_
------------------------------------------

`oneMKL <https://spec.oneapi.io/versions/latest/elements/oneMKL/source/index.html>`_
------------------------------------------------------------------------------------

The most likely scenario to miss oneMKL is that forgetting to inisitalize oneAPI. 
Please make sure to initialize oneAPI as

.. code-block:: bash

    <oneapi_install_dir>/setvars.sh --include-intel-llvm
