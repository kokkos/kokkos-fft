# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

ARG BASE=rocm/dev-ubuntu-24.04:6.4.3-complete
FROM $BASE

ARG ADDITIONAL_PACKAGES

RUN apt-get update && apt-get install -y \
        kmod \
        wget \
        ccache \
        cmake \
        git \
        file \
        hipfft \
        rocfft \
        libfftw3-dev \
        openmpi-bin \
        libopenmpi-dev \
        python3 \
        $ADDITIONAL_PACKAGES \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV FFTWDIR=/usr

RUN git config --global --add safe.directory "*"

# Make sure MPI is discoverable
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/opt/rocm/hipfft/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/openmpi:/opt/rocm/hip/:/opt/rocm/:$CMAKE_PREFIX_PATH

WORKDIR /work
CMD ["bash"]
