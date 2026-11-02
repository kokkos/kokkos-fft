# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

ARG BASE=intel/oneapi-basekit:2025.3.0-0-devel-ubuntu24.04
FROM $BASE

ARG ADDITIONAL_PACKAGES

RUN apt-get update && apt-get install -y \
        bc \
        wget \
        git \
        ccache \
        libfftw3-dev \
        openmpi-bin \
        libopenmpi-dev \
        $ADDITIONAL_PACKAGES \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV FFTWDIR=/usr 

RUN git config --global --add safe.directory "*"

WORKDIR /work
CMD ["bash"]
