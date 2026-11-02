# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

ARG BASE=ubuntu:24.04
FROM $BASE

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        git \
        bc \
        ccache \
        libfftw3-dev \
        python3 \
        gfortran \
        openmpi-bin \
        libopenmpi-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory "*"

WORKDIR /work
CMD ["bash"]
