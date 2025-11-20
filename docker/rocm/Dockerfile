# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

ARG BASE=rocm/dev-ubuntu-24.04:6.4.3-complete
FROM $BASE

ARG ADDITIONAL_PACKAGES

RUN apt-get update && apt-get install -y \
        kmod \
        wget \
        ccache \
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

ENV FFTWDIR=/usr \
    CMAKE_DIR=/opt/cmake \
    PATH=/opt/rocm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/hipfft/lib:$LD_LIBRARY_PATH \
    CMAKE_PREFIX_PATH=/opt/rocm/hip/:/opt/rocm/:$CMAKE_PREFIX_PATH

RUN git config --global --add safe.directory "*"

# Install newer CMake manually
ARG CMAKE_VERSION=3.23.2

RUN CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION} && \
    CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    CMAKE_SHA256=cmake-${CMAKE_VERSION}-SHA-256.txt && \
    CMAKE_SIGNATURE=cmake-${CMAKE_VERSION}-SHA-256.txt.asc && \
    PUBLIC_KEY_ID=CBA23971357C2E6590D9EFD3EC8FEF3A7BFB4EDA && \
    PUBLIC_KEY_URL=https://keys.openpgp.org/vks/v1/by-fingerprint/${PUBLIC_KEY_ID} && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256} && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SIGNATURE} && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SCRIPT} && \
    wget --quiet ${PUBLIC_KEY_URL} && \
    gpg --import ${PUBLIC_KEY_ID} && \
    gpg --verify ${CMAKE_SIGNATURE} ${CMAKE_SHA256} && \
    grep -i ${CMAKE_SCRIPT} ${CMAKE_SHA256} | sha256sum --check && \
    mkdir -p ${CMAKE_DIR} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    rm cmake* ${PUBLIC_KEY_ID}

# Make sure MPI is discoverable
ENV PATH=${CMAKE_DIR}/bin:/usr/lib/x86_64-linux-gnu/openmpi/bin:/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/opt/rocm/hipfft/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/openmpi:/opt/rocm/hip/:/opt/rocm/:$CMAKE_PREFIX_PATH

WORKDIR /work
CMD ["bash"]
