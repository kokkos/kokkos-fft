# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

ARG BASE=intel/oneapi-basekit:2025.1.0-0-devel-ubuntu24.04
FROM $BASE

ARG ADDITIONAL_PACKAGES

RUN apt-get update && apt-get install -y \
        bc \
        wget \
        git \
        ccache \
        libfftw3-dev \
        $ADDITIONAL_PACKAGES \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV FFTWDIR "/usr"

RUN git config --global --add safe.directory "*"

# Install newer CMake manually
ARG CMAKE_VERSION=3.25.2

ENV CMAKE_DIR=/opt/cmake
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
ENV PATH=${CMAKE_DIR}/bin:$PATH

WORKDIR /work
CMD ["bash"]

