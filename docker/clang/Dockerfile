# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

ARG BASE=ubuntu:24.04
FROM $BASE

# Clang/Clang-tidy are taken from the Ubuntu archive instead of apt.llvm.org.
# apt.llvm.org is reachable over IPv6 only, while the Github hosted runners are
# IPv4 only, so bootstrapping the LLVM repository always fails there with
# "Network is unreachable". The noble-updates pocket, enabled by default in the
# ubuntu:24.04 image, provides LLVM 19 and 20 (18 comes from the release
# pocket), so LLVM_VERSION may be set to 18, 19 or 20 on this base.
ARG LLVM_VERSION=19

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        git \
        bc \
        ccache \
        libfftw3-dev \
        openmpi-bin \
        libopenmpi-dev \
        clang-${LLVM_VERSION} \
        clang-tidy-${LLVM_VERSION} \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory "*"

# Verify the installation
RUN gcc --version && g++ --version && \
    clang-${LLVM_VERSION} --version && clang-tidy-${LLVM_VERSION} --version

# Install newer CMake manually
ARG CMAKE_VERSION=3.23.2

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

# Set the installed Clang as the default Clang, Clang++ and Clang-tidy
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-${LLVM_VERSION} 100

WORKDIR /work
CMD ["bash"]
