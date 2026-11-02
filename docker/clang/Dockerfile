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

# Set the installed Clang as the default Clang, Clang++ and Clang-tidy
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-${LLVM_VERSION} 100

WORKDIR /work
CMD ["bash"]
