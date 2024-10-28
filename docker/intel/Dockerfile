# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

ARG BASE=intel/oneapi-basekit:2024.0.1-devel-ubuntu20.04
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

RUN KEYDUMP_URL=https://cloud.cees.ornl.gov/download && \
    KEYDUMP_FILE=keydump && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE} && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE}.sig && \
    gpg --import ${KEYDUMP_FILE} && \
    gpg --verify ${KEYDUMP_FILE}.sig ${KEYDUMP_FILE} && \
    rm ${KEYDUMP_FILE}*

ARG CMAKE_VERSION=3.25.2
ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION} && \
    CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    CMAKE_SHA256=cmake-${CMAKE_VERSION}-SHA-256.txt && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256} && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256}.asc && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SCRIPT} && \
    gpg --verify ${CMAKE_SHA256}.asc ${CMAKE_SHA256} && \
    grep -i ${CMAKE_SCRIPT} ${CMAKE_SHA256} | sed -e s/linux/Linux/ | sha256sum --check && \
    mkdir -p ${CMAKE_DIR} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    rm cmake*
ENV PATH=${CMAKE_DIR}/bin:$PATH

WORKDIR /work
CMD ["bash"]
