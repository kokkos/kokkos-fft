#!/bin/bash

args=$#
ROOT_DIR=$1
WK_DIR=$(pwd)
TARGET="cpu"

# Install Kokkos
export KOKKOS_INSTALL_PREFIX=${ROOT_DIR}/usr/local/kokkos_${TARGET}
export Kokkos_DIR=${KOKKOS_INSTALL_PREFIX}/lib64/cmake/Kokkos
export KOKKOS_BUILD_DIR=build_Kokkos_${TARGET}

export KOKKOSFFT_INSTALL_PREFIX=${ROOT_DIR}/usr/local/kokkosFFT_${TARGET}
export KokkosFFT_DIR=${KOKKOSFFT_INSTALL_PREFIX}/lib64/cmake/kokkos-fft
export KOKKOSFFT_BUILD_DIR=build_KokkosFFT_${TARGET}

export EXAMPLE_BUILD_DIR=build_example_${TARGET}

# Install Kokkos
cd ${WK_DIR}
mkdir ${KOKKOS_BUILD_DIR} && cd ${KOKKOS_BUILD_DIR}

# Get Kokkos from github repo and build
git clone https://github.com/kokkos/kokkos.git
cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKokkos_ENABLE_OPENMP=ON -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PREFIX} kokkos
cmake --build . -j 8
cmake --install .

# Install KokkosFFT
cd ${WK_DIR}
mkdir ${KOKKOSFFT_BUILD_DIR} && cd ${KOKKOSFFT_BUILD_DIR}
cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKokkos_ENABLE_OPENMP=ON -DCMAKE_INSTALL_PREFIX=${KOKKOSFFT_INSTALL_PREFIX} ..
cmake --build . -j 8
cmake --install .

# Try to build an example
# Build KokkosFFT code using installed KokkosFFT
cd ${WK_DIR}
mkdir ${EXAMPLE_BUILD_DIR} && cd ${EXAMPLE_BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON ../install_test/src
cmake --build . -j 8

if [ $? -eq 0 ]; then
    echo "*** install test: build SUCCESSFUL ***"
else
    echo "*** install test: build FAILED ***"
    exit 1;
fi