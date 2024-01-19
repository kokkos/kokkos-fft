#!/bin/bash

args=$#
ROOT_DIR=$1
WK_DIR=$(pwd)
TARGET=$2
KOKKOS_TARGET="CUDA"

# Install Kokkos
export KOKKOS_INSTALL_PREFIX=${ROOT_DIR}/usr/local/kokkos_${KOKKOS_TARGET}
export Kokkos_DIR=${KOKKOS_INSTALL_PREFIX}/lib/cmake/Kokkos
export KOKKOS_BUILD_DIR=build_Kokkos_${KOKKOS_TARGET}

export KOKKOSFFT_INSTALL_PREFIX=${ROOT_DIR}/usr/local/kokkosFFT_${TARGET}
export KokkosFFT_DIR=${KOKKOSFFT_INSTALL_PREFIX}/lib/cmake/kokkos-fft
export KOKKOSFFT_BUILD_DIR=build_KokkosFFT_${TARGET}

export EXAMPLE_BUILD_DIR=build_example_${TARGET}
export CXX_COMPILER=${WK_DIR}/${KOKKOS_BUILD_DIR}/kokkos/bin/nvcc_wrapper

# Install Kokkos if not exist
if [ ! -d ${Kokkos_DIR} ]; then
    cd ${WK_DIR}
    mkdir ${KOKKOS_BUILD_DIR} && cd ${KOKKOS_BUILD_DIR}

    # Get Kokkos from github repo and build
    git clone https://github.com/kokkos/kokkos.git
    cmake -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
          -DCMAKE_CXX_STANDARD=17 -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
          -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PREFIX} kokkos

    cmake --build . -j 8
    cmake --install .
fi

# Install KokkosFFT
cd ${WK_DIR}
mkdir ${KOKKOSFFT_BUILD_DIR} && cd ${KOKKOSFFT_BUILD_DIR}
if [ $TARGET == "CUDA" ]; then
    cmake -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
            -DCMAKE_CXX_STANDARD=17 -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
            -DCMAKE_INSTALL_PREFIX=${KOKKOSFFT_INSTALL_PREFIX} ..
else
    cmake -DCMAKE_CXX_COMPILER=kokkos/bin/nvcc_wrapper \
          -DCMAKE_CXX_STANDARD=17 -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
          -DCMAKE_INSTALL_PREFIX=${KOKKOSFFT_INSTALL_PREFIX} -DKokkosFFT_ENABLE_HOST_AND_DEVICE=ON ..
fi
cmake --build . -j 8
cmake --install .

# Try to build an example
# Build KokkosFFT code using installed KokkosFFT
cd ${WK_DIR}
mkdir ${EXAMPLE_BUILD_DIR} && cd ${EXAMPLE_BUILD_DIR}
cmake -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_CXX_STANDARD=17 -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON ../install_test/src
cmake --build . -j 8

if [ $? -eq 0 ]; then
    echo "*** install test: build SUCCESSFUL ***"
else
    echo "*** install test: build FAILED ***"
    exit 1;
fi