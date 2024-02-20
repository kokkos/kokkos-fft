#!/bin/bash

args=$#
WK_DIR=/tmp/kokkos_fft_as_library
PWD=$(pwd)
ROOT_DIR=$1
BUILD_TYPE=$2
C_COMPILER=$3
CXX_COMPILER=$4
BACKEND_FLAG=$5
TARGET_FLAG=$6

# Install Kokkos
export KOKKOS_INSTALL_PREFIX=${ROOT_DIR}/usr/local/kokkos
export KOKKOS_BUILD_DIR=build_Kokkos

export KOKKOSFFT_INSTALL_PREFIX=${ROOT_DIR}/usr/local/kokkosFFT
export KOKKOSFFT_BUILD_DIR=build_KokkosFFT

export EXAMPLE_BUILD_DIR=build_example_${TARGET}

# Make work directory
mkdir -p ${WK_DIR}

# Install Kokkos
# Get Kokkos from github repo and build
cd ${WK_DIR}
mkdir ${KOKKOS_BUILD_DIR} && cd ${KOKKOS_BUILD_DIR}
git clone https://github.com/kokkos/kokkos.git
cmake -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PREFIX} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DCMAKE_C_COMPILER=${C_COMPILER} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_CXX_STANDARD=17 \
      ${BACKEND_FLAG}
      
cmake --build . -j 8
cmake --install .

# Install KokkosFFT on pre-installbed Kokkos
cd ${WK_DIR}
mkdir ${KOKKOSFFT_BUILD_DIR} && cd ${KOKKOSFFT_BUILD_DIR}
cmake -DCMAKE_PREFIX_PATH=${KOKKOS_INSTALL_PREFIX} \
      -DCMAKE_INSTALL_PREFIX=${KOKKOSFFT_INSTALL_PREFIX} \
      ${TARGET_FLAG}

cmake --build . -j 8
cmake --install .

# Try to build an example
# Build KokkosFFT code using installed KokkosFFT
cd ${WK_DIR}
mkdir ${EXAMPLE_BUILD_DIR} && cd ${EXAMPLE_BUILD_DIR}
cmake -DCMAKE_PREFIX_PATH="${KOKKOS_INSTALL_PREFIX};${KOKKOSFFT_INSTALL_PREFIX}" ../build_test/as_library
cmake --build . -j 8

if [ $? -eq 0 ]; then
    echo "*** KokkosFFT as library test: build SUCCESSFUL ***"
else
    echo "*** KokkosFFT as library test: build FAILED ***"
    exit 1;
fi