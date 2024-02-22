#!/bin/bash

args=$#
WK_DIR=/tmp/kokkos_fft_as_subdirectory
SRC_DIR=/work
ROOT_DIR=$1
BUILD_TYPE=$2
C_COMPILER=$3
CXX_COMPILER=$4

if [ $args -eq 5 ]; then
    # CPU build
    # e.g
    # /tmp Release gcc g++ -DKokkos_ENABLE_THREADS=ON
    # TARGET_FLAG is empty
    BACKEND_FLAG=$5
    TARGET_FLAG=$6
elif [ $args -eq 6 ]; then
    # GPU build without target flag
    # e.g.
    # /tmp Release gcc g++ \
    # -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON 
    # TARGET_FLAG is empty
    BACKEND_FLAG0=$5
    BACKEND_FLAG1=$6
    BACKEND_FLAG="${BACKEND_FLAG0} ${BACKEND_FLAG1}"
    TARGET_FLAG=$7
elif [ $args -eq 7 ]; then
    # GPU build with target flag
    # e.g.
    # /tmp Release gcc g++ \
    # -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkosFFT_ENABLE_HOST_AND_DEVICE=ON
    BACKEND_FLAG0=$5
    BACKEND_FLAG1=$6
    BACKEND_FLAG="${BACKEND_FLAG0} ${BACKEND_FLAG1}"
    TARGET_FLAG=$7
else
    echo "*** Error: Number of arguments must be 5, 6 or 7 ***"
    exit 1;
fi

# Make work directory (Project directory)
mkdir -p ${WK_DIR} && cd ${WK_DIR}

# Copy CMakeLists.txt and hello.cpp
cp ${SRC_DIR}/install_test/as_subdirectory/* .

# Prepare Kokkos and KokkosFFT under tpls
mkdir tpls && cd tpls

# Prepare tpls/kokkos 
cp -r ${SRC_DIR}/tpls/kokkos kokkos

# Copy KokkosFFT under tpls/kokkos-fft
cp -r ${SRC_DIR} kokkos-fft

ls

# Try to build an example
# Build KokkosFFT code using Kokkos and KokkosFFT as submodules

export EXAMPLE_BUILD_DIR=build_example
cd ${WK_DIR}
mkdir ${EXAMPLE_BUILD_DIR} && cd ${EXAMPLE_BUILD_DIR}
cmake -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      ${BACKEND_FLAG} \
      ${TARGET_FLAG} ..

cmake --build . -j 8

if [ $? -eq 0 ]; then
    echo "*** KokkosFFT as subdirectory test: build SUCCESSFUL ***"
else
    echo "*** KokkosFFT as subdirectory test: build FAILED ***"
    exit 1;
fi