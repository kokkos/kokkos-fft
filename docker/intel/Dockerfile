ARG BASE=nvidia/cuda:11.7.1-devel-ubuntu22.04
FROM $BASE

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y \
        bc \
        wget \
        ccache \
        ninja-build \
        python3 \
        git \
        libfftw3-dev \
        libomp-dev \
        && \
    apt-get clean && \
    git config --global --add safe.directory '*' && \
    rm -rf /var/lib/apt/lists/*

RUN KEYDUMP_URL=https://cloud.cees.ornl.gov/download && \
    KEYDUMP_FILE=keydump && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE} && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE}.sig && \
    gpg --import ${KEYDUMP_FILE} && \
    gpg --verify ${KEYDUMP_FILE}.sig ${KEYDUMP_FILE} && \
    rm ${KEYDUMP_FILE}*

ARG CMAKE_VERSION=3.23.2
ARG build_dir=/tmp/build

ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    mkdir -p ${build_dir} && mkdir -p ${CMAKE_DIR} \
    cd ${build_dir} && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_SCRIPT} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    cd ${HOME} && rm -rf ${build_dir}
ENV PATH=${CMAKE_DIR}/bin:$PATH

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update -o Dir::Etc::sourcelist="sources.list.d/oneAPI.list" -o APT::Get::List-Cleanup="0" && \
    apt-get install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2023.0.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://cloud.cees.ornl.gov/download/oneapi-for-nvidia-gpus-2023.0.0-linux.sh && \
    chmod +x oneapi-for-nvidia-gpus-2023.0.0-linux.sh && \
    ./oneapi-for-nvidia-gpus-2023.0.0-linux.sh -y && \
    rm oneapi-for-nvidia-gpus-2023.0.0-linux.sh

RUN wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46.sh &&\
    chmod +x ./l_BaseKit_p_2024.0.1.46.sh && \
    ./l_BaseKit_p_2024.0.1.46.sh -a -s --eula accept && \
    rm l_BaseKit_p_2024.0.1.46.sh

WORKDIR /work
ENV FFTWDIR "/usr"
RUN . /opt/intel/oneapi/setvars.sh --include-intel-llvm
ENV PATH=${CMAKE_DIR}/bin:/opt/intel/oneapi/compiler/latest/bin:$PATH
ENV CMAKE_PREFIX_PATH "$CMAKE_PREFIX_PATH;/opt/intel/oneapi/dpl/latest/lib/cmake/oneDPL/;/opt/intel/oneapi/mkl/latest/lib/cmake/mkl/"
ENV oneDPL_DIR=/opt/intel/oneapi/dpl/latest/lib/cmake/oneDPL
ENV MKL_DIR=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl
CMD ["bash"]