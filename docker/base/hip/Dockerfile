ARG ROCM_VERSION=5.2.0
ARG TENSORFLOW_VERSION=2.9
FROM rocm/tensorflow:rocm${ROCM_VERSION}-tf${TENSORFLOW_VERSION}-dev

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y git && \
    git config --global --add safe.directory '*'

# Install newer CMake manually
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

WORKDIR /work
ENV CMAKE_PREFIX_PATH "$CMAKE_PREFIX_PATH;/opt/rocm/hip/;/opt/rocm/" 
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/rocm/hipfft/lib
CMD ["bash"]
