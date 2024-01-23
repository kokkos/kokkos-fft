ARG TENSORFLOW_VERSION=21.12
FROM nvcr.io/nvidia/tensorflow:${TENSORFLOW_VERSION}-tf2-py3

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y sudo && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y libfftw3-dev && \
    apt-get clean && \
    git config --global --add safe.directory '*' && \
    rm -rf /var/lib/apt/lists/*

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
ENV FFTWDIR "/usr"
CMD ["bash"]