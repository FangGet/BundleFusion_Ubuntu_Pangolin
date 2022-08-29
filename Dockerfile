# FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04
#https://hub.docker.com/r/nvidia/cudagl/

RUN apt update &&  \
    DEBIAN_FRONTEND="noninteractive" apt install -y --no-install-recommends  \
    wget unzip git make cmake gcc clang gdb libeigen3-dev libncurses5-dev libncursesw5-dev libfreeimage-dev \
    # libs for FFMPEG functionality in OpenCV
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk-3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Download and install OpenCV
ENV OPENCV_VERSION=4.5.4

# RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip && \
#     unzip opencv.zip && \
RUN git clone -v --progress https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout tags/${OPENCV_VERSION} -b v${OPENCV_VERSION} && \
    mkdir -p build &&  \
    cd build && \
    #cmake -D WITH_CUDA=ON \
    cmake \
    -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF .. && \
    # -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF ../opencv-${OPENCV_VERSION} && \
    make -j 48 && \
    make install && \
    cd .. && \
    # rm opencv.zip && \
    # rm -rf opencv-${OPENCV_VERSION} && \
    rm -rf build

# Download and install pangolin
ENV PANGOLIN_VERSION=0.6
RUN apt update && apt install -y  libgl1-mesa-dev libglew-dev libboost-all-dev
RUN git clone -v --progress https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    git checkout tags/v${PANGOLIN_VERSION} -b v${PANGOLIN_VERSION} && \
    mkdir -p build &&  \
    cd build && \
    cmake .. && \
    make -j48 && \
    make install && \
    cd .. && \
    rm -rf build

#pangolon fix #replace "GL/glew.h" with "/usr/include/GL/glew.h"
COPY pangolin_fix.h /usr/local/include/pangolin/gl/glplatform.h

WORKDIR /app


ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


COPY CMakeLists.txt zParametersBundlingDefault.txt zParametersDefault.txt ./
# ADD cmake cmake
ADD include include
ADD src src
ADD example example


RUN mkdir build && \
    cd build &&  \
    cmake -DVISUALIZATION=ON .. &&  \
    make -j48

# RUN apt update && apt install -y mesa-utils
# ENTRYPOINT ["./build/bundle_fusion_example", "./zParametersDefault.txt", "./zParametersBundlingDefault.txt"] 