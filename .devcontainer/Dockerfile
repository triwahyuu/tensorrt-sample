FROM nvcr.io/nvidia/tensorrt:21.06-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y install libcanberra-gtk-module libcanberra-gtk3-module \
      language-pack-en-base \
      wget git gdb \
      libv4l-dev \
      libxvidcore-dev \
      libx264-dev \
      libgtk-3-dev \
      libavcodec-dev \
      libavformat-dev \
      libswscale-dev \
      libevent-dev \
      libjpeg-dev \
      libpng-dev \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev \
      libdc1394-22-dev \
      libavresample-dev \
      libyaml-cpp-dev curl libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

WORKDIR /tmp

RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.17.5/cmake-3.17.5-linux-x86_64.tar.gz && \
  tar -xzf cmake-3.17.5-linux-x86_64.tar.gz && \
  cp -rf cmake-3.17.5-Linux-x86_64/* /usr

#install opencv for c++
RUN git clone --single-branch -b 4.4.0 https://github.com/opencv/opencv_contrib.git && \
  git clone --single-branch -b 4.4.0 https://github.com/opencv/opencv.git && \
  mkdir -p opencv/build && \
  cd opencv/build && \
  cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_TBB=ON \
    -DENABLE_FAST_MATH=ON \
    -DCUDA_FAST_MATH=ON \
    -DWITH_CUBLAS=ON \
    -DWITH_LIBV4L=ON \
    -DWITH_GSTREAMER=ON \
    -DWITH_GSTREAMER_0_10=OFF .. && \
  make -j$(nproc) && \
  make install && \
  ldconfig && \
  cd /tmp && rm -rf /tmp/*

  
ENV ONNX_ML=0

RUN \
  git clone --single-branch -b v3.12.3 https://github.com/protocolbuffers/protobuf.git \
  && cd protobuf \
  && ./autogen.sh \
  && ./configure \
  && make -j$(nproc) \
  && make install \
  && ldconfig

RUN \
  git clone -b v1.9.0 https://github.com/onnx/onnx.git \
  && mkdir -p onnx/build \
  && cd onnx/build \
  && cmake -DONNX_NAMESPACE=onnx .. \
  && make -j$(nproc) \
  && make install \
  && ldconfig

RUN pip install pip --upgrade && \
  pip install torch torchvision numpy matplotlib onnx

ENV PATH="/usr/bin:${PATH}"
WORKDIR /workspace
