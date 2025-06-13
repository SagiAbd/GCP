ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the local MMCV wheel file into the container
COPY mmcv*.whl /tmp/

# Install MMEngine first, then install MMCV from local file
RUN pip install mmengine && \
    pip install /tmp/mmcv*.whl

# Clean up the wheel files
RUN rm /tmp/mmcv*.whl
# Install MMEngine and MMCV

# Install MMDetection
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .

WORKDIR /mmdetection

# Install additional dependencies commonly needed for MMDetection
RUN pip install --no-cache-dir \
    albumentations>=1.0.3 \
    matplotlib \
    opencv-python \
    pycocotools \
    scipy \
    terminaltables \
    timm \
    tqdm \
    shapely \
    ftfy \
    regex \
    yapf \
    cityscapesscripts \
    scikit-learn \
    scikit-image \
    interrogate \
    onnx \
    onnxruntime \
    openmim

# Install optional dependencies for visualization and testing
RUN pip install  tensorboard \
    xdoctest \
    memory_profiler \
    psutil

# Update pip and setuptools
RUN pip install --upgrade pip setuptools

# Final system update and cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
