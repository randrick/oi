# FROM python:3.6.15
# FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu20.04
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
# FROM nvidia/cuda:12.3.2-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update  \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    python3.8 \
    python3-pip \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 

COPY requirements_nvidia-opencv-tensorflow-cuda11.8.0-py38.txt /app/requirements_nvidia-opencv-tensorflow-cuda11.8.0-py38.txt
WORKDIR /app

ENV CUDA_ROOT="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:$PATH"

RUN pip install -r requirements_nvidia-opencv-tensorflow-cuda11.8.0-py38.txt \
    && apt-get update  \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata
# && sed -i "s/.decode\(\'utf-8\'\)//g"   /usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/saving/hdf5_format.py

