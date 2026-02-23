# FROM python:3.6.15
# FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu20.04
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update  \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libcurl3-dev \
    libleptonica-dev \
    liblog4cplus-dev \
    libpq-dev \
    libopencv-dev \
    libtesseract-dev \
    python3-pip \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 

COPY requirements_linux_container-heavy.txt /app/requirements_linux_container-heavy.txt
WORKDIR /app

ENV CUDA_ROOT="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:$PATH"

RUN pip install -r requirements_linux_container-heavy.txt \
    && pip install --upgrade numpy \
    && pip install --upgrade scikit-learn \
    && pip install 'h5py==2.10.0' --force-reinstall \
    && apt-get update  \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata   \
    && sed -i "s/.decode\(\'utf-8\'\)//g"   /usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/saving/hdf5_format.py

