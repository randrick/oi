# FROM python:3.6.15
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install prerequisites
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

COPY requirements_linux_container-light.txt /app/requirements_linux_container.txt
WORKDIR /app

RUN pip install -r /app/requirements_linux_container.txt \
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
    && dpkg-reconfigure -f noninteractive tzdata
#  \
# The only python module that appears to have issues with python3.8
# This does not make sense that the 3.8 dist would not work
# && sed -i "s/.decode\(\'utf-8\'\)//g"   /usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/saving/hdf5_format.py

