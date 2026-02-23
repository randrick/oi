# ##################### BASE IMAGE ##########################
# FROM python:3.6.15


# ########################## OPENALRP ##############################

# # Install prerequisites
# RUN apt-get update  \
#     && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#         build-essential \
#         cmake \
#         curl \
#         git \
#         libcurl3-dev \
#         libleptonica-dev \
#         liblog4cplus-dev \
#         libopencv-dev \
#         libtesseract-dev \
#         wget \
#         # for debugging container via shell
#         vim \
#     && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/norkator/openalpr.git
# WORKDIR /openalpr/src/build

# RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_INSTALL_SYSCONFDIR:PATH=/etc .. \
#     && make -j$(nproc) \
#     && make install


# ##################### OPEN INTELLIGENCE ##########################

# COPY . /app
# COPY requirements_linux_container-App.txt /app/requirements_linux_container.txt

# WORKDIR /app

# # RUN pip install -r requirements_linux.txt --no-dependencies
# RUN pip install -r requirements_linux_container.txt \
#     && apt-get update  \
#     && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#         ffmpeg \
#         libsm6 \
#         libxext6 \
#         tzdata \
#     && rm -rf /var/lib/apt/lists/* \
#     && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
#     && dpkg-reconfigure -f noninteractive tzdata
FROM oi-nvidia-light-py38:latest

WORKDIR /app
#CMD ["python", "./no_op.py"]
CMD ["python", "./VideoApp.py"]
