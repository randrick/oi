# FROM oi-linux-heavy-base:latest
FROM oi-nvidia-heavy-py38:latest

WORKDIR /usr/local/src
RUN git clone https://github.com/norkator/openalpr.git
WORKDIR /usr/local/src/openalpr/src/build

RUN apt-get update  \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i "s/\/storage\/projects\/alpr\/libraries\/opencv\//\/usr\/lib\/x86_64-linux-gnu\/cmake\/opencv4\//" ../CMakeLists.txt \
    && cmake \
        -DCMAKE_INSTALL_PREFIX:PATH=/usr \ 
        -DCMAKE_INSTALL_SYSCONFDIR:PATH=/etc \
        .. \
    && make -j$(nproc) \
    && make install \
    && make clean \
    && pip install --upgrade numpy==1.22

WORKDIR /app
COPY *.jpg /app
