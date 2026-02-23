FROM oi-openalpr-nvidia-py38:latest

RUN pip install 'h5py==2.10.0' --force-reinstall \
    && pip install --upgrade tensorflow_hub==0.12.0 \
    && pip install --upgrade numpy==1.22 \
    && apt-get update  \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD ["python", "./NewSR.py"]
