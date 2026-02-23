# FROM python:3.6.15

# COPY . /app
# COPY requirements_linux_container-App.txt /app/requirements_linux_container.txt

# WORKDIR /app

# RUN pip install -r requirements_linux_container.txt \
#     && apt-get update  \
#     && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#         ffmpeg \
#         libsm6 \
#         libxext6 \
#         tzdata \
#         && rm -rf /var/lib/apt/lists/* \
#         && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
#         && dpkg-reconfigure -f noninteractive tzdata

# FROM oi-nvidia-light-py38:latest
FROM oi-openalpr-nvidia-py38:latest

WORKDIR /app
# CMD ["python", "./no_op.py"]
CMD ["python", "./SimilarityProcess.py"]