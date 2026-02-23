# FROM python:3.6.15

# COPY requirements_linux_container-heavy.txt /app/requirements_linux_container.txt


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

# FROM oi-linux-heavy-base:latest
FROM oi-openalpr-nvidia-py38:latest

COPY models/retinaface_r50_v1/R50-0000.params /root/.insightface/models/retinaface_r50_v1/R50-0000.params
COPY models/retinaface_r50_v1/R50-symbol.json /root/.insightface/models/retinaface_r50_v1//R50-symbol.json

# COPY . /app
WORKDIR /app
# CMD ["python", "./no_op.py"]
CMD ["python", "./InsightFace.py"]
