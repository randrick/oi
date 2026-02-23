
FROM oi-openalpr-nvidia-py38:latest

WORKDIR /app
# RUN pip install --upgrade debugpy
CMD ["python", "./New_image_object_detection.py"]
