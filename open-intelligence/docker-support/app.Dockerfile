
FROM oi-openalpr-nvidia-py38:latest

   	
WORKDIR /app
#CMD ["python", "./no_op.py"]
CMD ["python", "./App.py"]
