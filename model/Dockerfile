FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y portaudio19-dev xxd
RUN pip install tensorflow-io pyaudio tdqm

