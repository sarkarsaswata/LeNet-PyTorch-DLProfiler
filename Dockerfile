FROM quangbd/dc-pytorch:1.6.0-python3.6-cuda11.0-cudnn8-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
libgl1\
libgl1-mesa-glx \
libglib2.0-0 -y
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
WORKDIR ./
COPY . ./
RUN python3.6 -m pip install -r requirements.txt
RUN python3.6 -m pip install opencv-python==4.0.0.21