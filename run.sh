#!/bin/sh

# Variables
PROJECT_DIR="/path/to/Ultra-Light-Fast-Generic-Face-Detector-1MB"
DOCKER_IMAGE="ultra_light_face_detector"
CONTAINER_NAME="ultra_face_container"

# Navigate to project directory
cd $PROJECT_DIR || { echo "Directory not found! Exiting."; exit 1; }

# Build Docker Image
docker build . -f DockerFile --no-cache -t $DOCKER_IMAGE

# Run Docker Commands
docker run --rm -it --gpus all $DOCKER_IMAGE cat /etc/os-release
docker run --rm -it --gpus all $DOCKER_IMAGE python3 -m pip list
docker run --rm -it --gpus all $DOCKER_IMAGE nvidia-smi

# Detect Images
docker run --rm -it --gpus all --name $CONTAINER_NAME $DOCKER_IMAGE python3 detect_imgs.py --path ./custom_data/
docker cp $CONTAINER_NAME:/home/ubuntu/detect_imgs_results/test.jpeg ./

# Detect Faces in Videos
docker run --rm -it --gpus all --name $CONTAINER_NAME $DOCKER_IMAGE python3 run_video_face_detect.py --path ./video_data/
docker cp $CONTAINER_NAME:/home/ubuntu/video_data/Detected_images/ ./

