# FROM docker.io/nvidia/cuda:11.4.3-devel-ubuntu20.04
# FROM docker.io/nvidia/cuda:11.7.1-runtime-ubuntu22.04
FROM docker.io/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install cmake gcc g++ -y
RUN apt-get install lsb-release -y
RUN apt-get install wget -y
RUN apt-get install gnupg2 -y
RUN apt-get install git -y
RUN apt-get install software-properties-common -y

RUN git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git /tmp/tcnn
ENV cuda=11.7
ENV TCNN_CUDA_ARCHITECTURES=89
RUN apt-get update -y
RUN bash /tmp/tcnn/dependencies/cuda-cmake-github-actions/scripts/actions/install_cuda_ubuntu.sh


RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
