FROM docker.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install cmake gcc g++ -y
RUN apt-get install lsb-release -y
RUN apt-get install wget -y
RUN apt-get install gnupg2 -y
RUN apt-get install git -y
RUN apt-get install software-properties-common -y

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
