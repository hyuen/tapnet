FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git lsb-core

RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
