# FROM nvcr.io/nvidia/pytorch:25.10-py3
FROM python:3.10-slim

WORKDIR /app

ENV \
    http_proxy="http://fwdproxy:8080" \
    https_proxy="http://fwdproxy:8080" 
ENV HTTP_PROXY="http://fwdproxy:8080"
ENV HTTPS_PROXY="http://fwdproxy:8080"

# RUN pip install --upgrade pip
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox

RUN apt-get update && apt-get install -y git wget
RUN pip install --upgrade pip
# COPY .gitconfig .
# RUN  git clone https://github.com/google-deepmind/tapnet.git
RUN pip install git+https://github.com/google-deepmind/tapnet.git
RUN pip install git+https://github.com/google-deepmind/recurrentgemma.git@main

# should consolidate all the pip install operations once we figure out the whole list
# add a second file to fiddle with new dependencies and avoid rebuilding everything from scratch
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY requirements2.txt .
RUN pip install -r requirements2.txt

#RUN --mount=type=bind,source=./weights,target=/tmp/weights \
#    cp /tmp/weights/bootstapnext_ckpt.npz ./bootstapnext_ckpt.npz
RUN wget https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
RUN wget https://storage.googleapis.com/dm-tapnet/tapnext/tapnext_ckpt.npz

#RUN --mount=type=bind,source=./weights,target=/tmp/weights \
#    cp /tmp/weights/tapnext_ckpt.npz ./tapnext_ckpt.npz

COPY app/ ./app/

CMD ["gunicorn", "app.main:app", "--workers", "1", "--worker-class",\
    "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
