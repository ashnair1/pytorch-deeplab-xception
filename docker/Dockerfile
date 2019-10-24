FROM nvcr.io/nvidia/pytorch:18.08-py3
MAINTAINER Ashwin Nair

RUN apt-get update && apt-get install -y --no-install-recommends

RUN apt-get -y install libglib2.0-0 libxrender1 libsm6 libxext6

RUN pip --no-cache-dir install \
	Cython

RUN pip install --upgrade pip
COPY docker/dl_requirements.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/dl_requirements.txt


WORKDIR /workspace/Deeplabv3+
RUN ["/bin/bash"]
