#syntax=docker/dockerfile:1.3
# The above comment is load bearing. Don't remove it!
# See https://abe-winter.github.io/caching/2020/05/26/runmount.html for why we're using this
# and https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md
# for details on this flag.

# Custom docker image based on https://github.com/graykode/ai-docstring/tree/master/docker
# SOURCE: https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
FROM continuumio/miniconda3:latest
# LABEL maintainer="nlkey2022@gmail.com"

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
&& DEBIAN_FRONTEND=noninteractive apt-get -qqy install curl git ca-certificates         libjpeg-dev         libpng-dev \
&& rm -rf /var/lib/apt/lists/*

ENV PYTORCH_WHEEL="torch==1.6.0"
ENV WEIGHT="1YrkwfM-0VBCJaa9NYaXUQPODdGPsmQY4"

RUN git clone https://github.com/graykode/ai-docstring.git /app/ai-docstring \
    && cd /app/ai-docstring/server

WORKDIR /app/ai-docstring/server

# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN /opt/conda/bin/conda install pip

RUN /opt/conda/bin/pip gdown torch torchvision

# RUN pip3 install gdown && \
    # pip3 install torch torchvision
# RUN gdown https://drive.google.com/uc?id=${WEIGHT}

# RUN pip install -r requirements.txt

# ENTRYPOINT ["python3", "app.py"]
