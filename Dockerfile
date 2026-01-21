FROM ubuntu:24.04

LABEL maintainer="dcc@de-alliantie.nl"

ARG homedir=/opt/verhuiskans
ARG azuredir=/.azure
ARG modeldir=/opt/verhuiskans/models

ARG OTAP="local"
ENV OTAP=${OTAP}

# Switch to root for changing dir ownership/permissions
USER 0

WORKDIR ${homedir}

# Azure ML does something within this dir -> make sure user 1001 has permissions
RUN mkdir ${azuredir}

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl

COPY ./requirements.txt .

RUN python3 -m venv venv

RUN venv/bin/pip install -r requirements.txt

COPY . .

RUN venv/bin/pip install -e .

CMD venv/bin/python src/main_predict.py