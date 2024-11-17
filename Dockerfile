FROM ubuntu:latest

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install \
    git \
    unzip \
    curl \
    gcc \
    libz-dev \
    libconfig8-dev \
    libarchive-dev \
    automake \
    autoconf \
    libtool

RUN git clone https://github.com/rieck/sally && cd sally

RUN ./bootstrap && \
    ./configure && \
    make && \
    make check \
    make install
