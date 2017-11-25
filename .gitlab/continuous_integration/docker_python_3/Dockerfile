FROM alpine
FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
    libgomp1 build-essential gfortran unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN conda create -n bat_ci -c conda-forge python=3 openturns=1.9 matplotlib numpy scipy scikit-learn pathos jsonschema sphinx sphinx_rtd_theme pytest pytest-runner mock ffmpeg pycodestyle pylint coverage && rm -rf /opt/conda/pkgs/*

RUN wget -O /tmp/z.$$ https://github.com/mathjax/MathJax/archive/master.zip && unzip -d /app /tmp/z.$$ && rm /tmp/z.$$

COPY . /app/

ENV PATH /opt/conda/envs/bat_ci/bin:$PATH
