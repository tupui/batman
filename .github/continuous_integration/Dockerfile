FROM alpine
FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
    libgomp1 build-essential \
    texlive-latex-base texlive-latex-extra dvipng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN conda create -n bat_ci -c conda-forge python=3 openturns matplotlib numpy pandas scipy scikit-learn pathos jsonschema paramiko sphinx sphinx_rtd_theme pytest pytest-runner mock ffmpeg pycodestyle pylint coverage && rm -rf /opt/conda/pkgs/*

COPY Dockerfile /app/

ENV PATH /opt/conda/envs/bat_ci/bin:$PATH
