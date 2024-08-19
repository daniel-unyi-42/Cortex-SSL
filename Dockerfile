FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    tmux \
    vim \
    htop \
    openssh-server \
    zip \
    unzip \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu117.html \
    torch_geometric

RUN pip install --no-cache-dir \
    debugpy \
    pytest \
    tensorboardX \
    matplotlib \
    seaborn \
    pandas \
    nibabel

RUN useradd -m -s /bin/bash daniel

WORKDIR /home/daniel

USER daniel
