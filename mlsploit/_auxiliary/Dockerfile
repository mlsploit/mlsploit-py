# MLsploit Containers v0.0.0

########## Base Image ##########

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 AS mlsploit-base

VOLUME /mnt/input
VOLUME /mnt/output

RUN apt-get -y -qq update \
    && apt-get install -y wget vim build-essential git curl

# Install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace


########## TF 1.15 ##########
FROM mlsploit-base AS mlsploit-tf1
RUN conda install tensorflow-gpu==1.15.0


########## TF 2.1 ##########
FROM mlsploit-base AS mlsploit-tf2
RUN pip install tensorflow==2.1.0


########## PyTorch 1.4 ##########
FROM mlsploit-base AS mlsploit-pytorch

RUN conda install \
        pytorch==1.4 \
        torchvision==0.5.0 \
        cudatoolkit=10.1 -c pytorch
