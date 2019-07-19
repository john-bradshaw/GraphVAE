FROM ubuntu:16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion unzip gcc-5 g++-5

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh \
 && /bin/bash ~/anaconda.sh -b -p /opt/conda  \
 && rm ~/anaconda.sh  \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh  \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN conda create -y -n py36 python=3.6 anaconda \
 && echo "conda activate py36" >> ~/.bashrc \
 && /opt/conda/bin/conda install -n py36 -y pytorch-cpu=1.1.0  torchvision-cpu=0.3.0  -c pytorch \
 && /opt/conda/bin/conda install -n py36 -y rdkit=2019.03.3  -c rdkit


RUN /opt/conda/envs/py36/bin/pip install dataclasses arrow tensorboardX docopt arrow tqdm

COPY . /graph_vae
WORKDIR /graph_vae/

CMD ["bash", "./docker_run_script.sh"]