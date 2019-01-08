FROM ubuntu:16.04

MAINTAINER Demarcus Briers (dbriers@bu.edu)

LABEL Description="Linux Environment for the publication "
LABEL tags="multicellular,biological model,pattern-synthesis, machine learning, optimization"
LABEL dependencies="python==3.7,Octave,Morpheus==1.9.2"

# Jupyter config based on https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile
ARG NB_USER="synth"
ARG NB_UID="1000"
ARG NB_GID="100"

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install Python3/pip3, and GNU Octave
RUN apt-get -y update && apt-get -y --no-install-recommends install make gcc gfortran g++ \
    libnetcdf-dev libxml2-dev libblas-dev liblapack-dev \
    libssl-dev pkg-config apt-transport-https python3 python3-pip octave ghostscript \
    wget \
    gnupg2 \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    curl \
    python3-venv && \
    apt-get -y clean && apt-get -y autoremove && rm -rf /var/lib/{cache,log}/ /tmp/* /var/tmp/*

# Pull in a tini init script
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini.asc /tini.asc
RUN gpg --batch --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 595E85A6B1B4779EA4DAAEC70B588DFF0527A9B7 \
 && gpg --batch --verify /tini.asc /tini
RUN chmod +x /tini

# Install nodejs which is a special pony and needs its own stuff
RUN curl -sSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add -
RUN echo "deb https://deb.nodesource.com/node_11.x xenial main" | tee /etc/apt/sources.list.d/nodesource.list
RUN echo "deb-src https://deb.nodesource.com/node_11.x xenial main" | tee -a /etc/apt/sources.list.d/nodesource.list
RUN apt-get update
RUN apt-get install -y nodejs

# install morpheus
ADD morpheus_1.9.2_amd64~b161102.deb /tmp/morpheus_1.9.2_amd64~b161102.deb
RUN apt-get install -y /tmp/morpheus_1.9.2_amd64~b161102.deb

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV SHELL=/bin/bash \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME=/home/$NB_USER

ADD fix-permissions /usr/local/bin/fix-permissions

RUN groupadd wheel -g 11 && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME

USER $NB_UID

# Setup work directory for backward-compatibility
RUN mkdir /home/$NB_USER/work && \
    fix-permissions /home/$NB_USER

# Install python3 modules
RUN python3 -m venv /home/$NB_USER/.venv
RUN /home/$NB_USER/.venv/bin/python3 -m pip install --upgrade pip setuptools && \
    /home/$NB_USER/.venv/bin/python3 -m pip install matplotlib==2.1.0 numpy==1.15.2 pandas==0.20.3 scikit-image==0.14.1 scipy==0.19.1 seaborn==0.8.1 'notebook==5.7.2' 'jupyterhub==0.9.4' 'jupyterlab==0.35.4' && \
    /home/$NB_USER/.venv/bin/jupyter labextension install @jupyterlab/hub-extension@^0.12.0 && \
    npm cache clean --force && \
    /home/$NB_USER/.venv/bin/jupyter notebook --generate-config && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions /home/$NB_USER

# files to container
# ADD runTest1.sh /usr/local/bin/runTest1.sh

# Switch back to root and start stuff
USER root

EXPOSE 8888
WORKDIR $HOME

# Configure container startup
ENTRYPOINT ["/tini", "-g", "--"]
CMD ["/home/synth/.venv/bin/jupyter", "lab"]

# Add local files as late as possible to avoid cache busting
COPY jupyter_notebook_config.py /etc/jupyter/
RUN fix-permissions /etc/jupyter/
RUN fix-permissions /home/$NB_USER

# Switch back to user to avoid accidental container runs as root
USER $NB_UID
