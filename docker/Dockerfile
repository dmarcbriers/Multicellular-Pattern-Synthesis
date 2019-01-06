FROM ubuntu:16.04

MAINTAINER Demarcus Briers (dbriers@bu.edu)

LABEL Description="Linux Environment for the publication "
LABEL tags="multicellular,biological model,pattern-synthesis, machine learning, optimization"
LABEL dependencies="python==3.7,Octave,Morpheus==1.9.2"

ENV DEBIAN_FRONTEND=noninteractive

# Install Python3/pip3, and GNU Octave
RUN apt-get -y update && apt-get -y --no-install-recommends install make gcc gfortran g++ \
libnetcdf-dev libxml2-dev libblas-dev liblapack-dev \
libssl-dev pkg-config apt-transport-https python3 python3-pip octave ghostscript && \
    apt-get -y clean && apt-get -y autoremove && rm -rf /var/lib/{cache,log}/ /tmp/* /var/tmp/*


# Install python3 modules
RUN python3 -m pip install --upgrade pip setuptools && python3 -m pip install matplotlib==2.1.0 numpy==1.15.2 pandas==0.20.3 scikit-image==0.14.1 scipy==0.19.1 seaborn==0.8.1

# install morpheus
ADD morpheus_1.9.2_amd64~b161102.deb /tmp/morpheus_1.9.2_amd64~b161102.deb
RUN apt install -y /tmp/morpheus_1.9.2_amd64~b161102.deb

# files to container
# ADD runTest1.sh /usr/local/bin/runTest1.sh