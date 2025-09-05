FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime


# Install utils
RUN apt update && apt install -y \
  tmux \
  htop \
  nano

RUN pip install --upgrade pip

# Install required packages

RUN pip install fastprogress
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install tqdm
RUN pip install tensorboard
RUN pip install scikit-image
RUN pip install easydict
RUN pip install tensorboardX
RUN pip install torchinfo
RUN pip install argconfigparse
RUN pip install torchmetrics
RUN pip install seaborn
RUN pip install hydra-core
RUN pip install shapely
RUN pip install geopandas
RUN pip install xarray
RUN pip install rasterio

# get the instructions from here:
# https://docs.docker.com/engine/reference/builder/
ARG HOST_UID
RUN useradd user -u $HOST_UID

#https://stackoverflow.com/questions/45553074/cannot-create-directory-permission-denied-inside-docker-container
RUN mkdir -p /home/user
RUN chown user /home/user

ENV workdir /home/user
WORKDIR ${workdir}

USER user

CMD bash
