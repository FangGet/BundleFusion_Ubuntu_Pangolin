#!/bin/bash

# get parameter from system
user=`id -un`

# start sharing xhost
xhost +local:root

# run docker
docker run --rm \
  --ipc=host \
  --gpus all \
  --privileged \
  -p 3753:22 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:$docker/.Xauthority \
  -v $HOME/work:$HOME/work \
  -v /mnt/Data/Datasets/bundlefusion:/mnt/Data/Datasets/bundlefusion \
  -e http_proxy=http://10.141.6.84:7890 \
  -e https_proxy=http://10.141.6.84:7890 \
  -e XAUTHORITY=$home_folder/.Xauthority \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it 10.141.6.125:5111/midea/bundlefusion
