# /bin/bash

xhost +local:
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/dri/card0:/dev/dri/card0 --network host udwadia
