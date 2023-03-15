xhost local:docker

docker run -v ${DATASETS}:/app/data \
--rm --gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw  \
--privileged \
--device /dev/dri \
--net=host \
--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
-it \
bundlefusion-cu10.0-cudagl:latest ./build/bundle_fusion_example ./zParametersDefault.txt ./zParametersBundlingDefault.txt data

#./build/bundle_fusion_example ./zParametersDefault.txt ./zParametersBundlingDefault.txt data
