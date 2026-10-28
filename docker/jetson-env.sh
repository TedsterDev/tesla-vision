#!/bin/sh
# Link the host's cuDNN 9 where LD_LIBRARY_PATH (set in docker/Dockerfile,
# alongside the CUDA 12.6 mount) expects it, then run the service's command.
#
# docker-compose.yaml mounts /usr/local/cuda-12.6 at /usr/local/cuda and the
# host's /usr/lib/aarch64-linux-gnu at /host-lib. The latter can NOT go on
# LD_LIBRARY_PATH whole - it would shadow this image's own glibc-era libraries
# with the host's - so only cuDNN is linked out of it, into a directory of its
# own. Services without those mounts (gif_worker, poller) find nothing to link
# and run exactly as before.
set -e

if [ -d /host-lib ]; then
  mkdir -p /opt/host-cudnn
  for library in /host-lib/libcudnn*.so.9*; do
    [ -e "$library" ] && ln -sf "$library" /opt/host-cudnn/
  done
fi

exec "$@"
