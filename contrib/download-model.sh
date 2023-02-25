#!/usr/bin/env bash

set -x

mkdir -p ~/Documents/my_models || true

filename=~/Documents/my_models/ScreenNetV1.pth

if [ -e "filename" ]; then
    echo 'File already exists' >&2
else
    # curl -L 'https://www.dropbox.com/s/9903r4jy02rmuzh/ScreenCropNetV1_378_epochs.pth?dl=1' >~/Documents/my_models/ScreenCropNetV1_378_epochs.pth
    curl -L 'https://www.dropbox.com/s/im6ytahqgbpyjvw/ScreenNetV1.pth?dl=1' >~/Documents/my_models/ScreenNetV1.pth
fi

set +x
