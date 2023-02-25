FROM python:3.10.9-bullseye
RUN apt-get update && apt-get install -y \
  gcc build-essential cmake \
  && apt-get install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
  && apt install \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    pkg-config \
    texinfo \
    wget \
    yasm \
    zlib1g-dev -y \
        && \
    apt install libopus-dev libmp3lame-dev libfdk-aac-dev libvpx-dev libx264-dev yasm libass-dev libtheora-dev libvorbis-dev mercurial cmake build-essential -y \
        && \
    apt install nasm  -y && \
    apt install  libx264-dev  -y && \
    apt install  libx265-dev libnuma-dev  -y && \
    apt install  libvpx-dev  -y && \
    apt install  libfdk-aac-dev  -y && \
    apt install libmp3lame-dev -y && \
    apt install libopus-dev -y && \
    apt install  libaom-dev -y && \
    apt install -y libunistring-dev && \
    add-apt-repository -y ppa:nilarimogard/webupd8 && \
    apt-get update && \
    apt-get install youtube-dl -y && \
    apt install imagemagick -y && \
    apt install fdupes  libimage-exiftool-perl icu-devtools icu-doc libicu-dev -y && \
    apt-get install -y libxslt-dev perl doxygen && \
    apt-get install -y hdf5-tools && \
    apt-get install -y libjpeg-dev && \
    apt-get install -y libpng-dev && \
    apt-get install -y libtiff-dev && \
    apt-get install -y openexr libeigen3-dev libbtbb-dev libbtbb1  libmkl-tbb-thread libtbb-dev libtbb2 && \
    apt-get install python3-numpy libopenblas-dev -y && \
    apt-get install -y libomp-dev && \
    apt-get install -y openmpi-bin tcl-dev && \
    apt install sqlite3 -y && \
    apt-get install -y xz-utils && \
    apt install zlib1g-dev -y && \
    apt-get install -y libmagic-dev libffi-dev && \
    apt-get install -y atomicparsley tree && \
    add-apt-repository -y ppa:alex-p/tesseract-ocr5 && \
    apt-get update && \
    apt install -y tesseract-ocr nmap && \
    apt-get install -y libavcodec-dev libavcodec-extra libghc-sdl2-dev libsdl2-dev  libsdl2-image-dev libsndifsdl2-dev libsdl2-ttf-dev python3-sdl2 && \
    apt-get install -y libsdl2-mixer-2.0-0 && \
    apt-get install -y libsdl2-mixer-dev && \
    apt-get install -y python3-pydub && \
    apt-get install -y squishyball \
    libsamplerate0-dev \
    libsamplerate0 \
    ladspa-sdk \
    python3-bitstring \
    python3-eyed3 \
    python3-guidata \
    python3-pdfrw \
    python3-releases \
    python3-netaddr && \
    apt-get -y install \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    ninja-build \
    pkg-config && \
    apt install libunistring-dev libaom-dev -y && \
    apt-get install build-essential cmake git unzip pkg-config libopenblas-dev  liblapack-dev -y && \
    apt-get install python3-numpy python3-scipy python3-matplotlib -y && \
    apt-get install libhdf5-serial-dev python3-h5py -y && \
    apt-get install graphviz -y && \
    apt-get install python3-opencv -y && \
    pip install pydot-ng && \
    apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev -y  && \
    apt-get install -y aria2 \
    libaria2 \
    libaria2-0 \
    libaria2-0-dev && \
    apt install libopencv-dev python3-opencv -y \
    && rm -rf /var/lib/apt/lists/*


# RUN pip install poetry==1.2.2

# # Configuring poetry
# RUN poetry config virtualenvs.create false

# # Copying requirements of a project
# COPY pyproject.toml poetry.lock /app/src/
# WORKDIR /app/src

# # Installing requirements
# RUN poetry install
# # Removing gcc
# RUN apt-get purge -y \
#   gcc \
#   && rm -rf /var/lib/apt/lists/*

# # Copying actuall application
# COPY . /app/src/
# RUN poetry install

# CMD ["/usr/local/bin/python", "-m", "fastapi_pytorch_postgresql_sandbox"]
