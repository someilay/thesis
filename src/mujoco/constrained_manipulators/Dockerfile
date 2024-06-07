# Build
FROM ubuntu:latest as build_and_run
LABEL authors="ilia-milioshin", name="build constarained manipulators"

ARG MUJOCO_VERSION=3.1.0

# Upgrade and install building tools
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt install build-essential -y && \
    apt install wget -y
RUN apt-get install -y cmake

# Install eigen3
RUN apt install libeigen3-dev -y

# Install pinocchio
RUN apt install -qqy lsb-release curl -y
RUN mkdir -p /etc/apt/keyrings
RUN curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | tee /etc/apt/keyrings/robotpkg.asc
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list
RUN apt update -y && \
    apt install -qqy robotpkg-py3*-pinocchio

# Install GLFW
RUN apt-get install libglfw3 -y && \
    apt-get install libglfw3-dev -y

# Install mujoco
RUN wget -O mujoco.tar.gz https://github.com/google-deepmind/mujoco/releases/download/3.1.0/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz && \
    tar -xvf mujoco.tar.gz && \
    mv mujoco-${MUJOCO_VERSION} mujoco

# Install pkg-config
RUN apt install pkg-config -y

# Install xauth
RUN apt install -y xauth

# Install inicpp
RUN wget https://github.com/SemaiCZE/inicpp/archive/refs/tags/v1.2.2.tar.gz && \
    tar -xvzf v1.2.2.tar.gz && \
    mkdir inicpp-build && \
    cmake -S inicpp-1.2.2 -B inicpp-build -DINICPP_BUILD_SHARED=ON && \
    make -C inicpp-build && \
    make -C inicpp-build install

RUN groupadd -r sample && useradd -r -g sample sample
USER sample

WORKDIR /home/sample

# Copy files
COPY main.cpp CMakeLists.txt config.ini urdf_manipulators.urdf ./
COPY mjcf_manipulators/ ./mjcf_manipulators/

# Build project
RUN cmake -B build -S . -DMUJOCO_DIR=/mujoco && \
    cmake --build build --config Release --target constrained_manipulators

ENTRYPOINT ["./build/constrained_manipulators"]
