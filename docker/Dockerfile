# FROM jacob-baxter

# Loading the main docker files
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04
# FROM cuda-9.0-local


# install packages
RUN apt-get update && apt-get install -q -y \
  dirmngr \
  gnupg2 \
  lsb-release \
  && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install ros packages
ENV ROS_DISTRO kinetic

RUN apt-get update && apt-get install -y \
  ros-kinetic-ros-core=1.3.2-0* \
  && rm -rf /var/lib/apt/lists/*

# Install packages for Baxter base
RUN apt-get update && apt-get install -y \
  git-core \
  python-argparse \
  python-wstool \
  python-vcstools \
  python-rosdep \
  ros-kinetic-control-msgs \
  vim

# Install packages for gazebo
RUN apt-get update && apt-get install -y \
  gazebo7 \
  ros-kinetic-qt-build \
  ros-kinetic-gazebo-ros-control \
  ros-kinetic-gazebo-ros-pkgs \
  ros-kinetic-ros-control \
  ros-kinetic-control-toolbox \
  ros-kinetic-realtime-tools \
  ros-kinetic-ros-controllers \
  ros-kinetic-xacro \
  ros-kinetic-tf-conversions \
  ros-kinetic-kdl-parser \
  && rm -rf /var/lib/apt/lists/*

# Install packages for Pointcloud generator (PCL)
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key 19274DEF \
&& echo "deb http://ppa.launchpad.net/v-launchpad-jochen-sprickerhof-de/pcl/ubuntu maverick main" >> /etc/apt/sources.list \
&& apt-get update \
&& apt-get install openni-dev=1.3.2.1-4+precise2 ps-engine=5.0.3.3-3+precise1 libpcl-all 


# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y \
  python-pip \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
  ros-kinetic-robot-state-publisher \
  && rm -rf /var/lib/apt/lists/*

# Install Catkin
RUN  pip install rospkg
RUN  pip install -U catkin_tools

# bootstrap rosdep
RUN rosdep init \
  && rosdep update

RUN apt-get update && apt-get install -y \
  ros-kinetic-gazebo-ros-pkgs \
  ros-kinetic-gazebo-ros-control \
  && rm -rf /var/lib/apt/lists/*

### taken from moveit source install Dockerfile (https://github.com/ros-planning/moveit/blob/kinetic-devel/.docker/source/Dockerfile)
# Replacing shell with bash for later docker build commands
RUN mv /bin/sh /bin/sh-old && \
  ln -s /bin/bash /bin/sh

# create ROS ENV
ENV CATKIN_WS=/root/catkin_ws
RUN source /opt/ros/kinetic/setup.bash
RUN mkdir -p $CATKIN_WS/src
WORKDIR ${CATKIN_WS}
RUN catkin init
RUN catkin config --extend /opt/ros/$ROS_DISTRO --cmake-args -DCMAKE_BUILD_TYPE=Release
WORKDIR $CATKIN_WS/src

# Download the baxter core files
RUN wstool init . \
  && wstool merge https://raw.githubusercontent.com/RethinkRobotics/baxter/master/baxter_sdk.rosinstall \
  && wstool update

WORKDIR $CATKIN_WS
ENV PYTHONIOENCODING UTF-8
RUN catkin build
RUN source devel/setup.bash

WORKDIR ${CATKIN_WS}/src

# Download the simulation files
RUN git clone -b kinetic-devel https://github.com/RethinkRobotics/baxter_simulator.git # get the regular simulator

WORKDIR $CATKIN_WS
RUN catkin build
RUN source devel/setup.bash

WORKDIR ${CATKIN_WS}/src

# # Download and install moveit source
# # RUN wstool init . && \
# RUN wstool merge https://raw.githubusercontent.com/ros-planning/moveit/${ROS_DISTRO}-devel/moveit.rosinstall && \
#   wstool update

# RUN apt-get -qq update && \
#   apt-get -qq install -y \
#   wget && \
#   rosdep update && \
#   rosdep install -y --from-paths . --ignore-src --rosdistro ${ROS_DISTRO} && \
#   rm -rf /var/lib/apt/lists/*

# Install MoveIt! the regular way
RUN apt-get -qq update && apt-get -qq install -y ros-kinetic-moveit && \
    rm -rf /var/lib/apt/lists/*


# Build repo
WORKDIR $CATKIN_WS
RUN catkin build
RUN source devel/setup.bash

WORKDIR ${CATKIN_WS}/src

# Download and install Baxter MoveIt
RUN git clone https://github.com/ros-planning/moveit_robots.git

RUN pip install --upgrade setuptools
RUN pip install matplotlib==2.0.2 pyassimp==4.1.3
RUN pip install cython
RUN pip install quadprog nltk pypcd
RUN pip install numpy --upgrade

WORKDIR ${CATKIN_WS}
RUN catkin build
RUN source devel/setup.bash

################## trying for CUDA 9.0 ##################################
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
  rm -rf /var/lib/apt/lists/* && \
  NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
  NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
  apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
  echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
  echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 9.0.176

ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
  cuda-cudart-$CUDA_PKG_VERSION && \
  ln -s cuda-9.0 /usr/local/cuda && \
  rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
  echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

# Adding TF dependencies
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cuda-command-line-tools-9-0 \
  cuda-cublas-9-0 \
  cuda-cufft-9-0 \
  cuda-curand-9-0 \
  cuda-cusolver-9-0 \
  cuda-cusparse-9-0 \
  libcudnn7=7.2.1.38-1+cuda9.0 \
  libnccl2=2.2.13-1+cuda9.0 \
  libfreetype6-dev \
  libhdf5-serial-dev \
  libpng12-dev \
  libzmq3-dev \
  pkg-config \
  software-properties-common \
  unzip \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
  apt-get update && \
  apt-get install libnvinfer4=4.1.2-1+cuda9.0

# Adding CUBLAS
# See : https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile
ENV NCCL_VERSION 2.4.2

RUN apt-get update && apt-get install -y --no-install-recommends \
  cuda-libraries-$CUDA_PKG_VERSION \
  cuda-cublas-9-0=9.0.176.4-1 \
  libnccl2=$NCCL_VERSION-1+cuda9.0 && \
  apt-mark hold libnccl2 && \
  rm -rf /var/lib/apt/lists/*


# Adding CUDNN
# See : https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/cudnn7/Dockerfile
# ENV CUDNN_VERSION 7.4.2.24
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#   libcudnn7=$CUDNN_VERSION-1+cuda9.0 && \
#   apt-mark hold libcudnn7 && \
#   rm -rf /var/lib/apt/lists/*

COPY cudnn/*.deb /opt/
RUN dpkg -i /opt/libcudnn7_7.6.0.64-1+cuda9.0_amd64.deb \
  && dpkg -i /opt/libcudnn7-dev_7.6.0.64-1+cuda9.0_amd64.deb \
  && cp /usr/include/cudnn.h /usr/lib/x86_64-linux-gnu/ \
  && rm -f /opt/libcudnn*.deb

# maybe this is necessary...
RUN export CUDA_NVRTC_LIB="/usr/local/cuda-9.0/lib64/"
RUN cp /usr/local/cuda-9.0/lib64/libcusparse.so.9.0 /usr/local/cuda-9.0/lib64/libcusparse.so && \
  cp /usr/local/cuda-9.0/lib64/libcurand.so.9.0 /usr/local/cuda-9.0/lib64/libcurand.so && \
  cp /usr/local/cuda-9.0/lib64/libnvrtc.so.9.0 /usr/local/cuda-9.0/lib64/libnvrtc.so && \
  cp /usr/local/cuda-9.0/lib64/libcublas.so.9.0 /usr/local/cuda-9.0/lib64/libcublas.so && \
  cp /usr/local/cuda-9.0/lib64/libcufft.so.9.0 /usr/local/cuda-9.0/lib64/libcufft.so

# dev versions give the header files that pytorch source wants to see during build
RUN apt-get update && apt-get install -y \
  cuda-cusparse-dev-9-0 \
  cuda-nvrtc-dev-9-0 \
  cuda-cublas-dev-9-0 \
  cuda-cufft-dev-9-0 \
  cuda-curand-dev-9-0 \
  cuda-cusolver-dev-9-0 && \
  rm -rf /var/lib/apt/lists/*

#########################################################################

# run below for CUDA 8.0
# RUN pip install https://download.pytorch.org/whl/cu80/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl
# RUN pip install torchvision

# run below for CUDA 9.0 PyTorch
RUN pip install torch torchvision

# copy in yaml file to load rrt* as default planner
COPY ./ompl_planning.yaml ${CATKIN_WS}/src/moveit_robots/baxter/baxter_moveit_config/config/


# Exposing the ports
EXPOSE 11311

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
  ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
  ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
