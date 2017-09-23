FROM nvidia/cuda:7.0-runtime-ubuntu14.04
MAINTAINER Varun Suresh <fab.varun@gmail.com>
# docker build -t dc/ctpn .
#docker run  --rm -it -v `pwd`:/opt/ctpn/CTPN/demo_images -p 8888:8888  dc/ctpn /bin/bash
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        zip \
        unzip \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CTPN_ROOT=/opt/ctpn
WORKDIR $CTPN_ROOT

RUN git clone --depth 1 https://github.com/tianzhi0549/CTPN.git
WORKDIR $CTPN_ROOT/CTPN/caffe



ADD ctpn_trained_model.caffemodel  $CTPN_ROOT/CTPN/models



# Missing "packaging" package
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  packaging

RUN cd python && for req in $(cat requirements.txt) pydot; do pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  $req; done && cd ..
RUN git clone https://github.com/NVIDIA/nccl.git
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda=7.0-28
WORKDIR /

# Download the CUDA drivers from https://developer.nvidia.com/rdp/cudnn-archive and place it here  :
ADD cudnn-7.0-linux-x64-v4.0-prod.tgz /

WORKDIR /cuda
RUN cp -P include/cudnn.h /usr/include
RUN cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/

WORKDIR $CTPN_ROOT/CTPN/caffe

ADD ./caffe/Makefile.config $CTPN_ROOT/CTPN/caffe
RUN cat Makefile.config
#RUN cp Makefile.config.example Makefile.config
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim
#RUN cd nccl && make -j install && 
#RUN    cd .. && rm -rf nccl && \
RUN   cd $CTPN_ROOT/CTPN/caffe  &&   \
   mkdir build && cd build && \
     cmake -DUSE_CUDNN=0 .. && \
#     WITH_PYTHON_LAYER=1 && \
     pwd && \
     make -j"$(nproc)" && make pycaffe

# Set the environment variables so that the paths are correctly configured
ENV PYCAFFE_ROOT $CTPN_ROOT/CTPN/caffe/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CTPN_ROOT/CTPN/caffe/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CTPN_ROOT/CTPN/caffe/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# To make sure the python layer builds - Need to figure out a cleaner way to do this.
RUN cp $CTPN_ROOT/CTPN/src/layers/* $CTPN_ROOT/CTPN/caffe/src/caffe/layers/
RUN cp $CTPN_ROOT/CTPN/src/*.py $CTPN_ROOT/CTPN/caffe/src/caffe/
RUN cp -r $CTPN_ROOT/CTPN/src/utils $CTPN_ROOT/CTPN/caffe/src/caffe/

# Install Opencv - 2.4.12 :

RUN cd ~ && \
    mkdir -p ocv-tmp && \
    cd ocv-tmp && \
    wget https://github.com/Itseez/opencv/archive/2.4.12.zip  && \
    unzip 2.4.12.zip && \
    cd opencv-2.4.12 && \
    mkdir release && \
    cd release && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_PYTHON_SUPPORT=ON \
          .. && \
    make -j8 && \
    make install && \
    rm -rf ~/ocv-tmp


RUN sudo ln /dev/null /dev/raw1394   
RUN  pip install numpy --upgrade
WORKDIR $CTPN_ROOT/CTPN
RUN make

RUN mkdir /opt/ctpn/CTPN/output
VOLUME ['/opt/ctpn/CTPN/output/']
RUN pip install --upgrade jupyter
RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
WORKDIR /opt/ctpn/CTPN/
EXPOSE 8888
CMD ["jupyter", "notebook", "--no-browser", "--allow-root"]


