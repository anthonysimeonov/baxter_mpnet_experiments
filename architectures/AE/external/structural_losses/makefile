nvcc = /usr/local/cuda-9.0/bin/nvcc
cudalib =  /usr/local/cuda-9.0/lib64
tensorflow = /home/arclabdl1/anaconda3/envs/ylmiao_tf/lib/python2.7/site-packages/tensorflow/include
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L$TF_LIB  -L /home/arclabdl1/anaconda3/envs/ylmiao_tf/lib/python2.7/site-packages/tensorflow -ltensorflow_framework


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -I$TF_INC/external/nsync/public -L$TF_LIB  -L /home/arclabdl1/anaconda3/envs/ylmiao_tf/lib/python2.7/site-packages/tensorflow -ltensorflow_framework


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L$TF_LIB  -L /home/arclabdl1/anaconda3/envs/ylmiao_tf/lib/python2.7/site-packages/tensorflow -ltensorflow_framework


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -I$TF_INC/external/nsync/public -L$TF_LIB  -L /home/arclabdl1/anaconda3/envs/ylmiao_tf/lib/python2.7/site-packages/tensorflow -ltensorflow_framework


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o
