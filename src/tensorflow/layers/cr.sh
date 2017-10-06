TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
nvcc probe.cu.cc -o probe.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 probe.cc probe.cu.o -o probe.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
