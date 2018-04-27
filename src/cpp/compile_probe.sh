TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
nvcc -std=c++11 -c -o probe.cu.o probe.cu.cc -D_FORCE_INLINES -O2 -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_30 --expt-relaxed-constexpr
g++ -std=c++11 -shared -o probe.so probe.cc probe.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -L /usr/local/cuda-9.1/lib64/ -O2 -L$TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0
