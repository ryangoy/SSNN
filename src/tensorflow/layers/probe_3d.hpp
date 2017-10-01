#ifndef PROBE_3D_H_
#define PROBE_3D_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, 
  				  T* weights, T* num_strides, T* out);
};

#endif PROBE_3D_H_