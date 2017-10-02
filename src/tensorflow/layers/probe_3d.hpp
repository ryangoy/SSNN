#ifndef PROBE_3D_H_
#define PROBE_3D_H_

template <typename Device, typename T>
struct Probe3DFunctor {
  void operator()(const Device& d, const T* input, 
  				  T* weights, T* dims, T* steps, T* output);
};

#endif PROBE_3D_H_