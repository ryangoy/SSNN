#ifndef PROBE_3D_H_
#define PROBE_3D_H_

template <typename Device, typename T>
struct Probe3DFunctor {
  void operator()(const Device& d, const T* input, 
  				  const T* weights, const T* dims, const T* steps, T* output);
};

#endif // PROBE_3D_H_