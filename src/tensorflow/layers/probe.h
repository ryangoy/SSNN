#ifndef PROBE_H_
#define PROBE_H_

template <typename Device, typename T>
struct ProbeFunctor {
    void operator()(const Device& d, const int* sizes, const T* input, 
            const T* weights, const T* dims, const T* steps, T* output);
};

#endif // PROBE_H_