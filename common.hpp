#ifndef COMMON_HPP
#define COMMON_HPP

#include <cuda_runtime_api.h>


constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val)
{
    return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val)
{
    return val * (1 << 10);
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(unsigned long long val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(unsigned long long val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(unsigned long long val)
{
    return val * (1 << 10);
}

#define CUDA_CHECK(status)                                                   \
  do {                                                                       \
    auto ret = (status);                                                     \
    if (ret != 0) {                                                          \
      std::cerr << "Cuda failure: " << cudaGetErrorString(ret) << std::endl; \
      abort();                                                               \
    }                                                                        \
  } while (0)

#define CUDNN_CHECK(call)                                                         \
  do {                                                                            \
    cudnnStatus_t status = call;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << "CUDNN failure: " << cudnnGetErrorString(status) << std::endl; \
      abort();                                                                    \
    }                                                                             \
  } while (0)

#endif // COMMON_HPP