#include <torch/extension.h>

#include <cuda_runtime_api.h>
#include <cstddef>

#define EXPORT extern "C"

EXPORT void* managed_malloc(size_t size, int device, void* stream) {
  (void)stream;
  int cur = -1;
  cudaGetDevice(&cur);
  if (device != cur && device >= 0) cudaSetDevice(device);

  // cudaMallocManaged allows for more memory to be allocated than the device memory size.
  // The cudaMemAttachGlobal flag makes the memory accessible from both host and device.
  void* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, (size_t)size, cudaMemAttachGlobal);
  if (err != cudaSuccess) return nullptr;

  if (device >= 0) {
    // cudaMemAdviseSetPreferredLocation sets the preferred location for the memory.
    // This is a hint that tries to prevent data from being migrated away from the device.
    cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetPreferredLocation, device);
    // cudaMemAdviseSetAccessedBy ensures the memory always lives in the device's page table.
    // Even if the memory has to be migrated away from the device, it still does not page fault.
    // The CUDA docs claim that cudaMemAdviseSetPreferredLocation completely overrides this flag,
    // but there is no harm in adding this flag as well for future-proofing.
    cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetAccessedBy, device);
  }
  return ptr;
}

EXPORT void managed_free(void* ptr, size_t size, int device, void* stream) {
  // Memory allocated with cudaMallocManaged should be released with cudaFree.
  (void)size; (void)device; (void)stream;
  if (ptr) cudaFree(ptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}