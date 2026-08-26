/* CUDA 11.8 lazy-loading probe. Compile against CUDA 11.8 headers/libs, not 11.4. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int print_maps(struct dl_phdr_info *info, size_t size, void *data) {
  (void)size;
  (void)data;
  if (info->dlpi_name &&
      (strstr(info->dlpi_name, "libcuda") ||
       strstr(info->dlpi_name, "libcudart"))) {
    printf("loaded %s\n", info->dlpi_name);
  }
  return 0;
}

static const char *mode_name(CUmoduleLoadingMode mode) {
  if (mode == CU_MODULE_LAZY_LOADING)
    return "LAZY";
  if (mode == CU_MODULE_EAGER_LOADING)
    return "EAGER";
  return "UNKNOWN";
}

int main(void) {
  const char *env = getenv("CUDA_MODULE_LOADING");
  printf("CUDA_MODULE_LOADING=%s\n", env ? env : "(unset)");
  dl_iterate_phdr(print_maps, NULL);

  CUresult cr = cuInit(0);
  if (cr != CUDA_SUCCESS) {
    const char *msg = NULL;
    cuGetErrorString(cr, &msg);
    fprintf(stderr, "cuInit failed: %s (%d)\n", msg ? msg : "?", (int)cr);
    return 1;
  }
  cudaError_t e = cudaFree(0);
  if (e != cudaSuccess) {
    fprintf(stderr, "cudaFree(0) failed: %s\n", cudaGetErrorString(e));
    return 1;
  }

  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  printf("CUDA Driver Version  = %d.%d (%d)\n", driver / 1000,
         (driver % 1000) / 10, driver);
  printf("CUDA Runtime Version = %d.%d (%d)\n", runtime / 1000,
         (runtime % 1000) / 10, runtime);

  CUmoduleLoadingMode mode = (CUmoduleLoadingMode)0;
  cr = cuModuleGetLoadingMode(&mode);
  if (cr != CUDA_SUCCESS) {
    const char *msg = NULL;
    cuGetErrorString(cr, &msg);
    fprintf(stderr, "cuModuleGetLoadingMode failed: %s (%d)\n",
            msg ? msg : "?", (int)cr);
    fprintf(stderr, "libcuda is too old or compat is not first on LD_LIBRARY_PATH\n");
    return 2;
  }
  printf("Module Loading Mode  = %s (%d)\n", mode_name(mode), (int)mode);

  int ok = (driver >= 11080) && (runtime >= 11080) &&
           (mode == CU_MODULE_LAZY_LOADING);
  printf("PROBE_OK=%s\n", ok ? "1" : "0");
  return ok ? 0 : 3;
}
