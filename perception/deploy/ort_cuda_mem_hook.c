/*
 * LD_PRELOAD hook for sherpa-onnx 1.13.6 (ORT 1.16.0).
 *
 * sherpa only calls CUDA EP V1 (device_id + Heuristic). Defaults are:
 *   gpu_mem_limit = SIZE_MAX
 *   arena_extend_strategy = kNextPowerOfTwo
 *   cudnn_conv_use_max_workspace = 1   (V2 default; can add GBs)
 *
 * Matcha is two sessions (acoustic + vocos). Each gets its own CUDA arena.
 * This intercepts OrtGetApiBase and rewrites AppendExecutionProvider_CUDA
 * to CUDA EP V2 with JP5-friendly knobs.
 *
 * Slots below are OrtApi 1.16.0 function-pointer indices (aarch64).
 *
 * Env:
 *   TTS_ORT_GPU_MEM_LIMIT_MB     default 256  (per session arena cap)
 *   TTS_ORT_ARENA_EXTEND         default kSameAsRequested
 *   TTS_ORT_CUDNN_MAX_WORKSPACE  default 0
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ORT_API_SLOTS 264
#define IDX_CUDA_V1 150
#define IDX_CUDA_V2 202
#define IDX_CREATE_CUDA 203
#define IDX_UPDATE_CUDA 204
#define IDX_RELEASE_CUDA 206

typedef struct OrtStatus OrtStatus;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtCUDAProviderOptionsV2 OrtCUDAProviderOptionsV2;

typedef struct {
  const void *(*GetApi)(uint32_t);
  const char *(*GetVersionString)(void);
} OrtApiBase;

typedef OrtStatus *(*fn_cuda_v1)(OrtSessionOptions *, const void *);
typedef OrtStatus *(*fn_cuda_v2)(OrtSessionOptions *,
                                const OrtCUDAProviderOptionsV2 *);
typedef OrtStatus *(*fn_create_cuda)(OrtCUDAProviderOptionsV2 **);
typedef OrtStatus *(*fn_update_cuda)(OrtCUDAProviderOptionsV2 *,
                                     const char *const *, const char *const *,
                                     size_t);
typedef void (*fn_release_cuda)(OrtCUDAProviderOptionsV2 *);

static const OrtApiBase *(*real_OrtGetApiBase)(void);
static const void *(*real_GetApi)(uint32_t);
static OrtApiBase g_base;
static void *g_api_copy[ORT_API_SLOTS];
static fn_cuda_v1 orig_cuda_v1;
static fn_cuda_v2 orig_cuda_v2;
static fn_create_cuda orig_create_cuda;
static fn_update_cuda orig_update_cuda;
static fn_release_cuda orig_release_cuda;

static long env_long(const char *key, long def) {
  const char *v = getenv(key);
  if (!v || !v[0])
    return def;
  char *end = NULL;
  long n = strtol(v, &end, 10);
  if (end == v)
    return def;
  return n;
}

static const char *env_str(const char *key, const char *def) {
  const char *v = getenv(key);
  if (!v || !v[0])
    return def;
  return v;
}

static OrtStatus *append_cuda_v2(OrtSessionOptions *so) {
  if (!orig_create_cuda || !orig_update_cuda || !orig_cuda_v2 ||
      !orig_release_cuda) {
    fprintf(stderr,
            "[ort_cuda_hook] FATAL: CUDA V2 slots empty (not ORT 1.16?)\n");
    if (orig_cuda_v1)
      return orig_cuda_v1(so, NULL);
    return NULL;
  }

  long mb = env_long("TTS_ORT_GPU_MEM_LIMIT_MB", 256);
  if (mb <= 0)
    mb = 256;
  const char *arena = env_str("TTS_ORT_ARENA_EXTEND", "kSameAsRequested");
  const char *ws = env_str("TTS_ORT_CUDNN_MAX_WORKSPACE", "0");

  char limit[32];
  snprintf(limit, sizeof(limit), "%ld", mb * 1024L * 1024L);

  OrtCUDAProviderOptionsV2 *v2 = NULL;
  OrtStatus *st = orig_create_cuda(&v2);
  if (st)
    return st;

  const char *keys[] = {
      "device_id",
      "gpu_mem_limit",
      "arena_extend_strategy",
      "cudnn_conv_algo_search",
      "cudnn_conv_use_max_workspace",
      "do_copy_in_default_stream",
  };
  const char *vals[] = {"0", limit, arena, "HEURISTIC", ws, "1"};
  st = orig_update_cuda(v2, keys, vals, 6);
  if (st) {
    orig_release_cuda(v2);
    return st;
  }
  st = orig_cuda_v2(so, v2);
  orig_release_cuda(v2);
  fprintf(stderr,
          "[ort_cuda_hook] CUDA V2 gpu_mem_limit_mb=%ld workspace=%s arena=%s\n",
          mb, ws, arena);
  return st;
}

static OrtStatus *hooked_cuda_v1(OrtSessionOptions *so, const void *ignored) {
  (void)ignored;
  return append_cuda_v2(so);
}

static OrtStatus *hooked_cuda_v2(OrtSessionOptions *so,
                                const OrtCUDAProviderOptionsV2 *ignored) {
  (void)ignored;
  return append_cuda_v2(so);
}

static const void *hooked_GetApi(uint32_t version) {
  const void *real = real_GetApi(version);
  if (!real)
    return real;
  memcpy(g_api_copy, real, sizeof(g_api_copy));
  orig_cuda_v1 = (fn_cuda_v1)g_api_copy[IDX_CUDA_V1];
  orig_cuda_v2 = (fn_cuda_v2)g_api_copy[IDX_CUDA_V2];
  orig_create_cuda = (fn_create_cuda)g_api_copy[IDX_CREATE_CUDA];
  orig_update_cuda = (fn_update_cuda)g_api_copy[IDX_UPDATE_CUDA];
  orig_release_cuda = (fn_release_cuda)g_api_copy[IDX_RELEASE_CUDA];
  g_api_copy[IDX_CUDA_V1] = (void *)hooked_cuda_v1;
  g_api_copy[IDX_CUDA_V2] = (void *)hooked_cuda_v2;
  return g_api_copy;
}

const OrtApiBase *OrtGetApiBase(void) {
  if (!real_OrtGetApiBase) {
    real_OrtGetApiBase =
        (const OrtApiBase *(*)(void))dlsym(RTLD_NEXT, "OrtGetApiBase");
    if (!real_OrtGetApiBase) {
      fprintf(stderr, "[ort_cuda_hook] dlsym OrtGetApiBase failed: %s\n",
              dlerror());
      return NULL;
    }
  }
  const OrtApiBase *real = real_OrtGetApiBase();
  if (!real)
    return real;
  real_GetApi = real->GetApi;
  g_base.GetApi = hooked_GetApi;
  g_base.GetVersionString = real->GetVersionString;
  return &g_base;
}

__attribute__((constructor)) static void hook_init(void) {
  fprintf(stderr, "[ort_cuda_hook] loaded (ORT 1.16 CUDA V2 mem knobs)\n");
}
