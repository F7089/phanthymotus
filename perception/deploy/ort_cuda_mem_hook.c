/*
 * LD_PRELOAD hook for sherpa-onnx 1.13.6 (ORT 1.16.0).
 *
 * 1) All CUDA EP V1 appends become CUDA V2 with JP5 arena knobs.
 * 2) Vocos CreateSession (after CUDA append) uses TensorRT EP + CUDA
 *    fallback, with engine cache. Acoustic stays CUDA. GetModelType()
 *    probes vocos on CPU (no CUDA append) and is left alone.
 *
 * OrtApi 1.16.0 slot indices (aarch64, counted from onnxruntime_c_api.h).
 *
 * Env:
 *   TTS_VOCOS_TRT=1                 default on
 *   TTS_VOCOS_TRT_CACHE             default /opt/vocos_trt_cache
 *   TTS_ORT_GPU_MEM_LIMIT_MB        default 256
 *   TTS_ORT_ARENA_EXTEND            default kSameAsRequested
 *   TTS_ORT_CUDNN_MAX_WORKSPACE     default 0
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ORT_API_SLOTS 261
#define IDX_CREATE_SESSION 4
#define IDX_CREATE_SESSION_OPTS 7
#define IDX_SET_INTRA 21
#define IDX_SET_INTER 22
#define IDX_CUDA_V1 149
#define IDX_TRT_V2 167
#define IDX_CREATE_TRT 168
#define IDX_UPDATE_TRT 169
#define IDX_RELEASE_TRT 171
#define IDX_CUDA_V2 201
#define IDX_CREATE_CUDA 202
#define IDX_UPDATE_CUDA 203
#define IDX_RELEASE_CUDA 205

typedef struct OrtStatus OrtStatus;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtCUDAProviderOptionsV2 OrtCUDAProviderOptionsV2;
typedef struct OrtTensorRTProviderOptionsV2 OrtTensorRTProviderOptionsV2;

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
typedef OrtStatus *(*fn_create_session)(const void *, const char *,
                                        const OrtSessionOptions *, void **);
typedef OrtStatus *(*fn_create_session_opts)(OrtSessionOptions **);
typedef OrtStatus *(*fn_set_threads)(OrtSessionOptions *, int);
typedef OrtStatus *(*fn_trt_v2)(OrtSessionOptions *,
                                const OrtTensorRTProviderOptionsV2 *);
typedef OrtStatus *(*fn_create_trt)(OrtTensorRTProviderOptionsV2 **);
typedef OrtStatus *(*fn_update_trt)(OrtTensorRTProviderOptionsV2 *,
                                    const char *const *, const char *const *,
                                    size_t);
typedef void (*fn_release_trt)(OrtTensorRTProviderOptionsV2 *);

static const OrtApiBase *(*real_OrtGetApiBase)(void);
static const void *(*real_GetApi)(uint32_t);
static OrtApiBase g_base;
static void *g_api_copy[ORT_API_SLOTS];
static fn_cuda_v1 orig_cuda_v1;
static fn_cuda_v2 orig_cuda_v2;
static fn_create_cuda orig_create_cuda;
static fn_update_cuda orig_update_cuda;
static fn_release_cuda orig_release_cuda;
static fn_create_session orig_create_session;
static fn_create_session_opts orig_create_session_opts;
static fn_set_threads orig_set_intra;
static fn_set_threads orig_set_inter;
static fn_trt_v2 orig_trt_v2;
static fn_create_trt orig_create_trt;
static fn_update_trt orig_update_trt;
static fn_release_trt orig_release_trt;
static int g_after_cuda_append;

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

static int env_on(const char *key, int def) {
  const char *v = getenv(key);
  if (!v || !v[0])
    return def;
  return !(v[0] == '0' && v[1] == 0);
}

static int path_is_vocos(const char *p) {
  const char *s;
  if (!p)
    return 0;
  for (s = p; *s; s++) {
    if ((s[0] == 'v' || s[0] == 'V') && (s[1] == 'o' || s[1] == 'O') &&
        (s[2] == 'c' || s[2] == 'C') && (s[3] == 'o' || s[3] == 'O') &&
        (s[4] == 's' || s[4] == 'S'))
      return 1;
  }
  return 0;
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

static OrtStatus *append_trt_v2(OrtSessionOptions *so) {
  if (!orig_create_trt || !orig_update_trt || !orig_trt_v2 || !orig_release_trt) {
    fprintf(stderr, "[ort_cuda_hook] TRT V2 slots empty; vocos stays CUDA\n");
    return NULL;
  }
  const char *cache = env_str("TTS_VOCOS_TRT_CACHE", "/opt/vocos_trt_cache");
  OrtTensorRTProviderOptionsV2 *trt = NULL;
  OrtStatus *st = orig_create_trt(&trt);
  if (st)
    return st;
  const char *keys[] = {
      "device_id",
      "trt_max_workspace_size",
      "trt_fp16_enable",
      "trt_engine_cache_enable",
      "trt_engine_cache_path",
      "trt_min_subgraph_size",
  };
  const char *vals[] = {"0", "268435456", "1", "1", cache, "5"};
  st = orig_update_trt(trt, keys, vals, 6);
  if (st) {
    orig_release_trt(trt);
    return st;
  }
  st = orig_trt_v2(so, trt);
  orig_release_trt(trt);
  fprintf(stderr, "[ort_cuda_hook] vocos TensorRT cache=%s fp16=1 workspace=256MB\n",
          cache);
  return st;
}

static OrtStatus *hooked_cuda_v1(OrtSessionOptions *so, const void *ignored) {
  (void)ignored;
  g_after_cuda_append = 1;
  return append_cuda_v2(so);
}

static OrtStatus *hooked_cuda_v2(OrtSessionOptions *so,
                                const OrtCUDAProviderOptionsV2 *ignored) {
  (void)ignored;
  g_after_cuda_append = 1;
  return append_cuda_v2(so);
}

static OrtStatus *hooked_create_session(const void *env, const char *path,
                                        const OrtSessionOptions *opts,
                                        void **out) {
  fprintf(stderr, "[ort_cuda_hook] CreateSession path=%s\n", path ? path : "");
  g_after_cuda_append = 0;
  if (!path_is_vocos(path) || !env_on("TTS_VOCOS_TRT", 1) ||
      !orig_create_session_opts || !orig_create_session)
    return orig_create_session(env, path, opts, out);

  OrtSessionOptions *so = NULL;
  OrtStatus *st = orig_create_session_opts(&so);
  if (st)
    return orig_create_session(env, path, opts, out);
  if (orig_set_intra)
    orig_set_intra(so, 2);
  if (orig_set_inter)
    orig_set_inter(so, 1);

  st = append_trt_v2(so);
  if (st) {
    fprintf(stderr, "[ort_cuda_hook] vocos TRT append failed; using CUDA only\n");
    append_cuda_v2(so);
  } else {
    append_cuda_v2(so);
  }
  fprintf(stderr, "[ort_cuda_hook] vocos TensorRT session %s\n", path ? path : "");
  return orig_create_session(env, path, so, out);
}

static int find_ort_cb(struct dl_phdr_info *info, size_t size, void *data) {
  (void)size;
  if (info->dlpi_name && strstr(info->dlpi_name, "libonnxruntime.so")) {
    *(const char **)data = info->dlpi_name;
    return 1;
  }
  return 0;
}

static void *load_ort_handle(void) {
  const char *path = NULL;
  static const char *cands[] = {
      "libonnxruntime.so.1.16.0",
      "libonnxruntime.so.1",
      "/usr/local/lib/python3.8/dist-packages/sherpa_onnx/lib/"
      "libonnxruntime.so.1.16.0",
      "/usr/local/lib/python3.10/dist-packages/sherpa_onnx/lib/"
      "libonnxruntime.so.1.16.0",
      NULL,
  };
  int i;
  void *h;
  dl_iterate_phdr(find_ort_cb, &path);
  if (path && path[0]) {
    h = dlopen(path, RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    if (h)
      return h;
    h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (h)
      return h;
  }
  for (i = 0; cands[i]; i++) {
    h = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL);
    if (h)
      return h;
  }
  return NULL;
}

static const void *hooked_GetApi(uint32_t version) {
  const void *real = real_GetApi(version);
  if (!real)
    return real;
  memcpy(g_api_copy, real, sizeof(g_api_copy));
  orig_create_session = (fn_create_session)g_api_copy[IDX_CREATE_SESSION];
  orig_create_session_opts =
      (fn_create_session_opts)g_api_copy[IDX_CREATE_SESSION_OPTS];
  orig_set_intra = (fn_set_threads)g_api_copy[IDX_SET_INTRA];
  orig_set_inter = (fn_set_threads)g_api_copy[IDX_SET_INTER];
  orig_cuda_v1 = (fn_cuda_v1)g_api_copy[IDX_CUDA_V1];
  orig_cuda_v2 = (fn_cuda_v2)g_api_copy[IDX_CUDA_V2];
  orig_create_cuda = (fn_create_cuda)g_api_copy[IDX_CREATE_CUDA];
  orig_update_cuda = (fn_update_cuda)g_api_copy[IDX_UPDATE_CUDA];
  orig_release_cuda = (fn_release_cuda)g_api_copy[IDX_RELEASE_CUDA];
  orig_trt_v2 = (fn_trt_v2)g_api_copy[IDX_TRT_V2];
  orig_create_trt = (fn_create_trt)g_api_copy[IDX_CREATE_TRT];
  orig_update_trt = (fn_update_trt)g_api_copy[IDX_UPDATE_TRT];
  orig_release_trt = (fn_release_trt)g_api_copy[IDX_RELEASE_TRT];
  g_api_copy[IDX_CUDA_V1] = (void *)hooked_cuda_v1;
  g_api_copy[IDX_CUDA_V2] = (void *)hooked_cuda_v2;
  g_api_copy[IDX_CREATE_SESSION] = (void *)hooked_create_session;
  fprintf(stderr,
          "[ort_cuda_hook] GetApi patched ver=%u create_session orig=%p\n",
          version, (void *)orig_create_session);
  return g_api_copy;
}

const OrtApiBase *OrtGetApiBase(void) {
  if (!real_OrtGetApiBase) {
    void *h = load_ort_handle();
    if (!h) {
      fprintf(stderr, "[ort_cuda_hook] dlopen libonnxruntime failed: %s\n",
              dlerror());
      return NULL;
    }
    real_OrtGetApiBase =
        (const OrtApiBase *(*)(void))dlsym(h, "OrtGetApiBase");
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
  fprintf(stderr, "[ort_cuda_hook] loaded (CUDA V2 knobs + vocos TRT)\n");
}
