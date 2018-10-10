#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
/* Crash and burn. */

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

/* Some simple utilities for wall-clock timing.

   The function get_wall_time() returns the wall time in microseconds
   (with an unspecified offset).
*/

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  lock = lock;
}

#endif

/* The simple OpenCL runtime framework used by Futhark. */

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define OPENCL_SUCCEED_FATAL(e) opencl_succeed_fatal(e, #e, __FILE__, __LINE__)
#define OPENCL_SUCCEED_NONFATAL(e) opencl_succeed_nonfatal(e, #e, __FILE__, __LINE__)
// Take care not to override an existing error.
#define OPENCL_SUCCEED_OR_RETURN(e) {             \
    char *error = OPENCL_SUCCEED_NONFATAL(e);     \
    if (error) {                                  \
      if (!ctx->error) {                          \
        ctx->error = error;                       \
        return 1;                                 \
      } else {                                    \
        free(error);                              \
      }                                           \
    }                                             \
  }

struct opencl_config {
  int debugging;
  int logging;
  int preferred_device_num;
  const char *preferred_platform;
  const char *preferred_device;

  const char* dump_program_to;
  const char* load_program_from;

  size_t default_group_size;
  size_t default_num_groups;
  size_t default_tile_size;
  size_t default_threshold;
  size_t transpose_block_dim;

  int default_group_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  size_t *size_values;
  const char **size_classes;
  const char **size_entry_points;
};

void opencl_config_init(struct opencl_config *cfg,
                        int num_sizes,
                        const char *size_names[],
                        size_t *size_values,
                        const char *size_classes[],
                        const char *size_entry_points[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_platform = "";
  cfg->preferred_device = "";
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;

  cfg->default_group_size = 256;
  cfg->default_num_groups = 128;
  cfg->default_tile_size = 32;
  cfg->default_threshold = 32*1024;
  cfg->transpose_block_dim = 16;

  cfg->default_group_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
  cfg->size_entry_points = size_entry_points;
}

/* An entry in the free list.  May be invalid, to avoid having to
   deallocate entries as soon as they are removed.  There is also a
   tag, to help with memory reuse. */
struct opencl_free_list_entry {
  size_t size;
  cl_mem mem;
  const char *tag;
  unsigned char valid;
};

struct opencl_free_list {
  struct opencl_free_list_entry *entries; // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

void free_list_init(struct opencl_free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = malloc(sizeof(struct opencl_free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

/* Remove invalid entries from the free list. */
void free_list_pack(struct opencl_free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }
  // Now p == l->used.
  l->entries = realloc(l->entries, l->used * sizeof(struct opencl_free_list_entry));
  l->capacity = l->used;
}

void free_list_destroy(struct opencl_free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

int free_list_find_invalid(struct opencl_free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

void free_list_insert(struct opencl_free_list *l, size_t size, cl_mem mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct opencl_free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

/* Find and remove a memory block of at least the desired size and
   tag.  Returns 0 on success.  */
int free_list_find(struct opencl_free_list *l, const char *tag, size_t *size_out, cl_mem *mem_out) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid && l->entries[i].tag == tag) {
      l->entries[i].valid = 0;
      *size_out = l->entries[i].size;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

/* Remove the first block in the free list.  Returns 0 if a block was
   removed, and nonzero if the free list was already empty. */
int free_list_first(struct opencl_free_list *l, cl_mem *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

struct opencl_context {
  cl_device_id device;
  cl_context ctx;
  cl_command_queue queue;

  struct opencl_config cfg;

  struct opencl_free_list free_list;

  size_t max_group_size;
  size_t max_num_groups;
  size_t max_tile_size;
  size_t max_threshold;

  size_t lockstep_width;
};

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_context*, struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

static const char* opencl_error_string(unsigned int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed_fatal(unsigned int ret,
                                 const char *call,
                                 const char *file,
                                 int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_succeed_nonfatal(unsigned int ret,
                                     const char *call,
                                     const char *file,
                                     int line) {
  if (ret != CL_SUCCESS) {
    return msgprintf("%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
                     file, line, call, ret, opencl_error_string(ret));
  } else {
    return NULL;
  }
}

void set_preferred_platform(struct opencl_config *cfg, const char *s) {
  cfg->preferred_platform = s;
}

void set_preferred_device(struct opencl_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED_FATAL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED_FATAL(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

static int is_blacklisted(const char *platform_name, const char *device_name,
                          const struct opencl_config *cfg) {
  if (strcmp(cfg->preferred_platform, "") != 0 ||
      strcmp(cfg->preferred_device, "") != 0) {
    return 0;
  } else if (strstr(platform_name, "Apple") != NULL &&
             strstr(device_name, "Intel(R) Core(TM)") != NULL) {
    return 1;
  } else {
    return 0;
  }
}

static struct opencl_device_option get_preferred_device(const struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (!is_blacklisted(device.platform_name, device.device_name, cfg) &&
        strstr(device.platform_name, cfg->preferred_platform) != NULL &&
        strstr(device.device_name, cfg->preferred_device) != NULL &&
        num_device_matches++ == cfg->preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.\n");
  exit(1); // Never reached
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int ret_val = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (ret_val != CL_SUCCESS && ret_val != CL_BUILD_PROGRAM_FAILURE) {
    assert(ret_val == 0);
  }

  cl_build_status build_status;
  ret_val = clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_STATUS,
                                  sizeof(cl_build_status),
                                  &build_status,
                                  NULL);
  assert(ret_val == 0);

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    ret_val = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    assert(ret_val == 0);

    build_log = malloc(ret_val_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    assert(ret_val == 0);

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

/* Fields in a bitmask indicating which types we must be sure are
   available. */
enum opencl_required_type { OPENCL_F64 = 1 };

// We take as input several strings representing the program, because
// C does not guarantee that the compiler supports particularly large
// literals.  Notably, Visual C has a limit of 2048 characters.  The
// array must be NULL-terminated.
static cl_program setup_opencl_with_command_queue(struct opencl_context *ctx,
                                                  cl_command_queue queue,
                                                  const char *srcs[],
                                                  int required_types) {
  int error;

  ctx->queue = queue;

  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx->ctx, NULL));

  // Fill out the device info.  This is redundant work if we are
  // called from setup_opencl() (which is the common case), but I
  // doubt it matters much.
  struct opencl_device_option device_option;
  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_DEVICE,
                                       sizeof(cl_device_id),
                                       &device_option.device,
                                       NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id),
                                 &device_option.platform,
                                 NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type),
                                 &device_option.device_type,
                                 NULL));
  device_option.platform_name = opencl_platform_info(device_option.platform, CL_PLATFORM_NAME);
  device_option.device_name = opencl_device_info(device_option.device, CL_DEVICE_NAME);

  ctx->device = device_option.device;

  if (required_types & OPENCL_F64) {
    cl_uint supported;
    OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                   sizeof(cl_uint), &supported, NULL));
    if (!supported) {
      panic(1, "Program uses double-precision floats, but this is not supported on the chosen device: %s",
            device_option.device_name);
    }
  }

  size_t max_group_size;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  if (max_group_size < ctx->cfg.default_group_size) {
    if (ctx->cfg.default_group_size_changed) {
      fprintf(stderr, "Note: Device limits default group size to %zu (down from %zu).\n",
              max_group_size, ctx->cfg.default_group_size);
    }
    ctx->cfg.default_group_size = max_group_size;
  }

  if (max_tile_size < ctx->cfg.default_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr, "Note: Device limits default tile size to %zu (down from %zu).\n",
              max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = max_tile_size;
  }

  ctx->max_group_size = max_group_size;
  ctx->max_tile_size = max_tile_size; // No limit.
  ctx->max_threshold = ctx->max_num_groups = 0; // No limit.

  // Now we go through all the sizes, clamp them to the valid range,
  // or set them to the default.
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    size_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    size_t max_value, default_value;
    if (strstr(size_class, "group_size") == size_class) {
      max_value = max_group_size;
      default_value = ctx->cfg.default_group_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = max_group_size; // Futhark assumes this constraint.
      default_value = ctx->cfg.default_num_groups;
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = sqrt(max_group_size);
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      max_value = 0; // No limit.
      default_value = ctx->cfg.default_threshold;
    } else {
      panic(1, "Unknown size class for size '%s': %s\n", size_name, size_class);
    }
    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %d (down from %d)\n",
              size_name, (int)max_value, (int)*size_value);
      *size_value = max_value;
    }
  }

  // Make sure this function is defined.
  post_opencl_setup(ctx, &device_option);

  if (ctx->cfg.logging) {
    fprintf(stderr, "Lockstep width: %d\n", (int)ctx->lockstep_width);
    fprintf(stderr, "Default group size: %d\n", (int)ctx->cfg.default_group_size);
    fprintf(stderr, "Default number of groups: %d\n", (int)ctx->cfg.default_num_groups);
  }

  char *fut_opencl_src = NULL;
  size_t src_size = 0;

  // Maybe we have to read OpenCL source from somewhere else (used for debugging).
  if (ctx->cfg.load_program_from != NULL) {
    FILE *f = fopen(ctx->cfg.load_program_from, "r");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    src_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fut_opencl_src = malloc(src_size);
    assert(fread(fut_opencl_src, 1, src_size, f) == src_size);
    fclose(f);
  } else {
    // Build the OpenCL program.  First we have to concatenate all the fragments.
    for (const char **src = srcs; src && *src; src++) {
      src_size += strlen(*src);
    }

    fut_opencl_src = malloc(src_size + 1);

    size_t n, i;
    for (i = 0, n = 0; srcs && srcs[i]; i++) {
      strncpy(fut_opencl_src+n, srcs[i], src_size-n);
      n += strlen(srcs[i]);
    }
    fut_opencl_src[src_size] = 0;

  }

  cl_program prog;
  error = 0;
  const char* src_ptr[] = {fut_opencl_src};

  if (ctx->cfg.dump_program_to != NULL) {
    FILE *f = fopen(ctx->cfg.dump_program_to, "w");
    assert(f != NULL);
    fputs(fut_opencl_src, f);
    fclose(f);
  }

  prog = clCreateProgramWithSource(ctx->ctx, 1, src_ptr, &src_size, &error);
  assert(error == 0);

  int compile_opts_size = 1024;
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    compile_opts_size += strlen(ctx->cfg.size_names[i]) + 20;
  }
  char *compile_opts = malloc(compile_opts_size);

  int w = snprintf(compile_opts, compile_opts_size,
                   "-DFUT_BLOCK_DIM=%d -DLOCKSTEP_WIDTH=%d ",
                   (int)ctx->cfg.transpose_block_dim,
                   (int)ctx->lockstep_width);

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    w += snprintf(compile_opts+w, compile_opts_size-w,
                  "-D%s=%d ", ctx->cfg.size_names[i],
                  (int)ctx->cfg.size_values[i]);
  }

  OPENCL_SUCCEED_FATAL(build_opencl_program(prog, device_option.device, compile_opts));
  free(compile_opts);
  free(fut_opencl_src);

  return prog;
}

static cl_program setup_opencl(struct opencl_context *ctx,
                               const char *srcs[],
                               int required_types) {

  ctx->lockstep_width = 1;

  free_list_init(&ctx->free_list);

  struct opencl_device_option device_option = get_preferred_device(&ctx->cfg);

  if (ctx->cfg.logging) {
    describe_device_option(device_option);
  }

  // Note that NVIDIA's OpenCL requires the platform property
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)device_option.platform,
    0
  };

  cl_int error;

  ctx->ctx = clCreateContext(properties, 1, &device_option.device, NULL, NULL, &error);
  assert(error == 0);

  cl_command_queue queue = clCreateCommandQueue(ctx->ctx, device_option.device, 0, &error);
  assert(error == 0);

  return setup_opencl_with_command_queue(ctx, queue, srcs, required_types);
}

// Allocate memory from driver. The problem is that OpenCL may perform
// lazy allocation, so we cannot know whether an allocation succeeded
// until the first time we try to use it.  Hence we immediately
// perform a write to see if the allocation succeeded.  This is slow,
// but the assumption is that this operation will be rare (most things
// will go through the free list).
int opencl_alloc_actual(struct opencl_context *ctx, size_t size, cl_mem *mem_out) {
  int error;
  *mem_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size, NULL, &error);

  if (error != CL_SUCCESS) {
    return error;
  }

  int x = 2;
  error = clEnqueueWriteBuffer(ctx->queue, *mem_out, 1, 0, sizeof(x), &x, 0, NULL, NULL);

  // No need to wait for completion here. clWaitForEvents() cannot
  // return mem object allocation failures. This implies that the
  // buffer is faulted onto the device on enqueue. (Observation by
  // Andreas Kloeckner.)

  return error;
}

int opencl_alloc(struct opencl_context *ctx, size_t min_size, const char *tag, cl_mem *mem_out) {
  assert(min_size >= 0);
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;

  if (free_list_find(&ctx->free_list, tag, &size, mem_out) == 0) {
    // Successfully found a free block.  Is it big enough?
    //
    // FIXME: we might also want to check whether the block is *too
    // big*, to avoid internal fragmentation.  However, this can
    // sharply impact performance on programs where arrays change size
    // frequently.  Fortunately, such allocations are usually fairly
    // short-lived, as they are necessarily within a loop, so the risk
    // of internal fragmentation resulting in an OOM situation is
    // limited.  However, it would be preferable if we could go back
    // and *shrink* oversize allocations when we encounter an OOM
    // condition.  That is technically feasible, since we do not
    // expose OpenCL pointer values directly to the application, but
    // instead rely on a level of indirection.
    if (size >= min_size) {
      return CL_SUCCESS;
    } else {
      // Not just right - free it.
      int error = clReleaseMemObject(*mem_out);
      if (error != CL_SUCCESS) {
        return error;
      }
    }
  }

  // We have to allocate a new block from the driver.  If the
  // allocation does not succeed, then we might be in an out-of-memory
  // situation.  We now start freeing things from the free list until
  // we think we have freed enough that the allocation will succeed.
  // Since we don't know how far the allocation is from fitting, we
  // have to check after every deallocation.  This might be pretty
  // expensive.  Let's hope that this case is hit rarely.

  int error = opencl_alloc_actual(ctx, min_size, mem_out);

  while (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    cl_mem mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      error = clReleaseMemObject(mem);
      if (error != CL_SUCCESS) {
        return error;
      }
    } else {
      break;
    }
    error = opencl_alloc_actual(ctx, min_size, mem_out);
  }

  return error;
}

int opencl_free(struct opencl_context *ctx, cl_mem mem, const char *tag) {
  size_t size;
  cl_mem existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, tag, &size, &existing_mem) == 0) {
    int error = clReleaseMemObject(existing_mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  int error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

  if (error == CL_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return error;
}

int opencl_free_all(struct opencl_context *ctx) {
  cl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    int error = clReleaseMemObject(mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  return CL_SUCCESS;
}

const char *opencl_program[] =
           {"#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable\n__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n",
            "{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    ret",
            "urn x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint",
            "8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8",
            "_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int8_t sext_i8_i8(int8_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i8_i16(int8_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i8_i32(int8_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i8_i64(int8_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i16_i8(int16_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i16_i16(int16_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i16_i32(int16_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i16_i64(int16_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i32_i8(int32_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i32_i16(int32_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i32_i32(int32_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i32_i64(int32_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i64_i8(int64_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i64_i16(int64_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i64_i32(int64_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i64_i64(int64_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i8_i8(uint8_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i8_i16(uint8_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i8_i32(uint8_",
            "t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i8_i64(uint8_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i16_i8(uint16_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i16_i16(uint16_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i16_i32(uint16_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i16_i64(uint16_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i32_i8(uint32_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i32_i16(uint32_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i32_i32(uint32_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i32_i64(uint32_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i64_i8(uint64_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i64_i16(uint64_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i64_i32(uint64_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i64_i64(uint64_t x)\n{\n    return x;\n}\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return x < y ? x : y;\n}\nstatic inline float fmax32(float x, float y)\n{\n    return x < y ? y : x;\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    ret",
            "urn x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#define group_sizze_4849 (group_size_4848)\n#define max_num_groups_4851 (max_num_groups_4850)\n__kernel void chunked_reduce_kernel_4870(__local volatile\n                       ",
            "                  int64_t *mem_aligned_0,\n                                         __local volatile\n                                         int64_t *mem_aligned_1,\n                                         __local volatile\n                                         int64_t *mem_aligned_2,\n                                         __local volatile\n                                         int64_t *mem_aligned_3,\n                                         __local volatile\n                                         int64_t *mem_aligned_4,\n                                         __local volatile\n                                         int64_t *mem_aligned_5,\n                                         int32_t sizze_4756,\n                                         unsigned char cond_4759,\n                                         unsigned char x_4760,\n                                         unsigned char cond_4762,\n                                         unsigned char cond_4763,\n                                         unsigned char x_4764,\n                                         unsigned char x_4765,\n                                         int32_t num_threads_4857,\n                                         int32_t per_thread_elements_4860,\n                                         int32_t per_chunk_5109, __global\n                                         unsigned char *mem_5137, __global\n                                         unsigned char *mem_5158, __global\n                                         unsigned char *mem_5161, __global\n                                         unsigned char *mem_5164, __global\n                                         unsigned char *mem_5167, __global\n                                         unsigned char *mem_5170, __global\n                                         unsigned char *mem_5173)\n{\n    __local volatile char *restrict mem_5140 = mem_aligned_0;\n    __local volatile char *restrict mem_5143 = mem_aligned_1;\n    __local volatile char *restrict m",
            "em_5146 = mem_aligned_2;\n    __local volatile char *restrict mem_5149 = mem_aligned_3;\n    __local volatile char *restrict mem_5152 = mem_aligned_4;\n    __local volatile char *restrict mem_5155 = mem_aligned_5;\n    int32_t wave_sizze_5214;\n    int32_t group_sizze_5215;\n    bool thread_active_5216;\n    int32_t global_tid_4870;\n    int32_t local_tid_4871;\n    int32_t group_id_4872;\n    \n    global_tid_4870 = get_global_id(0);\n    local_tid_4871 = get_local_id(0);\n    group_sizze_5215 = get_local_size(0);\n    wave_sizze_5214 = LOCKSTEP_WIDTH;\n    group_id_4872 = get_group_id(0);\n    thread_active_5216 = 1;\n    \n    int32_t chunk_sizze_4898;\n    int32_t starting_point_5217 = global_tid_4870 * per_thread_elements_4860;\n    int32_t remaining_elements_5218 = sizze_4756 - starting_point_5217;\n    \n    if (sle32(remaining_elements_5218, 0) || sle32(sizze_4756,\n                                                   starting_point_5217)) {\n        chunk_sizze_4898 = 0;\n    } else {\n        if (slt32(sizze_4756, (global_tid_4870 + 1) *\n                  per_thread_elements_4860)) {\n            chunk_sizze_4898 = sizze_4756 - global_tid_4870 *\n                per_thread_elements_4860;\n        } else {\n            chunk_sizze_4898 = per_thread_elements_4860;\n        }\n    }\n    \n    int32_t slice_offset_4899;\n    int32_t res_4908;\n    int32_t res_4909;\n    int32_t res_4910;\n    int32_t res_4911;\n    int32_t res_4912;\n    int32_t res_4913;\n    \n    if (thread_active_5216) {\n        slice_offset_4899 = per_thread_elements_4860 * global_tid_4870;\n        \n        int32_t acc_4916;\n        int32_t acc_4917;\n        int32_t acc_4918;\n        int32_t acc_4919;\n        int32_t acc_4920;\n        int32_t acc_4921;\n        \n        acc_4916 = 0;\n        acc_4917 = 0;\n        acc_4918 = 0;\n        acc_4919 = 0;\n        acc_4920 = 0;\n        acc_4921 = 0;\n        for (int32_t i_4915 = 0; i_4915 < chunk_sizze_4898; i_4915++) {\n            int32_t j_p_i_t_s_5121 = slice_offset_4899 + i_4915;\n     ",
            "       int32_t new_index_5122 = squot32(j_p_i_t_s_5121, per_chunk_5109);\n            int32_t binop_y_5124 = per_chunk_5109 * new_index_5122;\n            int32_t new_index_5125 = j_p_i_t_s_5121 - binop_y_5124;\n            int32_t x_4924 = *(__global int32_t *) &mem_5137[(new_index_5125 *\n                                                              num_threads_4857 +\n                                                              new_index_5122) *\n                                                             4];\n            bool res_4932 = x_4924 == 0;\n            bool x_4933 = cond_4759 && res_4932;\n            bool res_4934 = x_4760 || x_4933;\n            int32_t res_4935;\n            \n            if (res_4934) {\n                res_4935 = 1;\n            } else {\n                res_4935 = 0;\n            }\n            \n            bool cond_4942 = acc_4919 == 0;\n            bool cond_4947 = acc_4921 == 0;\n            bool x_4949 = res_4932 && cond_4947;\n            bool res_4950 = sle32(acc_4921, x_4924);\n            bool res_4951 = acc_4921 == x_4924;\n            bool x_4952 = cond_4763 && res_4951;\n            bool res_4953 = x_4764 || x_4952;\n            bool x_4954 = cond_4762 && res_4950;\n            bool y_4955 = x_4765 && res_4953;\n            bool res_4956 = x_4954 || y_4955;\n            bool x_4957 = cond_4759 && x_4949;\n            bool y_4958 = x_4760 && res_4956;\n            bool res_4959 = x_4957 || y_4958;\n            bool x_4960 = !cond_4942;\n            bool y_4961 = res_4959 && x_4960;\n            bool res_4962 = cond_4942 || y_4961;\n            int32_t res_4963;\n            \n            if (res_4962) {\n                int32_t arg_4964;\n                int32_t res_4965;\n                int32_t res_4966;\n                \n                arg_4964 = acc_4918 + res_4935;\n                res_4965 = smax32(acc_4916, arg_4964);\n                res_4966 = smax32(res_4935, res_4965);\n                res_4963 = res_4966;\n            } else {\n                int",
            "32_t res_4967 = smax32(acc_4916, res_4935);\n                \n                res_4963 = res_4967;\n            }\n            \n            int32_t res_4968;\n            \n            if (cond_4942) {\n                res_4968 = res_4935;\n            } else {\n                bool cond_4969;\n                bool x_4970;\n                int32_t res_4971;\n                \n                cond_4969 = acc_4919 == acc_4917;\n                x_4970 = res_4962 && cond_4969;\n                if (x_4970) {\n                    int32_t res_4972 = acc_4917 + res_4935;\n                    \n                    res_4971 = res_4972;\n                } else {\n                    res_4971 = acc_4917;\n                }\n                res_4968 = res_4971;\n            }\n            \n            bool x_4975 = res_4934 && res_4962;\n            int32_t res_4976;\n            \n            if (x_4975) {\n                int32_t res_4977 = acc_4918 + res_4935;\n                \n                res_4976 = res_4977;\n            } else {\n                res_4976 = res_4935;\n            }\n            \n            int32_t res_4978 = 1 + acc_4919;\n            int32_t res_4979;\n            \n            if (cond_4942) {\n                res_4979 = x_4924;\n            } else {\n                res_4979 = acc_4920;\n            }\n            \n            int32_t acc_tmp_5219 = res_4963;\n            int32_t acc_tmp_5220 = res_4968;\n            int32_t acc_tmp_5221 = res_4976;\n            int32_t acc_tmp_5222 = res_4978;\n            int32_t acc_tmp_5223 = res_4979;\n            int32_t acc_tmp_5224;\n            \n            acc_tmp_5224 = x_4924;\n            acc_4916 = acc_tmp_5219;\n            acc_4917 = acc_tmp_5220;\n            acc_4918 = acc_tmp_5221;\n            acc_4919 = acc_tmp_5222;\n            acc_4920 = acc_tmp_5223;\n            acc_4921 = acc_tmp_5224;\n        }\n        res_4908 = acc_4916;\n        res_4909 = acc_4917;\n        res_4910 = acc_4918;\n        res_4911 = acc_4919;\n        res_4912 = acc_4920;\n  ",
            "      res_4913 = acc_4921;\n    }\n    \n    int32_t final_result_4993;\n    int32_t final_result_4994;\n    int32_t final_result_4995;\n    int32_t final_result_4996;\n    int32_t final_result_4997;\n    int32_t final_result_4998;\n    \n    for (int32_t comb_iter_5225 = 0; comb_iter_5225 < squot32(group_sizze_4849 +\n                                                              group_sizze_4849 -\n                                                              1,\n                                                              group_sizze_4849);\n         comb_iter_5225++) {\n        int32_t combine_id_4886;\n        int32_t flat_comb_id_5226 = comb_iter_5225 * group_sizze_4849 +\n                local_tid_4871;\n        \n        combine_id_4886 = flat_comb_id_5226;\n        if (slt32(combine_id_4886, group_sizze_4849) && 1) {\n            *(__local int32_t *) &mem_5140[combine_id_4886 * 4] = res_4908;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int32_t comb_iter_5227 = 0; comb_iter_5227 < squot32(group_sizze_4849 +\n                                                              group_sizze_4849 -\n                                                              1,\n                                                              group_sizze_4849);\n         comb_iter_5227++) {\n        int32_t combine_id_4887;\n        int32_t flat_comb_id_5228 = comb_iter_5227 * group_sizze_4849 +\n                local_tid_4871;\n        \n        combine_id_4887 = flat_comb_id_5228;\n        if (slt32(combine_id_4887, group_sizze_4849) && 1) {\n            *(__local int32_t *) &mem_5143[combine_id_4887 * 4] = res_4909;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int32_t comb_iter_5229 = 0; comb_iter_5229 < squot32(group_sizze_4849 +\n                                                              group_sizze_4849 -\n                                                              1,\n                                                              group_sizze_4849);\n         comb_iter_5229++) {\n    ",
            "    int32_t combine_id_4888;\n        int32_t flat_comb_id_5230 = comb_iter_5229 * group_sizze_4849 +\n                local_tid_4871;\n        \n        combine_id_4888 = flat_comb_id_5230;\n        if (slt32(combine_id_4888, group_sizze_4849) && 1) {\n            *(__local int32_t *) &mem_5146[combine_id_4888 * 4] = res_4910;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int32_t comb_iter_5231 = 0; comb_iter_5231 < squot32(group_sizze_4849 +\n                                                              group_sizze_4849 -\n                                                              1,\n                                                              group_sizze_4849);\n         comb_iter_5231++) {\n        int32_t combine_id_4889;\n        int32_t flat_comb_id_5232 = comb_iter_5231 * group_sizze_4849 +\n                local_tid_4871;\n        \n        combine_id_4889 = flat_comb_id_5232;\n        if (slt32(combine_id_4889, group_sizze_4849) && 1) {\n            *(__local int32_t *) &mem_5149[combine_id_4889 * 4] = res_4911;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int32_t comb_iter_5233 = 0; comb_iter_5233 < squot32(group_sizze_4849 +\n                                                              group_sizze_4849 -\n                                                              1,\n                                                              group_sizze_4849);\n         comb_iter_5233++) {\n        int32_t combine_id_4890;\n        int32_t flat_comb_id_5234 = comb_iter_5233 * group_sizze_4849 +\n                local_tid_4871;\n        \n        combine_id_4890 = flat_comb_id_5234;\n        if (slt32(combine_id_4890, group_sizze_4849) && 1) {\n            *(__local int32_t *) &mem_5152[combine_id_4890 * 4] = res_4912;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int32_t comb_iter_5235 = 0; comb_iter_5235 < squot32(group_sizze_4849 +\n                                                              group_sizze_4849 -\n                                  ",
            "                            1,\n                                                              group_sizze_4849);\n         comb_iter_5235++) {\n        int32_t combine_id_4891;\n        int32_t flat_comb_id_5236 = comb_iter_5235 * group_sizze_4849 +\n                local_tid_4871;\n        \n        combine_id_4891 = flat_comb_id_5236;\n        if (slt32(combine_id_4891, group_sizze_4849) && 1) {\n            *(__local int32_t *) &mem_5155[combine_id_4891 * 4] = res_4913;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t offset_5238;\n    int32_t skip_waves_5237;\n    int32_t my_index_4999;\n    int32_t other_index_5000;\n    int32_t x_5001;\n    int32_t x_5002;\n    int32_t x_5003;\n    int32_t x_5004;\n    int32_t x_5005;\n    int32_t x_5006;\n    int32_t x_5007;\n    int32_t x_5008;\n    int32_t x_5009;\n    int32_t x_5010;\n    int32_t x_5011;\n    int32_t x_5012;\n    \n    my_index_4999 = local_tid_4871;\n    offset_5238 = 0;\n    other_index_5000 = local_tid_4871 + offset_5238;\n    if (slt32(local_tid_4871, group_sizze_4849)) {\n        x_5001 = *(__local int32_t *) &mem_5140[(local_tid_4871 + offset_5238) *\n                                                4];\n        x_5002 = *(__local int32_t *) &mem_5143[(local_tid_4871 + offset_5238) *\n                                                4];\n        x_5003 = *(__local int32_t *) &mem_5146[(local_tid_4871 + offset_5238) *\n                                                4];\n        x_5004 = *(__local int32_t *) &mem_5149[(local_tid_4871 + offset_5238) *\n                                                4];\n        x_5005 = *(__local int32_t *) &mem_5152[(local_tid_4871 + offset_5238) *\n                                                4];\n        x_5006 = *(__local int32_t *) &mem_5155[(local_tid_4871 + offset_5238) *\n                                                4];\n    }\n    offset_5238 = 1;\n    other_index_5000 = local_tid_4871 + offset_5238;\n    while (slt32(offset_5238, wave_sizze_5214)) {\n        if (slt32(other_index_5",
            "000, group_sizze_4849) && ((local_tid_4871 -\n                                                           squot32(local_tid_4871,\n                                                                   wave_sizze_5214) *\n                                                           wave_sizze_5214) &\n                                                          (2 * offset_5238 -\n                                                           1)) == 0) {\n            // read array element\n            {\n                x_5007 = *(volatile __local\n                           int32_t *) &mem_5140[(local_tid_4871 + offset_5238) *\n                                                4];\n                x_5008 = *(volatile __local\n                           int32_t *) &mem_5143[(local_tid_4871 + offset_5238) *\n                                                4];\n                x_5009 = *(volatile __local\n                           int32_t *) &mem_5146[(local_tid_4871 + offset_5238) *\n                                                4];\n                x_5010 = *(volatile __local\n                           int32_t *) &mem_5149[(local_tid_4871 + offset_5238) *\n                                                4];\n                x_5011 = *(volatile __local\n                           int32_t *) &mem_5152[(local_tid_4871 + offset_5238) *\n                                                4];\n                x_5012 = *(volatile __local\n                           int32_t *) &mem_5155[(local_tid_4871 + offset_5238) *\n                                                4];\n            }\n            \n            bool cond_5013;\n            bool res_5014;\n            bool x_5015;\n            bool y_5016;\n            bool cond_5017;\n            bool cond_5018;\n            bool res_5019;\n            bool x_5020;\n            bool res_5021;\n            bool res_5022;\n            bool x_5023;\n            bool res_5024;\n            bool x_5025;\n            bool y_5026;\n            bool res_5027;\n            bool x_5028;\n  ",
            "          bool y_5029;\n            bool res_5030;\n            bool x_5031;\n            bool y_5032;\n            bool res_5033;\n            int32_t res_5034;\n            int32_t res_5039;\n            int32_t res_5044;\n            int32_t res_5049;\n            int32_t res_5050;\n            int32_t res_5051;\n            \n            if (thread_active_5216) {\n                cond_5013 = x_5004 == 0;\n                res_5014 = x_5010 == 0;\n                x_5015 = !cond_5013;\n                y_5016 = res_5014 && x_5015;\n                cond_5017 = cond_5013 || y_5016;\n                cond_5018 = x_5006 == 0;\n                res_5019 = x_5011 == 0;\n                x_5020 = cond_5018 && res_5019;\n                res_5021 = sle32(x_5006, x_5011);\n                res_5022 = x_5006 == x_5011;\n                x_5023 = cond_4763 && res_5022;\n                res_5024 = x_4764 || x_5023;\n                x_5025 = cond_4762 && res_5021;\n                y_5026 = x_4765 && res_5024;\n                res_5027 = x_5025 || y_5026;\n                x_5028 = cond_4759 && x_5020;\n                y_5029 = x_4760 && res_5027;\n                res_5030 = x_5028 || y_5029;\n                x_5031 = !cond_5017;\n                y_5032 = res_5030 && x_5031;\n                res_5033 = cond_5017 || y_5032;\n                if (res_5033) {\n                    int32_t arg_5035;\n                    int32_t res_5036;\n                    int32_t res_5037;\n                    \n                    arg_5035 = x_5003 + x_5008;\n                    res_5036 = smax32(x_5001, arg_5035);\n                    res_5037 = smax32(x_5007, res_5036);\n                    res_5034 = res_5037;\n                } else {\n                    int32_t res_5038 = smax32(x_5001, x_5007);\n                    \n                    res_5034 = res_5038;\n                }\n                if (cond_5013) {\n                    res_5039 = x_5008;\n                } else {\n                    bool cond_5040;\n                    bool x_5041;\n     ",
            "               int32_t res_5042;\n                    \n                    cond_5040 = x_5004 == x_5002;\n                    x_5041 = res_5033 && cond_5040;\n                    if (x_5041) {\n                        int32_t res_5043 = x_5002 + x_5008;\n                        \n                        res_5042 = res_5043;\n                    } else {\n                        res_5042 = x_5002;\n                    }\n                    res_5039 = res_5042;\n                }\n                if (res_5014) {\n                    res_5044 = x_5003;\n                } else {\n                    bool cond_5045;\n                    bool x_5046;\n                    int32_t res_5047;\n                    \n                    cond_5045 = x_5010 == x_5009;\n                    x_5046 = res_5033 && cond_5045;\n                    if (x_5046) {\n                        int32_t res_5048 = x_5003 + x_5009;\n                        \n                        res_5047 = res_5048;\n                    } else {\n                        res_5047 = x_5009;\n                    }\n                    res_5044 = res_5047;\n                }\n                res_5049 = x_5004 + x_5010;\n                if (cond_5013) {\n                    res_5050 = x_5011;\n                } else {\n                    res_5050 = x_5005;\n                }\n                if (res_5014) {\n                    res_5051 = x_5006;\n                } else {\n                    res_5051 = x_5012;\n                }\n            }\n            x_5001 = res_5034;\n            x_5002 = res_5039;\n            x_5003 = res_5044;\n            x_5004 = res_5049;\n            x_5005 = res_5050;\n            x_5006 = res_5051;\n            *(volatile __local int32_t *) &mem_5140[local_tid_4871 * 4] =\n                x_5001;\n            *(volatile __local int32_t *) &mem_5143[local_tid_4871 * 4] =\n                x_5002;\n            *(volatile __local int32_t *) &mem_5146[local_tid_4871 * 4] =\n                x_5003;\n            *(volatile __local int32_t ",
            "*) &mem_5149[local_tid_4871 * 4] =\n                x_5004;\n            *(volatile __local int32_t *) &mem_5152[local_tid_4871 * 4] =\n                x_5005;\n            *(volatile __local int32_t *) &mem_5155[local_tid_4871 * 4] =\n                x_5006;\n        }\n        offset_5238 *= 2;\n        other_index_5000 = local_tid_4871 + offset_5238;\n    }\n    skip_waves_5237 = 1;\n    while (slt32(skip_waves_5237, squot32(group_sizze_4849 + wave_sizze_5214 -\n                                          1, wave_sizze_5214))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        offset_5238 = skip_waves_5237 * wave_sizze_5214;\n        other_index_5000 = local_tid_4871 + offset_5238;\n        if (slt32(other_index_5000, group_sizze_4849) && ((local_tid_4871 -\n                                                           squot32(local_tid_4871,\n                                                                   wave_sizze_5214) *\n                                                           wave_sizze_5214) ==\n                                                          0 &&\n                                                          (squot32(local_tid_4871,\n                                                                   wave_sizze_5214) &\n                                                           (2 *\n                                                            skip_waves_5237 -\n                                                            1)) == 0)) {\n            // read array element\n            {\n                x_5007 = *(__local int32_t *) &mem_5140[(local_tid_4871 +\n                                                         offset_5238) * 4];\n                x_5008 = *(__local int32_t *) &mem_5143[(local_tid_4871 +\n                                                         offset_5238) * 4];\n                x_5009 = *(__local int32_t *) &mem_5146[(local_tid_4871 +\n                                                         offset_5238) * 4];\n                x_5010 = *(__local int32_t *) &mem_5149",
            "[(local_tid_4871 +\n                                                         offset_5238) * 4];\n                x_5011 = *(__local int32_t *) &mem_5152[(local_tid_4871 +\n                                                         offset_5238) * 4];\n                x_5012 = *(__local int32_t *) &mem_5155[(local_tid_4871 +\n                                                         offset_5238) * 4];\n            }\n            \n            bool cond_5013;\n            bool res_5014;\n            bool x_5015;\n            bool y_5016;\n            bool cond_5017;\n            bool cond_5018;\n            bool res_5019;\n            bool x_5020;\n            bool res_5021;\n            bool res_5022;\n            bool x_5023;\n            bool res_5024;\n            bool x_5025;\n            bool y_5026;\n            bool res_5027;\n            bool x_5028;\n            bool y_5029;\n            bool res_5030;\n            bool x_5031;\n            bool y_5032;\n            bool res_5033;\n            int32_t res_5034;\n            int32_t res_5039;\n            int32_t res_5044;\n            int32_t res_5049;\n            int32_t res_5050;\n            int32_t res_5051;\n            \n            if (thread_active_5216) {\n                cond_5013 = x_5004 == 0;\n                res_5014 = x_5010 == 0;\n                x_5015 = !cond_5013;\n                y_5016 = res_5014 && x_5015;\n                cond_5017 = cond_5013 || y_5016;\n                cond_5018 = x_5006 == 0;\n                res_5019 = x_5011 == 0;\n                x_5020 = cond_5018 && res_5019;\n                res_5021 = sle32(x_5006, x_5011);\n                res_5022 = x_5006 == x_5011;\n                x_5023 = cond_4763 && res_5022;\n                res_5024 = x_4764 || x_5023;\n                x_5025 = cond_4762 && res_5021;\n                y_5026 = x_4765 && res_5024;\n                res_5027 = x_5025 || y_5026;\n                x_5028 = cond_4759 && x_5020;\n                y_5029 = x_4760 && res_5027;\n                res_5030 = x_5028 || y_",
            "5029;\n                x_5031 = !cond_5017;\n                y_5032 = res_5030 && x_5031;\n                res_5033 = cond_5017 || y_5032;\n                if (res_5033) {\n                    int32_t arg_5035;\n                    int32_t res_5036;\n                    int32_t res_5037;\n                    \n                    arg_5035 = x_5003 + x_5008;\n                    res_5036 = smax32(x_5001, arg_5035);\n                    res_5037 = smax32(x_5007, res_5036);\n                    res_5034 = res_5037;\n                } else {\n                    int32_t res_5038 = smax32(x_5001, x_5007);\n                    \n                    res_5034 = res_5038;\n                }\n                if (cond_5013) {\n                    res_5039 = x_5008;\n                } else {\n                    bool cond_5040;\n                    bool x_5041;\n                    int32_t res_5042;\n                    \n                    cond_5040 = x_5004 == x_5002;\n                    x_5041 = res_5033 && cond_5040;\n                    if (x_5041) {\n                        int32_t res_5043 = x_5002 + x_5008;\n                        \n                        res_5042 = res_5043;\n                    } else {\n                        res_5042 = x_5002;\n                    }\n                    res_5039 = res_5042;\n                }\n                if (res_5014) {\n                    res_5044 = x_5003;\n                } else {\n                    bool cond_5045;\n                    bool x_5046;\n                    int32_t res_5047;\n                    \n                    cond_5045 = x_5010 == x_5009;\n                    x_5046 = res_5033 && cond_5045;\n                    if (x_5046) {\n                        int32_t res_5048 = x_5003 + x_5009;\n                        \n                        res_5047 = res_5048;\n                    } else {\n                        res_5047 = x_5009;\n                    }\n                    res_5044 = res_5047;\n                }\n                res_5049 = x_5004 + x_5",
            "010;\n                if (cond_5013) {\n                    res_5050 = x_5011;\n                } else {\n                    res_5050 = x_5005;\n                }\n                if (res_5014) {\n                    res_5051 = x_5006;\n                } else {\n                    res_5051 = x_5012;\n                }\n            }\n            x_5001 = res_5034;\n            x_5002 = res_5039;\n            x_5003 = res_5044;\n            x_5004 = res_5049;\n            x_5005 = res_5050;\n            x_5006 = res_5051;\n            *(__local int32_t *) &mem_5140[local_tid_4871 * 4] = x_5001;\n            *(__local int32_t *) &mem_5143[local_tid_4871 * 4] = x_5002;\n            *(__local int32_t *) &mem_5146[local_tid_4871 * 4] = x_5003;\n            *(__local int32_t *) &mem_5149[local_tid_4871 * 4] = x_5004;\n            *(__local int32_t *) &mem_5152[local_tid_4871 * 4] = x_5005;\n            *(__local int32_t *) &mem_5155[local_tid_4871 * 4] = x_5006;\n        }\n        skip_waves_5237 *= 2;\n    }\n    final_result_4993 = x_5001;\n    final_result_4994 = x_5002;\n    final_result_4995 = x_5003;\n    final_result_4996 = x_5004;\n    final_result_4997 = x_5005;\n    final_result_4998 = x_5006;\n    if (local_tid_4871 == 0) {\n        *(__global int32_t *) &mem_5158[group_id_4872 * 4] = final_result_4993;\n    }\n    if (local_tid_4871 == 0) {\n        *(__global int32_t *) &mem_5161[group_id_4872 * 4] = final_result_4994;\n    }\n    if (local_tid_4871 == 0) {\n        *(__global int32_t *) &mem_5164[group_id_4872 * 4] = final_result_4995;\n    }\n    if (local_tid_4871 == 0) {\n        *(__global int32_t *) &mem_5167[group_id_4872 * 4] = final_result_4996;\n    }\n    if (local_tid_4871 == 0) {\n        *(__global int32_t *) &mem_5170[group_id_4872 * 4] = final_result_4997;\n    }\n    if (local_tid_4871 == 0) {\n        *(__global int32_t *) &mem_5173[group_id_4872 * 4] = final_result_4998;\n    }\n}\n__kernel void fut_kernel_map_transpose_i32(__global int32_t *odata,\n                                        ",
            "   uint odata_offset, __global\n                                           int32_t *idata, uint idata_offset,\n                                           uint width, uint height,\n                                           uint input_size, uint output_size,\n                                           __local int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_global_id(0);\n    y_index = get_global_id(1);\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowheight_i32(__global int32_t *odata,\n                                                     uint odata_offset, __global\n                                                     int32_t *idata,\n                                                     uint idata_offset,\n                                                     uint width, uint height,\n                                                     uint input",
            "_size,\n                                                     uint output_size,\n                                                     uint mulx, __local\n                                                     int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +\n        get_local_id(1) % mulx * FUT_BLOCK_DIM;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +\n        get_local_id(0) % mulx * FUT_BLOCK_DIM;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowwidth_i32(__global int32_t *odata,\n                                                    uint odata_offset, __global\n                                                    int32_t *idata,\n                                                    uint idata_offset,\n                                                    uint",
            " width, uint height,\n                                                    uint input_size,\n                                                    uint output_size, uint muly,\n                                                    __local int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +\n        get_local_id(0) % muly * FUT_BLOCK_DIM;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +\n        get_local_id(1) % muly * FUT_BLOCK_DIM;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_small_i32(__global int32_t *odata,\n                                                 uint odata_offset, __global\n                                                 int32_t *idata,\n                                                 uint idata_offset,\n                                        ",
            "         uint num_arrays, uint width,\n                                                 uint height, uint input_size,\n                                                 uint output_size)\n{\n    uint our_array_offset = get_global_id(0) / (height * width) * (height *\n                                                                   width);\n    uint x_index = get_global_id(0) % (height * width) / height;\n    uint y_index = get_global_id(0) % height;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays.\n    odata += our_array_offset;\n    idata += our_array_offset;\n    \n    uint index_in = y_index * width + x_index;\n    uint index_out = x_index * height + y_index;\n    \n    if (get_global_id(0) < input_size)\n        odata[index_out] = idata[index_in];\n}\n__kernel void reduce_kernel_5058(__local volatile int64_t *mem_aligned_0,\n                                 __local volatile int64_t *mem_aligned_1,\n                                 __local volatile int64_t *mem_aligned_2,\n                                 __local volatile int64_t *mem_aligned_3,\n                                 __local volatile int64_t *mem_aligned_4,\n                                 __local volatile int64_t *mem_aligned_5,\n                                 unsigned char cond_4759, unsigned char x_4760,\n                                 unsigned char cond_4762,\n                                 unsigned char cond_4763, unsigned char x_4764,\n                                 unsigned char x_4765, int32_t num_groups_4856,\n                                 __global unsigned char *mem_5158, __global\n                                 unsigned char *mem_5161, __global\n                                 unsigned char *mem_5164, __global\n                                 unsigned char *mem_5167, __global\n                                 unsigned char *mem_5170, __global\n          ",
            "                       unsigned char *mem_5173, __global\n                                 unsigned char *mem_5194, __global\n                                 unsigned char *mem_5197, __global\n                                 unsigned char *mem_5200, __global\n                                 unsigned char *mem_5203, __global\n                                 unsigned char *mem_5206, __global\n                                 unsigned char *mem_5209)\n{\n    __local volatile char *restrict mem_5176 = mem_aligned_0;\n    __local volatile char *restrict mem_5179 = mem_aligned_1;\n    __local volatile char *restrict mem_5182 = mem_aligned_2;\n    __local volatile char *restrict mem_5185 = mem_aligned_3;\n    __local volatile char *restrict mem_5188 = mem_aligned_4;\n    __local volatile char *restrict mem_5191 = mem_aligned_5;\n    int32_t wave_sizze_5245;\n    int32_t group_sizze_5246;\n    bool thread_active_5247;\n    int32_t global_tid_5058;\n    int32_t local_tid_5059;\n    int32_t group_id_5060;\n    \n    global_tid_5058 = get_global_id(0);\n    local_tid_5059 = get_local_id(0);\n    group_sizze_5246 = get_local_size(0);\n    wave_sizze_5245 = LOCKSTEP_WIDTH;\n    group_id_5060 = get_group_id(0);\n    thread_active_5247 = 1;\n    \n    bool in_bounds_5061;\n    int32_t x_5092;\n    int32_t x_5094;\n    int32_t x_5096;\n    int32_t x_5098;\n    int32_t x_5100;\n    int32_t x_5102;\n    \n    if (thread_active_5247) {\n        in_bounds_5061 = slt32(local_tid_5059, num_groups_4856);\n        if (in_bounds_5061) {\n            int32_t x_5062 = *(__global int32_t *) &mem_5158[global_tid_5058 *\n                                                             4];\n            \n            x_5092 = x_5062;\n        } else {\n            x_5092 = 0;\n        }\n        if (in_bounds_5061) {\n            int32_t x_5064 = *(__global int32_t *) &mem_5161[global_tid_5058 *\n                                                             4];\n            \n            x_5094 = x_5064;\n        } else {\n            x_5094 = 0;\n  ",
            "      }\n        if (in_bounds_5061) {\n            int32_t x_5066 = *(__global int32_t *) &mem_5164[global_tid_5058 *\n                                                             4];\n            \n            x_5096 = x_5066;\n        } else {\n            x_5096 = 0;\n        }\n        if (in_bounds_5061) {\n            int32_t x_5068 = *(__global int32_t *) &mem_5167[global_tid_5058 *\n                                                             4];\n            \n            x_5098 = x_5068;\n        } else {\n            x_5098 = 0;\n        }\n        if (in_bounds_5061) {\n            int32_t x_5070 = *(__global int32_t *) &mem_5170[global_tid_5058 *\n                                                             4];\n            \n            x_5100 = x_5070;\n        } else {\n            x_5100 = 0;\n        }\n        if (in_bounds_5061) {\n            int32_t x_5072 = *(__global int32_t *) &mem_5173[global_tid_5058 *\n                                                             4];\n            \n            x_5102 = x_5072;\n        } else {\n            x_5102 = 0;\n        }\n    }\n    \n    int32_t final_result_5081;\n    int32_t final_result_5082;\n    int32_t final_result_5083;\n    int32_t final_result_5084;\n    int32_t final_result_5085;\n    int32_t final_result_5086;\n    \n    for (int32_t comb_iter_5248 = 0; comb_iter_5248 <\n         squot32(max_num_groups_4851 + max_num_groups_4851 - 1,\n                 max_num_groups_4851); comb_iter_5248++) {\n        int32_t combine_id_5080;\n        int32_t flat_comb_id_5249 = comb_iter_5248 * max_num_groups_4851 +\n                local_tid_5059;\n        \n        combine_id_5080 = flat_comb_id_5249;\n        if (slt32(combine_id_5080, max_num_groups_4851) && 1) {\n            *(__local int32_t *) &mem_5176[combine_id_5080 * 4] = x_5092;\n            *(__local int32_t *) &mem_5179[combine_id_5080 * 4] = x_5094;\n            *(__local int32_t *) &mem_5182[combine_id_5080 * 4] = x_5096;\n            *(__local int32_t *) &mem_5185[combine_id_5080 * 4] =",
            " x_5098;\n            *(__local int32_t *) &mem_5188[combine_id_5080 * 4] = x_5100;\n            *(__local int32_t *) &mem_5191[combine_id_5080 * 4] = x_5102;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t offset_5251;\n    int32_t skip_waves_5250;\n    int32_t x_4772;\n    int32_t x_4773;\n    int32_t x_4774;\n    int32_t x_4775;\n    int32_t x_4776;\n    int32_t x_4777;\n    int32_t x_4778;\n    int32_t x_4779;\n    int32_t x_4780;\n    int32_t x_4781;\n    int32_t x_4782;\n    int32_t x_4783;\n    int32_t my_index_4868;\n    int32_t other_index_4869;\n    \n    my_index_4868 = local_tid_5059;\n    offset_5251 = 0;\n    other_index_4869 = local_tid_5059 + offset_5251;\n    if (slt32(local_tid_5059, max_num_groups_4851)) {\n        x_4772 = *(__local int32_t *) &mem_5176[(local_tid_5059 + offset_5251) *\n                                                4];\n        x_4773 = *(__local int32_t *) &mem_5179[(local_tid_5059 + offset_5251) *\n                                                4];\n        x_4774 = *(__local int32_t *) &mem_5182[(local_tid_5059 + offset_5251) *\n                                                4];\n        x_4775 = *(__local int32_t *) &mem_5185[(local_tid_5059 + offset_5251) *\n                                                4];\n        x_4776 = *(__local int32_t *) &mem_5188[(local_tid_5059 + offset_5251) *\n                                                4];\n        x_4777 = *(__local int32_t *) &mem_5191[(local_tid_5059 + offset_5251) *\n                                                4];\n    }\n    offset_5251 = 1;\n    other_index_4869 = local_tid_5059 + offset_5251;\n    while (slt32(offset_5251, wave_sizze_5245)) {\n        if (slt32(other_index_4869, max_num_groups_4851) && ((local_tid_5059 -\n                                                              squot32(local_tid_5059,\n                                                                      wave_sizze_5245) *\n                                                              wave_sizze_5245) &\n     ",
            "                                                        (2 * offset_5251 -\n                                                              1)) == 0) {\n            // read array element\n            {\n                x_4778 = *(volatile __local\n                           int32_t *) &mem_5176[(local_tid_5059 + offset_5251) *\n                                                4];\n                x_4779 = *(volatile __local\n                           int32_t *) &mem_5179[(local_tid_5059 + offset_5251) *\n                                                4];\n                x_4780 = *(volatile __local\n                           int32_t *) &mem_5182[(local_tid_5059 + offset_5251) *\n                                                4];\n                x_4781 = *(volatile __local\n                           int32_t *) &mem_5185[(local_tid_5059 + offset_5251) *\n                                                4];\n                x_4782 = *(volatile __local\n                           int32_t *) &mem_5188[(local_tid_5059 + offset_5251) *\n                                                4];\n                x_4783 = *(volatile __local\n                           int32_t *) &mem_5191[(local_tid_5059 + offset_5251) *\n                                                4];\n            }\n            \n            bool cond_4784;\n            bool res_4785;\n            bool x_4786;\n            bool y_4787;\n            bool cond_4788;\n            bool cond_4789;\n            bool res_4790;\n            bool x_4791;\n            bool res_4792;\n            bool res_4793;\n            bool x_4794;\n            bool res_4795;\n            bool x_4796;\n            bool y_4797;\n            bool res_4798;\n            bool x_4799;\n            bool y_4800;\n            bool res_4801;\n            bool x_4802;\n            bool y_4803;\n            bool res_4804;\n            int32_t res_4805;\n            int32_t res_4810;\n            int32_t res_4815;\n            int32_t res_4820;\n            int32_t res_4821;\n            in",
            "t32_t res_4822;\n            \n            if (thread_active_5247) {\n                cond_4784 = x_4775 == 0;\n                res_4785 = x_4781 == 0;\n                x_4786 = !cond_4784;\n                y_4787 = res_4785 && x_4786;\n                cond_4788 = cond_4784 || y_4787;\n                cond_4789 = x_4777 == 0;\n                res_4790 = x_4782 == 0;\n                x_4791 = cond_4789 && res_4790;\n                res_4792 = sle32(x_4777, x_4782);\n                res_4793 = x_4777 == x_4782;\n                x_4794 = cond_4763 && res_4793;\n                res_4795 = x_4764 || x_4794;\n                x_4796 = cond_4762 && res_4792;\n                y_4797 = x_4765 && res_4795;\n                res_4798 = x_4796 || y_4797;\n                x_4799 = cond_4759 && x_4791;\n                y_4800 = x_4760 && res_4798;\n                res_4801 = x_4799 || y_4800;\n                x_4802 = !cond_4788;\n                y_4803 = res_4801 && x_4802;\n                res_4804 = cond_4788 || y_4803;\n                if (res_4804) {\n                    int32_t arg_4806;\n                    int32_t res_4807;\n                    int32_t res_4808;\n                    \n                    arg_4806 = x_4774 + x_4779;\n                    res_4807 = smax32(x_4772, arg_4806);\n                    res_4808 = smax32(x_4778, res_4807);\n                    res_4805 = res_4808;\n                } else {\n                    int32_t res_4809 = smax32(x_4772, x_4778);\n                    \n                    res_4805 = res_4809;\n                }\n                if (cond_4784) {\n                    res_4810 = x_4779;\n                } else {\n                    bool cond_4811;\n                    bool x_4812;\n                    int32_t res_4813;\n                    \n                    cond_4811 = x_4775 == x_4773;\n                    x_4812 = res_4804 && cond_4811;\n                    if (x_4812) {\n                        int32_t res_4814 = x_4773 + x_4779;\n                        \n                ",
            "        res_4813 = res_4814;\n                    } else {\n                        res_4813 = x_4773;\n                    }\n                    res_4810 = res_4813;\n                }\n                if (res_4785) {\n                    res_4815 = x_4774;\n                } else {\n                    bool cond_4816;\n                    bool x_4817;\n                    int32_t res_4818;\n                    \n                    cond_4816 = x_4781 == x_4780;\n                    x_4817 = res_4804 && cond_4816;\n                    if (x_4817) {\n                        int32_t res_4819 = x_4774 + x_4780;\n                        \n                        res_4818 = res_4819;\n                    } else {\n                        res_4818 = x_4780;\n                    }\n                    res_4815 = res_4818;\n                }\n                res_4820 = x_4775 + x_4781;\n                if (cond_4784) {\n                    res_4821 = x_4782;\n                } else {\n                    res_4821 = x_4776;\n                }\n                if (res_4785) {\n                    res_4822 = x_4777;\n                } else {\n                    res_4822 = x_4783;\n                }\n            }\n            x_4772 = res_4805;\n            x_4773 = res_4810;\n            x_4774 = res_4815;\n            x_4775 = res_4820;\n            x_4776 = res_4821;\n            x_4777 = res_4822;\n            *(volatile __local int32_t *) &mem_5176[local_tid_5059 * 4] =\n                x_4772;\n            *(volatile __local int32_t *) &mem_5179[local_tid_5059 * 4] =\n                x_4773;\n            *(volatile __local int32_t *) &mem_5182[local_tid_5059 * 4] =\n                x_4774;\n            *(volatile __local int32_t *) &mem_5185[local_tid_5059 * 4] =\n                x_4775;\n            *(volatile __local int32_t *) &mem_5188[local_tid_5059 * 4] =\n                x_4776;\n            *(volatile __local int32_t *) &mem_5191[local_tid_5059 * 4] =\n                x_4777;\n        }\n        offset_5251 *= 2;\n",
            "        other_index_4869 = local_tid_5059 + offset_5251;\n    }\n    skip_waves_5250 = 1;\n    while (slt32(skip_waves_5250, squot32(max_num_groups_4851 +\n                                          wave_sizze_5245 - 1,\n                                          wave_sizze_5245))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        offset_5251 = skip_waves_5250 * wave_sizze_5245;\n        other_index_4869 = local_tid_5059 + offset_5251;\n        if (slt32(other_index_4869, max_num_groups_4851) && ((local_tid_5059 -\n                                                              squot32(local_tid_5059,\n                                                                      wave_sizze_5245) *\n                                                              wave_sizze_5245) ==\n                                                             0 &&\n                                                             (squot32(local_tid_5059,\n                                                                      wave_sizze_5245) &\n                                                              (2 *\n                                                               skip_waves_5250 -\n                                                               1)) == 0)) {\n            // read array element\n            {\n                x_4778 = *(__local int32_t *) &mem_5176[(local_tid_5059 +\n                                                         offset_5251) * 4];\n                x_4779 = *(__local int32_t *) &mem_5179[(local_tid_5059 +\n                                                         offset_5251) * 4];\n                x_4780 = *(__local int32_t *) &mem_5182[(local_tid_5059 +\n                                                         offset_5251) * 4];\n                x_4781 = *(__local int32_t *) &mem_5185[(local_tid_5059 +\n                                                         offset_5251) * 4];\n                x_4782 = *(__local int32_t *) &mem_5188[(local_tid_5059 +\n                                               ",
            "          offset_5251) * 4];\n                x_4783 = *(__local int32_t *) &mem_5191[(local_tid_5059 +\n                                                         offset_5251) * 4];\n            }\n            \n            bool cond_4784;\n            bool res_4785;\n            bool x_4786;\n            bool y_4787;\n            bool cond_4788;\n            bool cond_4789;\n            bool res_4790;\n            bool x_4791;\n            bool res_4792;\n            bool res_4793;\n            bool x_4794;\n            bool res_4795;\n            bool x_4796;\n            bool y_4797;\n            bool res_4798;\n            bool x_4799;\n            bool y_4800;\n            bool res_4801;\n            bool x_4802;\n            bool y_4803;\n            bool res_4804;\n            int32_t res_4805;\n            int32_t res_4810;\n            int32_t res_4815;\n            int32_t res_4820;\n            int32_t res_4821;\n            int32_t res_4822;\n            \n            if (thread_active_5247) {\n                cond_4784 = x_4775 == 0;\n                res_4785 = x_4781 == 0;\n                x_4786 = !cond_4784;\n                y_4787 = res_4785 && x_4786;\n                cond_4788 = cond_4784 || y_4787;\n                cond_4789 = x_4777 == 0;\n                res_4790 = x_4782 == 0;\n                x_4791 = cond_4789 && res_4790;\n                res_4792 = sle32(x_4777, x_4782);\n                res_4793 = x_4777 == x_4782;\n                x_4794 = cond_4763 && res_4793;\n                res_4795 = x_4764 || x_4794;\n                x_4796 = cond_4762 && res_4792;\n                y_4797 = x_4765 && res_4795;\n                res_4798 = x_4796 || y_4797;\n                x_4799 = cond_4759 && x_4791;\n                y_4800 = x_4760 && res_4798;\n                res_4801 = x_4799 || y_4800;\n                x_4802 = !cond_4788;\n                y_4803 = res_4801 && x_4802;\n                res_4804 = cond_4788 || y_4803;\n                if (res_4804) {\n                    int32_t arg_4806;\n          ",
            "          int32_t res_4807;\n                    int32_t res_4808;\n                    \n                    arg_4806 = x_4774 + x_4779;\n                    res_4807 = smax32(x_4772, arg_4806);\n                    res_4808 = smax32(x_4778, res_4807);\n                    res_4805 = res_4808;\n                } else {\n                    int32_t res_4809 = smax32(x_4772, x_4778);\n                    \n                    res_4805 = res_4809;\n                }\n                if (cond_4784) {\n                    res_4810 = x_4779;\n                } else {\n                    bool cond_4811;\n                    bool x_4812;\n                    int32_t res_4813;\n                    \n                    cond_4811 = x_4775 == x_4773;\n                    x_4812 = res_4804 && cond_4811;\n                    if (x_4812) {\n                        int32_t res_4814 = x_4773 + x_4779;\n                        \n                        res_4813 = res_4814;\n                    } else {\n                        res_4813 = x_4773;\n                    }\n                    res_4810 = res_4813;\n                }\n                if (res_4785) {\n                    res_4815 = x_4774;\n                } else {\n                    bool cond_4816;\n                    bool x_4817;\n                    int32_t res_4818;\n                    \n                    cond_4816 = x_4781 == x_4780;\n                    x_4817 = res_4804 && cond_4816;\n                    if (x_4817) {\n                        int32_t res_4819 = x_4774 + x_4780;\n                        \n                        res_4818 = res_4819;\n                    } else {\n                        res_4818 = x_4780;\n                    }\n                    res_4815 = res_4818;\n                }\n                res_4820 = x_4775 + x_4781;\n                if (cond_4784) {\n                    res_4821 = x_4782;\n                } else {\n                    res_4821 = x_4776;\n                }\n                if (res_4785) {\n                    res_4",
            "822 = x_4777;\n                } else {\n                    res_4822 = x_4783;\n                }\n            }\n            x_4772 = res_4805;\n            x_4773 = res_4810;\n            x_4774 = res_4815;\n            x_4775 = res_4820;\n            x_4776 = res_4821;\n            x_4777 = res_4822;\n            *(__local int32_t *) &mem_5176[local_tid_5059 * 4] = x_4772;\n            *(__local int32_t *) &mem_5179[local_tid_5059 * 4] = x_4773;\n            *(__local int32_t *) &mem_5182[local_tid_5059 * 4] = x_4774;\n            *(__local int32_t *) &mem_5185[local_tid_5059 * 4] = x_4775;\n            *(__local int32_t *) &mem_5188[local_tid_5059 * 4] = x_4776;\n            *(__local int32_t *) &mem_5191[local_tid_5059 * 4] = x_4777;\n        }\n        skip_waves_5250 *= 2;\n    }\n    final_result_5081 = x_4772;\n    final_result_5082 = x_4773;\n    final_result_5083 = x_4774;\n    final_result_5084 = x_4775;\n    final_result_5085 = x_4776;\n    final_result_5086 = x_4777;\n    if (local_tid_5059 == 0) {\n        *(__global int32_t *) &mem_5194[group_id_5060 * 4] = final_result_5081;\n    }\n    if (local_tid_5059 == 0) {\n        *(__global int32_t *) &mem_5197[group_id_5060 * 4] = final_result_5082;\n    }\n    if (local_tid_5059 == 0) {\n        *(__global int32_t *) &mem_5200[group_id_5060 * 4] = final_result_5083;\n    }\n    if (local_tid_5059 == 0) {\n        *(__global int32_t *) &mem_5203[group_id_5060 * 4] = final_result_5084;\n    }\n    if (local_tid_5059 == 0) {\n        *(__global int32_t *) &mem_5206[group_id_5060 * 4] = final_result_5085;\n    }\n    if (local_tid_5059 == 0) {\n        *(__global int32_t *) &mem_5209[group_id_5060 * 4] = final_result_5086;\n    }\n}\n",
            NULL};
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
    const char *desc;
} ;
struct memblock_local {
    int *references;
    unsigned char mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
static const char *size_names[] = {"group_size_4848", "max_num_groups_4850"};
static const char *size_classes[] = {"group_size", "num_groups"};
static const char *size_entry_points[] = {"main", "main"};
int futhark_get_num_sizes(void)
{
    return 2;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
const char *futhark_get_size_entry(int i)
{
    return size_entry_points[i];
}
struct sizes {
    size_t group_sizze_4848;
    size_t max_num_groups_4850;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[2];
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    opencl_config_init(&cfg->opencl, 2, size_names, cfg->sizes, size_classes,
                       size_entry_points);
    cfg->opencl.transpose_block_dim = 16;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.logging = cfg->opencl.debugging = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->opencl.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->opencl, s);
}
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s)
{
    set_preferred_platform(&cfg->opencl, s);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->opencl.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->opencl.load_program_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->opencl.default_group_size = size;
    cfg->opencl.default_group_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->opencl.default_num_groups = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_tile_size = size;
    cfg->opencl.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 2; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int logging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_local;
    int64_t cur_mem_usage_local;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    int total_runs;
    long total_runtime;
    cl_kernel chunked_reduce_kernel_4870;
    int chunked_reduce_kernel_4870_total_runtime;
    int chunked_reduce_kernel_4870_runs;
    cl_kernel fut_kernel_map_transpose_i32;
    int fut_kernel_map_transpose_i32_total_runtime;
    int fut_kernel_map_transpose_i32_runs;
    cl_kernel fut_kernel_map_transpose_lowheight_i32;
    int fut_kernel_map_transpose_lowheight_i32_total_runtime;
    int fut_kernel_map_transpose_lowheight_i32_runs;
    cl_kernel fut_kernel_map_transpose_lowwidth_i32;
    int fut_kernel_map_transpose_lowwidth_i32_total_runtime;
    int fut_kernel_map_transpose_lowwidth_i32_runs;
    cl_kernel fut_kernel_map_transpose_small_i32;
    int fut_kernel_map_transpose_small_i32_total_runtime;
    int fut_kernel_map_transpose_small_i32_runs;
    cl_kernel reduce_kernel_5058;
    int reduce_kernel_5058_total_runtime;
    int reduce_kernel_5058_runs;
    struct opencl_context opencl;
    struct sizes sizes;
} ;
void post_opencl_setup(struct opencl_context *ctx,
                       struct opencl_device_option *option)
{
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "NVIDIA CUDA") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "AMD Accelerated Parallel Processing") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 64;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_num_groups = 128;
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_group_size = 256;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_tile_size = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_CPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_CPU)
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ctx->cfg.default_num_groups),
                        &ctx->cfg.default_num_groups, NULL);
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_group_size = 32;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_tile_size = 4;
}
static void init_context_early(struct futhark_context_config *cfg,
                               struct futhark_context *ctx)
{
    cl_int error;
    
    ctx->opencl.cfg = cfg->opencl;
    ctx->detail_memory = cfg->opencl.debugging;
    ctx->debugging = cfg->opencl.debugging;
    ctx->logging = cfg->opencl.logging;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_local = 0;
    ctx->cur_mem_usage_local = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->chunked_reduce_kernel_4870_total_runtime = 0;
    ctx->chunked_reduce_kernel_4870_runs = 0;
    ctx->fut_kernel_map_transpose_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_i32_runs = 0;
    ctx->fut_kernel_map_transpose_lowheight_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowheight_i32_runs = 0;
    ctx->fut_kernel_map_transpose_lowwidth_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowwidth_i32_runs = 0;
    ctx->fut_kernel_map_transpose_small_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_small_i32_runs = 0;
    ctx->reduce_kernel_5058_total_runtime = 0;
    ctx->reduce_kernel_5058_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    
    {
        ctx->chunked_reduce_kernel_4870 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_4870",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_4870");
    }
    {
        ctx->fut_kernel_map_transpose_i32 = clCreateKernel(prog,
                                                           "fut_kernel_map_transpose_i32",
                                                           &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_i32");
    }
    {
        ctx->fut_kernel_map_transpose_lowheight_i32 = clCreateKernel(prog,
                                                                     "fut_kernel_map_transpose_lowheight_i32",
                                                                     &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowheight_i32");
    }
    {
        ctx->fut_kernel_map_transpose_lowwidth_i32 = clCreateKernel(prog,
                                                                    "fut_kernel_map_transpose_lowwidth_i32",
                                                                    &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowwidth_i32");
    }
    {
        ctx->fut_kernel_map_transpose_small_i32 = clCreateKernel(prog,
                                                                 "fut_kernel_map_transpose_small_i32",
                                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_small_i32");
    }
    {
        ctx->reduce_kernel_5058 = clCreateKernel(prog, "reduce_kernel_5058",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_5058");
    }
    ctx->sizes.group_sizze_4848 = cfg->sizes[0];
    ctx->sizes.max_num_groups_4850 = cfg->sizes[1];
    return 0;
}
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program,
                                   required_types);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
struct futhark_context *futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                                               cl_command_queue queue)
{
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl_with_command_queue(&ctx->opencl, queue,
                                                      opencl_program,
                                                      required_types);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(clFinish(ctx->opencl.queue));
    return ctx->error != NULL;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(opencl_free_all(&ctx->opencl));
    return ctx->error != NULL;
}
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx)
{
    return ctx->opencl.queue;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED_OR_RETURN(opencl_free(&ctx->opencl, block->mem,
                                                 block->desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'device'",
              ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    OPENCL_SUCCEED_OR_RETURN(opencl_alloc(&ctx->opencl, size, desc,
                                          &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref_local(struct futhark_context *ctx,
                                struct memblock_local *block, const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'local'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_local -= block->size;
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_local);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_local(struct futhark_context *ctx,
                                struct memblock_local *block, int64_t size,
                                const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'local'",
              ctx->cur_mem_usage_local);
    
    int ret = memblock_unref_local(ctx, block, desc);
    
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_local += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'local'",
                (long long) ctx->cur_mem_usage_local);
    if (ctx->cur_mem_usage_local > ctx->peak_mem_usage_local) {
        ctx->peak_mem_usage_local = ctx->cur_mem_usage_local;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set_local(struct futhark_context *ctx,
                              struct memblock_local *lhs,
                              struct memblock_local *rhs, const char *lhs_desc)
{
    int ret = memblock_unref_local(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory) {
        fprintf(stderr, "Peak memory usage for space 'device': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for space 'local': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_local);
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->debugging) {
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_4870             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_4870_runs,
                (long) ctx->chunked_reduce_kernel_4870_total_runtime /
                (ctx->chunked_reduce_kernel_4870_runs !=
                 0 ? ctx->chunked_reduce_kernel_4870_runs : 1),
                (long) ctx->chunked_reduce_kernel_4870_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_4870_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_4870_runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_i32           executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_i32_runs,
                (long) ctx->fut_kernel_map_transpose_i32_total_runtime /
                (ctx->fut_kernel_map_transpose_i32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_i32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_i32_total_runtime);
        ctx->total_runtime += ctx->fut_kernel_map_transpose_i32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_i32_runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_lowheight_i32 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_lowheight_i32_runs,
                (long) ctx->fut_kernel_map_transpose_lowheight_i32_total_runtime /
                (ctx->fut_kernel_map_transpose_lowheight_i32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_lowheight_i32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_lowheight_i32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_lowheight_i32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_lowheight_i32_runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_lowwidth_i32  executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_lowwidth_i32_runs,
                (long) ctx->fut_kernel_map_transpose_lowwidth_i32_total_runtime /
                (ctx->fut_kernel_map_transpose_lowwidth_i32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_lowwidth_i32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_lowwidth_i32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_lowwidth_i32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_lowwidth_i32_runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_small_i32     executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_small_i32_runs,
                (long) ctx->fut_kernel_map_transpose_small_i32_total_runtime /
                (ctx->fut_kernel_map_transpose_small_i32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_small_i32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_small_i32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_small_i32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_small_i32_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_5058                     executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_5058_runs,
                (long) ctx->reduce_kernel_5058_total_runtime /
                (ctx->reduce_kernel_5058_runs !=
                 0 ? ctx->reduce_kernel_5058_runs : 1),
                (long) ctx->reduce_kernel_5058_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_5058_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_5058_runs;
        if (ctx->debugging)
            fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_map_transpose_opencl_i32(struct futhark_context *ctx,
                                           struct memblock_device destmem_0,
                                           int32_t destoffset_1,
                                           struct memblock_device srcmem_2,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8);
static int futrts_main(struct futhark_context *ctx,
                       int32_t *out_scalar_out_5278, int64_t xs_mem_sizze_5126,
                       struct memblock_device xs_mem_5127, int32_t sizze_4756,
                       int32_t pind_4757);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static int futrts_map_transpose_opencl_i32(struct futhark_context *ctx,
                                           struct memblock_device destmem_0,
                                           int32_t destoffset_1,
                                           struct memblock_device srcmem_2,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8)
{
    if (!(num_arrays_4 * x_elems_5 * y_elems_6 == 0)) {
        if (in_elems_7 == out_elems_8 && ((num_arrays_4 == 1 || x_elems_5 *
                                           y_elems_6 == in_elems_7) &&
                                          (x_elems_5 == 1 || y_elems_6 == 1))) {
            if (in_elems_7 * sizeof(int32_t) > 0) {
                OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                             srcmem_2.mem,
                                                             destmem_0.mem,
                                                             srcoffset_3,
                                                             destoffset_1,
                                                             in_elems_7 *
                                                             sizeof(int32_t), 0,
                                                             NULL, NULL));
                if (ctx->debugging)
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            }
        } else {
            if (sle32(x_elems_5, squot32(16, 2)) && slt32(16, y_elems_6)) {
                int32_t muly_9 = squot32(16, x_elems_5);
                int32_t new_height_10;
                
                new_height_10 = squot32(y_elems_6 + muly_9 - 1, muly_9);
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        0,
                                                        sizeof(destmem_0.mem),
                                                        &destmem_0.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        1, sizeof(destoffset_1),
                                                        &destoffset_1));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        2, sizeof(srcmem_2.mem),
                                                        &srcmem_2.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        3, sizeof(srcoffset_3),
                                                        &srcoffset_3));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        4, sizeof(x_elems_5),
                                                        &x_elems_5));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        5, sizeof(y_elems_6),
                                                        &y_elems_6));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        6, sizeof(in_elems_7),
                                                        &in_elems_7));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        7, sizeof(out_elems_8),
                                                        &out_elems_8));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        8, sizeof(muly_9),
                                                        &muly_9));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                        9, 272 *
                                                        sizeof(int32_t), NULL));
                if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16), 16)) *
                    (new_height_10 + srem32(16 - srem32(new_height_10, 16),
                                            16)) * num_arrays_4 != 0) {
                    const size_t global_work_sizze_5258[3] = {x_elems_5 +
                                                              srem32(16 -
                                                                     srem32(x_elems_5,
                                                                            16),
                                                                     16),
                                                              new_height_10 +
                                                              srem32(16 -
                                                                     srem32(new_height_10,
                                                                            16),
                                                                     16),
                                                              num_arrays_4};
                    const size_t local_work_sizze_5262[3] = {16, 16, 1};
                    int64_t time_start_5259 = 0, time_end_5260 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_lowwidth_i32");
                        fprintf(stderr, "%zu", global_work_sizze_5258[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_5258[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_5258[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_5262[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_5262[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_5262[2]);
                        fprintf(stderr, "].\n");
                        time_start_5259 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                                    3, NULL,
                                                                    global_work_sizze_5258,
                                                                    local_work_sizze_5262,
                                                                    0, NULL,
                                                                    NULL));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_5260 = get_wall_time();
                        
                        long time_diff_5261 = time_end_5260 - time_start_5259;
                        
                        ctx->fut_kernel_map_transpose_lowwidth_i32_total_runtime +=
                            time_diff_5261;
                        ctx->fut_kernel_map_transpose_lowwidth_i32_runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "fut_kernel_map_transpose_lowwidth_i32",
                                time_diff_5261);
                    }
                }
            } else {
                if (sle32(y_elems_6, squot32(16, 2)) && slt32(16, x_elems_5)) {
                    int32_t mulx_11 = squot32(16, y_elems_6);
                    int32_t new_width_12;
                    
                    new_width_12 = squot32(x_elems_5 + mulx_11 - 1, mulx_11);
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            0,
                                                            sizeof(destmem_0.mem),
                                                            &destmem_0.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            1,
                                                            sizeof(destoffset_1),
                                                            &destoffset_1));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            2,
                                                            sizeof(srcmem_2.mem),
                                                            &srcmem_2.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            3,
                                                            sizeof(srcoffset_3),
                                                            &srcoffset_3));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            4,
                                                            sizeof(x_elems_5),
                                                            &x_elems_5));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            5,
                                                            sizeof(y_elems_6),
                                                            &y_elems_6));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            6,
                                                            sizeof(in_elems_7),
                                                            &in_elems_7));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            7,
                                                            sizeof(out_elems_8),
                                                            &out_elems_8));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            8, sizeof(mulx_11),
                                                            &mulx_11));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_i32,
                                                            9, 272 *
                                                            sizeof(int32_t),
                                                            NULL));
                    if (1 * (new_width_12 + srem32(16 - srem32(new_width_12,
                                                               16), 16)) *
                        (y_elems_6 + srem32(16 - srem32(y_elems_6, 16), 16)) *
                        num_arrays_4 != 0) {
                        const size_t global_work_sizze_5263[3] = {new_width_12 +
                                                                  srem32(16 -
                                                                         srem32(new_width_12,
                                                                                16),
                                                                         16),
                                                                  y_elems_6 +
                                                                  srem32(16 -
                                                                         srem32(y_elems_6,
                                                                                16),
                                                                         16),
                                                                  num_arrays_4};
                        const size_t local_work_sizze_5267[3] = {16, 16, 1};
                        int64_t time_start_5264 = 0, time_end_5265 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "fut_kernel_map_transpose_lowheight_i32");
                            fprintf(stderr, "%zu", global_work_sizze_5263[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_5263[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_5263[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_5267[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_5267[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_5267[2]);
                            fprintf(stderr, "].\n");
                            time_start_5264 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->fut_kernel_map_transpose_lowheight_i32,
                                                                        3, NULL,
                                                                        global_work_sizze_5263,
                                                                        local_work_sizze_5267,
                                                                        0, NULL,
                                                                        NULL));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_5265 = get_wall_time();
                            
                            long time_diff_5266 = time_end_5265 -
                                 time_start_5264;
                            
                            ctx->fut_kernel_map_transpose_lowheight_i32_total_runtime +=
                                time_diff_5266;
                            ctx->fut_kernel_map_transpose_lowheight_i32_runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_lowheight_i32",
                                    time_diff_5266);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, squot32(16, 2)) && sle32(y_elems_6,
                                                                  squot32(16,
                                                                          2))) {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                0,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                2,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                3,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                4,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                5,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                6,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                7,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_i32,
                                                                8,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        if (1 * (num_arrays_4 * x_elems_5 * y_elems_6 +
                                 srem32(256 - srem32(num_arrays_4 * x_elems_5 *
                                                     y_elems_6, 256), 256)) !=
                            0) {
                            const size_t global_work_sizze_5268[1] =
                                         {num_arrays_4 * x_elems_5 * y_elems_6 +
                                         srem32(256 - srem32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256),
                                                256)};
                            const size_t local_work_sizze_5272[1] = {256};
                            int64_t time_start_5269 = 0, time_end_5270 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_small_i32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_5268[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_5272[0]);
                                fprintf(stderr, "].\n");
                                time_start_5269 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->fut_kernel_map_transpose_small_i32,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_5268,
                                                                            local_work_sizze_5272,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_5270 = get_wall_time();
                                
                                long time_diff_5271 = time_end_5270 -
                                     time_start_5269;
                                
                                ctx->fut_kernel_map_transpose_small_i32_total_runtime +=
                                    time_diff_5271;
                                ctx->fut_kernel_map_transpose_small_i32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_small_i32",
                                        time_diff_5271);
                            }
                        }
                    } else {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                0,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                2,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                3,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                6,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                7,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_i32,
                                                                8, 272 *
                                                                sizeof(int32_t),
                                                                NULL));
                        if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16),
                                                    16)) * (y_elems_6 +
                                                            srem32(16 -
                                                                   srem32(y_elems_6,
                                                                          16),
                                                                   16)) *
                            num_arrays_4 != 0) {
                            const size_t global_work_sizze_5273[3] =
                                         {x_elems_5 + srem32(16 -
                                                             srem32(x_elems_5,
                                                                    16), 16),
                                          y_elems_6 + srem32(16 -
                                                             srem32(y_elems_6,
                                                                    16), 16),
                                          num_arrays_4};
                            const size_t local_work_sizze_5277[3] = {16, 16, 1};
                            int64_t time_start_5274 = 0, time_end_5275 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_i32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_5273[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_5273[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_5273[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_5277[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_5277[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_5277[2]);
                                fprintf(stderr, "].\n");
                                time_start_5274 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->fut_kernel_map_transpose_i32,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_5273,
                                                                            local_work_sizze_5277,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_5275 = get_wall_time();
                                
                                long time_diff_5276 = time_end_5275 -
                                     time_start_5274;
                                
                                ctx->fut_kernel_map_transpose_i32_total_runtime +=
                                    time_diff_5276;
                                ctx->fut_kernel_map_transpose_i32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_i32",
                                        time_diff_5276);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
static int futrts_main(struct futhark_context *ctx,
                       int32_t *out_scalar_out_5278, int64_t xs_mem_sizze_5126,
                       struct memblock_device xs_mem_5127, int32_t sizze_4756,
                       int32_t pind_4757)
{
    int32_t scalar_out_5212;
    bool cond_4759 = pind_4757 == 1;
    bool x_4760 = !cond_4759;
    bool cond_4762 = pind_4757 == 2;
    bool cond_4763 = pind_4757 == 3;
    bool x_4764 = !cond_4763;
    bool x_4765 = !cond_4762;
    int32_t group_sizze_4849;
    
    group_sizze_4849 = ctx->sizes.group_sizze_4848;
    
    int32_t max_num_groups_4851;
    
    max_num_groups_4851 = ctx->sizes.max_num_groups_4850;
    
    int32_t y_4852 = group_sizze_4849 - 1;
    int32_t x_4853 = sizze_4756 + y_4852;
    int32_t w_div_group_sizze_4854 = squot32(x_4853, group_sizze_4849);
    int32_t num_groups_maybe_zzero_4855 = smin32(max_num_groups_4851,
                                                 w_div_group_sizze_4854);
    int32_t num_groups_4856 = smax32(1, num_groups_maybe_zzero_4855);
    int32_t num_threads_4857 = group_sizze_4849 * num_groups_4856;
    int32_t y_4858 = num_threads_4857 - 1;
    int32_t x_4859 = sizze_4756 + y_4858;
    int32_t per_thread_elements_4860 = squot32(x_4859, num_threads_4857);
    int32_t y_5104 = smod32(sizze_4756, num_threads_4857);
    int32_t x_5105 = num_threads_4857 - y_5104;
    int32_t y_5106 = smod32(x_5105, num_threads_4857);
    int32_t padded_sizze_5107 = sizze_4756 + y_5106;
    int32_t per_chunk_5109 = squot32(padded_sizze_5107, num_threads_4857);
    int64_t binop_x_5129 = sext_i32_i64(y_5106);
    int64_t bytes_5128 = 4 * binop_x_5129;
    struct memblock_device mem_5130;
    
    mem_5130.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5130, bytes_5128, "mem_5130"))
        return 1;
    
    int64_t binop_x_5132 = sext_i32_i64(padded_sizze_5107);
    int64_t bytes_5131 = 4 * binop_x_5132;
    struct memblock_device mem_5133;
    
    mem_5133.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5133, bytes_5131, "mem_5133"))
        return 1;
    
    int32_t tmp_offs_5213 = 0;
    
    if (sizze_4756 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     xs_mem_5127.mem,
                                                     mem_5133.mem, 0,
                                                     tmp_offs_5213 * 4,
                                                     sizze_4756 *
                                                     sizeof(int32_t), 0, NULL,
                                                     NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    tmp_offs_5213 += sizze_4756;
    if (y_5106 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     mem_5130.mem, mem_5133.mem,
                                                     0, tmp_offs_5213 * 4,
                                                     y_5106 * sizeof(int32_t),
                                                     0, NULL, NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    tmp_offs_5213 += y_5106;
    if (memblock_unref_device(ctx, &mem_5130, "mem_5130") != 0)
        return 1;
    
    int32_t convop_x_5135 = num_threads_4857 * per_chunk_5109;
    int64_t binop_x_5136 = sext_i32_i64(convop_x_5135);
    int64_t bytes_5134 = 4 * binop_x_5136;
    struct memblock_device mem_5137;
    
    mem_5137.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5137, bytes_5134, "mem_5137"))
        return 1;
    
    int call_ret_5279 = futrts_map_transpose_opencl_i32(ctx, mem_5137, 0,
                                                        mem_5133, 0, 1,
                                                        per_chunk_5109,
                                                        num_threads_4857,
                                                        num_threads_4857 *
                                                        per_chunk_5109,
                                                        num_threads_4857 *
                                                        per_chunk_5109);
    
    assert(call_ret_5279 == 0);
    if (memblock_unref_device(ctx, &mem_5133, "mem_5133") != 0)
        return 1;
    
    int64_t binop_x_5157 = sext_i32_i64(num_groups_4856);
    int64_t bytes_5156 = 4 * binop_x_5157;
    struct memblock_device mem_5158;
    
    mem_5158.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5158, bytes_5156, "mem_5158"))
        return 1;
    
    struct memblock_device mem_5161;
    
    mem_5161.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5161, bytes_5156, "mem_5161"))
        return 1;
    
    struct memblock_device mem_5164;
    
    mem_5164.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5164, bytes_5156, "mem_5164"))
        return 1;
    
    struct memblock_device mem_5167;
    
    mem_5167.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5167, bytes_5156, "mem_5167"))
        return 1;
    
    struct memblock_device mem_5170;
    
    mem_5170.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5170, bytes_5156, "mem_5170"))
        return 1;
    
    struct memblock_device mem_5173;
    
    mem_5173.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5173, bytes_5156, "mem_5173"))
        return 1;
    
    int64_t binop_x_5139 = sext_i32_i64(group_sizze_4849);
    int64_t bytes_5138 = 4 * binop_x_5139;
    struct memblock_local mem_5140;
    
    mem_5140.references = NULL;
    
    struct memblock_local mem_5143;
    
    mem_5143.references = NULL;
    
    struct memblock_local mem_5146;
    
    mem_5146.references = NULL;
    
    struct memblock_local mem_5149;
    
    mem_5149.references = NULL;
    
    struct memblock_local mem_5152;
    
    mem_5152.references = NULL;
    
    struct memblock_local mem_5155;
    
    mem_5155.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_4756);
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 0,
                                            bytes_5138, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 1,
                                            bytes_5138, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 2,
                                            bytes_5138, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 3,
                                            bytes_5138, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 4,
                                            bytes_5138, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 5,
                                            bytes_5138, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 6,
                                            sizeof(sizze_4756), &sizze_4756));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 7,
                                            sizeof(cond_4759), &cond_4759));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 8,
                                            sizeof(x_4760), &x_4760));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 9,
                                            sizeof(cond_4762), &cond_4762));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 10,
                                            sizeof(cond_4763), &cond_4763));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 11,
                                            sizeof(x_4764), &x_4764));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 12,
                                            sizeof(x_4765), &x_4765));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 13,
                                            sizeof(num_threads_4857),
                                            &num_threads_4857));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 14,
                                            sizeof(per_thread_elements_4860),
                                            &per_thread_elements_4860));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 15,
                                            sizeof(per_chunk_5109),
                                            &per_chunk_5109));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 16,
                                            sizeof(mem_5137.mem),
                                            &mem_5137.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 17,
                                            sizeof(mem_5158.mem),
                                            &mem_5158.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 18,
                                            sizeof(mem_5161.mem),
                                            &mem_5161.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 19,
                                            sizeof(mem_5164.mem),
                                            &mem_5164.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 20,
                                            sizeof(mem_5167.mem),
                                            &mem_5167.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 21,
                                            sizeof(mem_5170.mem),
                                            &mem_5170.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->chunked_reduce_kernel_4870, 22,
                                            sizeof(mem_5173.mem),
                                            &mem_5173.mem));
    if (1 * (num_groups_4856 * group_sizze_4849) != 0) {
        const size_t global_work_sizze_5280[1] = {num_groups_4856 *
                     group_sizze_4849};
        const size_t local_work_sizze_5284[1] = {group_sizze_4849};
        int64_t time_start_5281 = 0, time_end_5282 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_4870");
            fprintf(stderr, "%zu", global_work_sizze_5280[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_5284[0]);
            fprintf(stderr, "].\n");
            time_start_5281 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->chunked_reduce_kernel_4870,
                                                        1, NULL,
                                                        global_work_sizze_5280,
                                                        local_work_sizze_5284,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_5282 = get_wall_time();
            
            long time_diff_5283 = time_end_5282 - time_start_5281;
            
            ctx->chunked_reduce_kernel_4870_total_runtime += time_diff_5283;
            ctx->chunked_reduce_kernel_4870_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_4870", time_diff_5283);
        }
    }
    if (memblock_unref_device(ctx, &mem_5137, "mem_5137") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5140, "mem_5140") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5143, "mem_5143") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5146, "mem_5146") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5149, "mem_5149") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5152, "mem_5152") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5155, "mem_5155") != 0)
        return 1;
    
    struct memblock_device mem_5194;
    
    mem_5194.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5194, 4, "mem_5194"))
        return 1;
    
    struct memblock_device mem_5197;
    
    mem_5197.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5197, 4, "mem_5197"))
        return 1;
    
    struct memblock_device mem_5200;
    
    mem_5200.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5200, 4, "mem_5200"))
        return 1;
    
    struct memblock_device mem_5203;
    
    mem_5203.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5203, 4, "mem_5203"))
        return 1;
    
    struct memblock_device mem_5206;
    
    mem_5206.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5206, 4, "mem_5206"))
        return 1;
    
    struct memblock_device mem_5209;
    
    mem_5209.references = NULL;
    if (memblock_alloc_device(ctx, &mem_5209, 4, "mem_5209"))
        return 1;
    
    int64_t binop_x_5175 = sext_i32_i64(max_num_groups_4851);
    int64_t bytes_5174 = 4 * binop_x_5175;
    struct memblock_local mem_5176;
    
    mem_5176.references = NULL;
    
    struct memblock_local mem_5179;
    
    mem_5179.references = NULL;
    
    struct memblock_local mem_5182;
    
    mem_5182.references = NULL;
    
    struct memblock_local mem_5185;
    
    mem_5185.references = NULL;
    
    struct memblock_local mem_5188;
    
    mem_5188.references = NULL;
    
    struct memblock_local mem_5191;
    
    mem_5191.references = NULL;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 0,
                                            bytes_5174, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 1,
                                            bytes_5174, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 2,
                                            bytes_5174, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 3,
                                            bytes_5174, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 4,
                                            bytes_5174, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 5,
                                            bytes_5174, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 6,
                                            sizeof(cond_4759), &cond_4759));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 7,
                                            sizeof(x_4760), &x_4760));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 8,
                                            sizeof(cond_4762), &cond_4762));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 9,
                                            sizeof(cond_4763), &cond_4763));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 10,
                                            sizeof(x_4764), &x_4764));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 11,
                                            sizeof(x_4765), &x_4765));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 12,
                                            sizeof(num_groups_4856),
                                            &num_groups_4856));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 13,
                                            sizeof(mem_5158.mem),
                                            &mem_5158.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 14,
                                            sizeof(mem_5161.mem),
                                            &mem_5161.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 15,
                                            sizeof(mem_5164.mem),
                                            &mem_5164.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 16,
                                            sizeof(mem_5167.mem),
                                            &mem_5167.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 17,
                                            sizeof(mem_5170.mem),
                                            &mem_5170.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 18,
                                            sizeof(mem_5173.mem),
                                            &mem_5173.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 19,
                                            sizeof(mem_5194.mem),
                                            &mem_5194.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 20,
                                            sizeof(mem_5197.mem),
                                            &mem_5197.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 21,
                                            sizeof(mem_5200.mem),
                                            &mem_5200.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 22,
                                            sizeof(mem_5203.mem),
                                            &mem_5203.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 23,
                                            sizeof(mem_5206.mem),
                                            &mem_5206.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->reduce_kernel_5058, 24,
                                            sizeof(mem_5209.mem),
                                            &mem_5209.mem));
    if (1 * max_num_groups_4851 != 0) {
        const size_t global_work_sizze_5285[1] = {max_num_groups_4851};
        const size_t local_work_sizze_5289[1] = {max_num_groups_4851};
        int64_t time_start_5286 = 0, time_end_5287 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_5058");
            fprintf(stderr, "%zu", global_work_sizze_5285[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_5289[0]);
            fprintf(stderr, "].\n");
            time_start_5286 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->reduce_kernel_5058,
                                                        1, NULL,
                                                        global_work_sizze_5285,
                                                        local_work_sizze_5289,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_5287 = get_wall_time();
            
            long time_diff_5288 = time_end_5287 - time_start_5286;
            
            ctx->reduce_kernel_5058_total_runtime += time_diff_5288;
            ctx->reduce_kernel_5058_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_5058",
                    time_diff_5288);
        }
    }
    if (memblock_unref_device(ctx, &mem_5158, "mem_5158") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5161, "mem_5161") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5164, "mem_5164") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5167, "mem_5167") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5170, "mem_5170") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5173, "mem_5173") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5176, "mem_5176") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5179, "mem_5179") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5182, "mem_5182") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5185, "mem_5185") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5188, "mem_5188") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5191, "mem_5191") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5197, "mem_5197") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5200, "mem_5200") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5203, "mem_5203") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5206, "mem_5206") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5209, "mem_5209") != 0)
        return 1;
    
    int32_t read_res_5290;
    
    OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                 mem_5194.mem, CL_TRUE, 0,
                                                 sizeof(int32_t),
                                                 &read_res_5290, 0, NULL,
                                                 NULL));
    
    int32_t res_4766 = read_res_5290;
    
    if (memblock_unref_device(ctx, &mem_5194, "mem_5194") != 0)
        return 1;
    scalar_out_5212 = res_4766;
    *out_scalar_out_5278 = scalar_out_5212;
    if (memblock_unref_local(ctx, &mem_5191, "mem_5191") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5188, "mem_5188") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5185, "mem_5185") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5182, "mem_5182") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5179, "mem_5179") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5176, "mem_5176") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5209, "mem_5209") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5206, "mem_5206") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5203, "mem_5203") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5200, "mem_5200") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5197, "mem_5197") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5194, "mem_5194") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5155, "mem_5155") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5152, "mem_5152") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5149, "mem_5149") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5146, "mem_5146") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5143, "mem_5143") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_5140, "mem_5140") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5173, "mem_5173") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5170, "mem_5170") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5167, "mem_5167") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5164, "mem_5164") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5161, "mem_5161") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5158, "mem_5158") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5137, "mem_5137") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5133, "mem_5133") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_5130, "mem_5130") != 0)
        return 1;
    return 0;
}
struct futhark_i32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int dim0)
{
    struct futhark_i32_1d *arr = malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(int32_t),
                              "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * sizeof(int32_t),
                                                      data + 0, 0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0)
{
    struct futhark_i32_1d *arr = malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(int32_t),
                              "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * sizeof(int32_t), 0,
                                                     NULL, NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_1d(struct futhark_context *ctx, struct futhark_i32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     sizeof(int32_t), data + 0,
                                                     0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                 struct futhark_i32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr)
{
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx, int32_t *out0, const
                       int32_t in0, const struct futhark_i32_1d *in1)
{
    int64_t xs_mem_sizze_5126;
    struct memblock_device xs_mem_5127;
    
    xs_mem_5127.references = NULL;
    
    int32_t sizze_4756;
    int32_t pind_4757;
    int32_t scalar_out_5212;
    
    lock_lock(&ctx->lock);
    pind_4757 = in0;
    xs_mem_5127 = in1->mem;
    xs_mem_sizze_5126 = in1->mem.size;
    sizze_4756 = in1->shape[0];
    
    int ret = futrts_main(ctx, &scalar_out_5212, xs_mem_sizze_5126, xs_mem_5127,
                          sizze_4756, pind_4757);
    
    if (ret == 0) {
        *out0 = scalar_out_5212;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
