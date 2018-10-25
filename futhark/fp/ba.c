/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


/*
 * Initialisation
*/

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
const char *futhark_get_size_entry(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
struct futhark_context
*futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                        cl_command_queue queue);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx);

/*
 * Arrays
*/

struct f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              cl_mem data, int offset,
                                              int dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
cl_mem futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                 struct futhark_f32_1d *arr);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);
struct f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0,
                                              int dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
cl_mem futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                 struct futhark_f32_2d *arr);
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr);
struct i32_2d ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx,
                                          int32_t *data, int dim0, int dim1);
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0,
                                              int dim1);
int futhark_free_i32_2d(struct futhark_context *ctx,
                        struct futhark_i32_2d *arr);
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data);
cl_mem futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                 struct futhark_i32_2d *arr);
int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                              struct futhark_i32_2d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0,
                       struct futhark_i32_2d **out1, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const int32_t in2, const
                       struct futhark_f32_1d *in3, const
                       struct futhark_f32_1d *in4, const int32_t in5);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
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

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims-1; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    shape[i] = 0;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

// Reading little-endian byte sequences.  On big-endian hosts, we flip
// the resulting bytes.

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_2byte(void* dest) {
  uint16_t x;
  int num_elems_read = fread(&x, 2, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  *(uint16_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_4byte(void* dest) {
  uint32_t x;
  int num_elems_read = fread(&x, 4, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  *(uint32_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_8byte(void* dest) {
  uint64_t x;
  int num_elems_read = fread(&x, 8, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  *(uint64_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int write_byte(void* dest) {
  int num_elems_written = fwrite(dest, 1, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_2byte(void* dest) {
  uint16_t x = *(uint16_t*)dest;
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  int num_elems_written = fwrite(&x, 2, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_4byte(void* dest) {
  uint32_t x = *(uint32_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  int num_elems_written = fwrite(&x, 4, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_8byte(void* dest) {
  uint64_t x = *(uint64_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  int num_elems_written = fwrite(&x, 8, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
  const writer write_bin; // Write in binary format.
  const bin_reader read_bin; // Read in binary format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = read_le_8byte(&bin_shape);
    if (ret != 0) { panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i); }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    char* elems = (char*) *data;
    for (uint64_t i=0; i<elem_count; i++) {
      char* elem = elems+(i*elem_size);
      for (unsigned int j=0; j<elem_size/2; j++) {
        char head = elem[j];
        int tail_index = elem_size-1-j;
        elem[j] = elem[tail_index];
        elem[tail_index] = head;
      }
    }
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[]");
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  fwrite(shape, sizeof(int64_t), rank, out);

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    return expected_type->read_bin(dest);
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"platform", required_argument, NULL,
                                            7}, {"device", required_argument,
                                                 NULL, 8},
                                           {"default-group-size",
                                            required_argument, NULL, 9},
                                           {"default-num-groups",
                                            required_argument, NULL, 10},
                                           {"default-tile-size",
                                            required_argument, NULL, 11},
                                           {"default-threshold",
                                            required_argument, NULL, 12},
                                           {"dump-opencl", required_argument,
                                            NULL, 13}, {"load-opencl",
                                                        required_argument, NULL,
                                                        14}, {"print-sizes",
                                                              no_argument, NULL,
                                                              15}, {"size",
                                                                    required_argument,
                                                                    NULL, 16},
                                           {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bp:d:", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e')
            entry_point = optarg;
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'p')
            futhark_context_config_set_platform(cfg, optarg);
        if (ch == 8 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 9)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 11)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 12)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 13)
            futhark_context_config_dump_program_to(cfg, optarg);
        if (ch == 14)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 15) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++) {
                if (strcmp(futhark_get_size_entry(i), entry_point) == 0)
                    printf("%s (%s)\n", futhark_get_size_name(i),
                           futhark_get_size_class(i));
            }
            exit(0);
        }
        if (ch == 16) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    panic(1, "Unknown size: %s\n", name);
            } else
                panic(1, "Invalid argument for size option: %s\n", optarg);
        }
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?')
            panic(-1, "Unknown option %s\n", argv[optind - 1]);
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_9187;
    int64_t read_shape_9188[1];
    float *read_arr_9189 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9189, read_shape_9188, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9190;
    int64_t read_shape_9191[1];
    float *read_arr_9192 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9192, read_shape_9191, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    int32_t read_value_9193;
    
    if (read_scalar(&i32_info, &read_value_9193) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9194;
    int64_t read_shape_9195[1];
    float *read_arr_9196 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9196, read_shape_9195, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9197;
    int64_t read_shape_9198[1];
    float *read_arr_9199 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9199, read_shape_9198, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4, "[]",
              f32_info.type_name, strerror(errno));
    
    int32_t read_value_9200;
    
    if (read_scalar(&i32_info, &read_value_9200) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 5,
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_9201;
    struct futhark_i32_2d *result_9202;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_9187 = futhark_new_f32_1d(ctx, read_arr_9189,
                                                     read_shape_9188[0])) != 0);
        assert((read_value_9190 = futhark_new_f32_1d(ctx, read_arr_9192,
                                                     read_shape_9191[0])) != 0);
        ;
        assert((read_value_9194 = futhark_new_f32_1d(ctx, read_arr_9196,
                                                     read_shape_9195[0])) != 0);
        assert((read_value_9197 = futhark_new_f32_1d(ctx, read_arr_9199,
                                                     read_shape_9198[0])) != 0);
        ;
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_9201, &result_9202, read_value_9187,
                               read_value_9190, read_value_9193,
                               read_value_9194, read_value_9197,
                               read_value_9200);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9187) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9190) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9194) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9197) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, result_9201) == 0);
        assert(futhark_free_i32_2d(ctx, result_9202) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_9187 = futhark_new_f32_1d(ctx, read_arr_9189,
                                                     read_shape_9188[0])) != 0);
        assert((read_value_9190 = futhark_new_f32_1d(ctx, read_arr_9192,
                                                     read_shape_9191[0])) != 0);
        ;
        assert((read_value_9194 = futhark_new_f32_1d(ctx, read_arr_9196,
                                                     read_shape_9195[0])) != 0);
        assert((read_value_9197 = futhark_new_f32_1d(ctx, read_arr_9199,
                                                     read_shape_9198[0])) != 0);
        ;
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_9201, &result_9202, read_value_9187,
                               read_value_9190, read_value_9193,
                               read_value_9194, read_value_9197,
                               read_value_9200);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9187) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9190) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9194) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9197) == 0);
        ;
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_9201) == 0);
            assert(futhark_free_i32_2d(ctx, result_9202) == 0);
        }
    }
    free(read_arr_9189);
    free(read_arr_9192);
    ;
    free(read_arr_9196);
    free(read_arr_9199);
    ;
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_9201)[0] *
                            futhark_shape_f32_2d(ctx, result_9201)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_9201, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_9201), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_2d(ctx,
                                                                    result_9202)[0] *
                              futhark_shape_i32_2d(ctx, result_9202)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_2d(ctx, result_9202, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_2d(ctx, result_9202), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_9201) == 0);
    assert(futhark_free_i32_2d(ctx, result_9202) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    
    int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
    entry_point_fun *entry_point_fun = NULL;
    
    for (int i = 0; i < num_entry_points; i++) {
        if (strcmp(entry_points[i].name, entry_point) == 0) {
            entry_point_fun = entry_points[i].fun;
            break;
        }
    }
    if (entry_point_fun == NULL) {
        fprintf(stderr,
                "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                entry_point);
        for (int i = 0; i < num_entry_points; i++)
            fprintf(stderr, "%s\n", entry_points[i].name);
        return 1;
    }
    entry_point_fun(ctx);
    if (runtime_file != NULL)
        fclose(runtime_file);
    futhark_debugging_report(ctx);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
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
            "urn x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#define group_sizze_8700 (group_size_8699)\n__kernel void fut_kernel_map_transpose_f32(__global float *odata,\n                                           uint odata_offset, __g",
            "lobal\n                                           float *idata, uint idata_offset,\n                                           uint width, uint height,\n                                           uint input_size, uint output_size,\n                                           __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_global_id(0);\n    y_index = get_global_id(1);\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_i32(__global int32_t *odata,\n                                           uint odata_offset, __global\n                                           int32_t *idata, uint idata_offset,\n                                           uint width, uint height,\n                                           uint input_size, uint output_size,\n                                           __local int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    ui",
            "nt our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_global_id(0);\n    y_index = get_global_id(1);\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowheight_f32(__global float *odata,\n                                                     uint odata_offset, __global\n                                                     float *idata,\n                                                     uint idata_offset,\n                                                     uint width, uint height,\n                                                     uint input_size,\n                                                     uint output_size,\n                                                     uint mulx, __local\n                                                     float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata",
            "_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +\n        get_local_id(1) % mulx * FUT_BLOCK_DIM;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +\n        get_local_id(0) % mulx * FUT_BLOCK_DIM;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowheight_i32(__global int32_t *odata,\n                                                     uint odata_offset, __global\n                                                     int32_t *idata,\n                                                     uint idata_offset,\n                                                     uint width, uint height,\n                                                     uint input_size,\n                                                     uint output_size,\n                                                     uint mulx, __local\n                                                     int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n   ",
            " \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +\n        get_local_id(1) % mulx * FUT_BLOCK_DIM;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +\n        get_local_id(0) % mulx * FUT_BLOCK_DIM;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowwidth_f32(__global float *odata,\n                                                    uint odata_offset, __global\n                                                    float *idata,\n                                                    uint idata_offset,\n                                                    uint width, uint height,\n                                                    uint input_size,\n                                                    uint output_size, uint muly,\n                                                    __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint",
            " our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +\n        get_local_id(0) % muly * FUT_BLOCK_DIM;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +\n        get_local_id(1) % muly * FUT_BLOCK_DIM;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowwidth_i32(__global int32_t *odata,\n                                                    uint odata_offset, __global\n                                                    int32_t *idata,\n                                                    uint idata_offset,\n                                                    uint width, uint height,\n                                                    uint input_size,\n                                                    uint output_size, uint muly,\n                                                    __local int32_t *block)\n{\n    uint x_index;\n  ",
            "  uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +\n        get_local_id(0) % muly * FUT_BLOCK_DIM;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +\n        get_local_id(1) % muly * FUT_BLOCK_DIM;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_small_f32(__global float *odata,\n                                                 uint odata_offset, __global\n                                                 float *idata,\n                                                 uint idata_offset,\n                                                 uint num_arrays, uint width,\n                                                 uint height, uint input_size,\n                                                 uint output_size)\n{\n    uint our_array_offset = get_global_id(0) / (height * width) * (height *\n           ",
            "                                                        width);\n    uint x_index = get_global_id(0) % (height * width) / height;\n    uint y_index = get_global_id(0) % height;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays.\n    odata += our_array_offset;\n    idata += our_array_offset;\n    \n    uint index_in = y_index * width + x_index;\n    uint index_out = x_index * height + y_index;\n    \n    if (get_global_id(0) < input_size)\n        odata[index_out] = idata[index_in];\n}\n__kernel void fut_kernel_map_transpose_small_i32(__global int32_t *odata,\n                                                 uint odata_offset, __global\n                                                 int32_t *idata,\n                                                 uint idata_offset,\n                                                 uint num_arrays, uint width,\n                                                 uint height, uint input_size,\n                                                 uint output_size)\n{\n    uint our_array_offset = get_global_id(0) / (height * width) * (height *\n                                                                   width);\n    uint x_index = get_global_id(0) % (height * width) / height;\n    uint y_index = get_global_id(0) % height;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays.\n    odata += our_array_offset;\n    idata += our_array_offset;\n    \n    uint index_in = y_index * width + x_index;\n    uint index_out = x_index * height + y_index;\n    \n    if (get_global_id(0) < input_size)\n        odata[index_out] = idata[index_in];\n}\n__kernel void map_kernel_8655(int32_t sizze_8380, int32_t sizze_8381, __global\n                              unsigned char *mem_9006, __global\n  ",
            "                            unsigned char *mem_9014)\n{\n    int32_t wave_sizze_9091;\n    int32_t group_sizze_9092;\n    bool thread_active_9093;\n    int32_t gtid_8646;\n    int32_t gtid_8647;\n    int32_t global_tid_8655;\n    int32_t local_tid_8656;\n    int32_t group_id_8657;\n    \n    global_tid_8655 = get_global_id(0);\n    local_tid_8656 = get_local_id(0);\n    group_sizze_9092 = get_local_size(0);\n    wave_sizze_9091 = LOCKSTEP_WIDTH;\n    group_id_8657 = get_group_id(0);\n    gtid_8646 = squot32(global_tid_8655, sizze_8381);\n    gtid_8647 = global_tid_8655 - squot32(global_tid_8655, sizze_8381) *\n        sizze_8381;\n    thread_active_9093 = slt32(gtid_8646, sizze_8380) && slt32(gtid_8647,\n                                                               sizze_8381);\n    \n    float res_8658;\n    \n    if (thread_active_9093) {\n        res_8658 = *(__global float *) &mem_9006[gtid_8646 * 4];\n    }\n    if (thread_active_9093) {\n        *(__global float *) &mem_9014[(gtid_8646 * sizze_8381 + gtid_8647) *\n                                      4] = res_8658;\n    }\n}\n__kernel void map_kernel_8671(int32_t sizze_8380, int32_t sizze_8381, __global\n                              unsigned char *mem_9003, __global\n                              unsigned char *mem_9010)\n{\n    int32_t wave_sizze_9088;\n    int32_t group_sizze_9089;\n    bool thread_active_9090;\n    int32_t gtid_8662;\n    int32_t gtid_8663;\n    int32_t global_tid_8671;\n    int32_t local_tid_8672;\n    int32_t group_id_8673;\n    \n    global_tid_8671 = get_global_id(0);\n    local_tid_8672 = get_local_id(0);\n    group_sizze_9089 = get_local_size(0);\n    wave_sizze_9088 = LOCKSTEP_WIDTH;\n    group_id_8673 = get_group_id(0);\n    gtid_8662 = squot32(global_tid_8671, sizze_8381);\n    gtid_8663 = global_tid_8671 - squot32(global_tid_8671, sizze_8381) *\n        sizze_8381;\n    thread_active_9090 = slt32(gtid_8662, sizze_8380) && slt32(gtid_8663,\n                                                               sizze_8381);\n    \n    float r",
            "es_8674;\n    \n    if (thread_active_9090) {\n        res_8674 = *(__global float *) &mem_9003[gtid_8662 * 4];\n    }\n    if (thread_active_9090) {\n        *(__global float *) &mem_9010[(gtid_8662 * sizze_8381 + gtid_8663) *\n                                      4] = res_8674;\n    }\n}\n__kernel void map_kernel_8683(int32_t sizze_8380, __global\n                              unsigned char *angles_mem_8998, __global\n                              unsigned char *mem_9003, __global\n                              unsigned char *mem_9006)\n{\n    int32_t wave_sizze_9085;\n    int32_t group_sizze_9086;\n    bool thread_active_9087;\n    int32_t gtid_8676;\n    int32_t global_tid_8683;\n    int32_t local_tid_8684;\n    int32_t group_id_8685;\n    \n    global_tid_8683 = get_global_id(0);\n    local_tid_8684 = get_local_id(0);\n    group_sizze_9086 = get_local_size(0);\n    wave_sizze_9085 = LOCKSTEP_WIDTH;\n    group_id_8685 = get_group_id(0);\n    gtid_8676 = global_tid_8683;\n    thread_active_9087 = slt32(gtid_8676, sizze_8380);\n    \n    float x_8686;\n    float res_8687;\n    float res_8688;\n    float res_8689;\n    \n    if (thread_active_9087) {\n        x_8686 = *(__global float *) &angles_mem_8998[gtid_8676 * 4];\n        res_8687 = 1.7453292e-2F * x_8686;\n        res_8688 = futrts_sin32(res_8687);\n        res_8689 = futrts_cos32(res_8687);\n    }\n    if (thread_active_9087) {\n        *(__global float *) &mem_9003[gtid_8676 * 4] = res_8688;\n    }\n    if (thread_active_9087) {\n        *(__global float *) &mem_9006[gtid_8676 * 4] = res_8689;\n    }\n}\n__kernel void map_kernel_8705(float res_8388, float res_8389, float res_8399,\n                              int32_t res_8402, int32_t nesting_sizze_8664,\n                              __global unsigned char *mem_9016, __global\n                              unsigned char *mem_9019, __global\n                              unsigned char *mem_9022, __global\n                              unsigned char *mem_9024, __global\n                              unsigne",
            "d char *mem_9027, __global\n                              unsigned char *mem_9030, __global\n                              unsigned char *mem_9033, __global\n                              unsigned char *mem_9055, __global\n                              unsigned char *mem_9059)\n{\n    int32_t wave_sizze_9097;\n    int32_t group_sizze_9098;\n    bool thread_active_9099;\n    int32_t gtid_8698;\n    int32_t global_tid_8705;\n    int32_t local_tid_8706;\n    int32_t group_id_8707;\n    \n    global_tid_8705 = get_global_id(0);\n    local_tid_8706 = get_local_id(0);\n    group_sizze_9098 = get_local_size(0);\n    wave_sizze_9097 = LOCKSTEP_WIDTH;\n    group_id_8707 = get_group_id(0);\n    gtid_8698 = global_tid_8705;\n    thread_active_9099 = slt32(gtid_8698, nesting_sizze_8664);\n    \n    bool cond_8708;\n    float res_8709;\n    float res_8710;\n    bool res_8711;\n    float res_8712;\n    bool cond_8715;\n    float res_8716;\n    int32_t res_8717;\n    float res_8718;\n    bool res_8719;\n    float res_8720;\n    float res_8727;\n    float res_8728;\n    bool cond_8751;\n    bool res_8752;\n    bool x_8753;\n    bool cond_8754;\n    bool res_8755;\n    bool x_8756;\n    bool cond_8757;\n    bool res_8758;\n    bool x_8759;\n    bool x_8760;\n    bool x_8761;\n    bool y_8762;\n    bool res_8763;\n    bool x_8764;\n    float y_8765;\n    bool res_8766;\n    float res_8769;\n    float res_8770;\n    float res_8771;\n    float res_8772;\n    int32_t res_8773;\n    \n    if (thread_active_9099) {\n        cond_8708 = *(__global bool *) &mem_9016[gtid_8698];\n        res_8709 = *(__global float *) &mem_9019[gtid_8698 * 4];\n        res_8710 = *(__global float *) &mem_9022[gtid_8698 * 4];\n        res_8711 = *(__global bool *) &mem_9024[gtid_8698];\n        res_8712 = *(__global float *) &mem_9027[gtid_8698 * 4];\n        for (int32_t i_9100 = 0; i_9100 < res_8402; i_9100++) {\n            *(__global float *) &mem_9030[(group_id_8707 * (res_8402 *\n                                                            group_sizze_8700) +\n        ",
            "                                   i_9100 * group_sizze_8700 +\n                                           local_tid_8706) * 4] = -1.0F;\n        }\n        for (int32_t i_9101 = 0; i_9101 < res_8402; i_9101++) {\n            *(__global int32_t *) &mem_9033[(group_id_8707 * (res_8402 *\n                                                              group_sizze_8700) +\n                                             i_9101 * group_sizze_8700 +\n                                             local_tid_8706) * 4] = -1;\n        }\n        cond_8715 = res_8712 < 0.0F;\n        if (cond_8715) {\n            res_8716 = -1.0F;\n        } else {\n            res_8716 = 1.0F;\n        }\n        res_8717 = fptosi_f32_i32(res_8709);\n        res_8718 = sitofp_i32_f32(res_8717);\n        res_8719 = 0.0F <= res_8709;\n        if (res_8719) {\n            bool res_8721;\n            float res_8722;\n            \n            res_8721 = res_8718 < res_8709;\n            if (res_8721) {\n                res_8722 = res_8718;\n            } else {\n                res_8722 = res_8709;\n            }\n            res_8720 = res_8722;\n        } else {\n            bool res_8723;\n            float res_8724;\n            \n            res_8723 = res_8709 < res_8718;\n            if (res_8723) {\n                int32_t res_8725;\n                float res_8726;\n                \n                res_8725 = res_8717 - 1;\n                res_8726 = sitofp_i32_f32(res_8725);\n                res_8724 = res_8726;\n            } else {\n                res_8724 = res_8709;\n            }\n            res_8720 = res_8724;\n        }\n        res_8727 = 1.0F + res_8720;\n        if (cond_8715) {\n            int32_t res_8729;\n            float res_8730;\n            bool res_8731;\n            float res_8732;\n            float res_8739;\n            \n            res_8729 = fptosi_f32_i32(res_8710);\n            res_8730 = sitofp_i32_f32(res_8729);\n            res_8731 = 0.0F <= res_8710;\n            if (res_8731) {\n                bool res_8733;\n",
            "                float res_8734;\n                \n                res_8733 = res_8730 < res_8710;\n                if (res_8733) {\n                    int32_t res_8735;\n                    float res_8736;\n                    \n                    res_8735 = 1 + res_8729;\n                    res_8736 = sitofp_i32_f32(res_8735);\n                    res_8734 = res_8736;\n                } else {\n                    res_8734 = res_8710;\n                }\n                res_8732 = res_8734;\n            } else {\n                bool res_8737;\n                float res_8738;\n                \n                res_8737 = res_8710 < res_8730;\n                if (res_8737) {\n                    res_8738 = res_8730;\n                } else {\n                    res_8738 = res_8710;\n                }\n                res_8732 = res_8738;\n            }\n            res_8739 = res_8732 - 1.0F;\n            res_8728 = res_8739;\n        } else {\n            int32_t res_8740;\n            float res_8741;\n            bool res_8742;\n            float res_8743;\n            float res_8750;\n            \n            res_8740 = fptosi_f32_i32(res_8710);\n            res_8741 = sitofp_i32_f32(res_8740);\n            res_8742 = 0.0F <= res_8710;\n            if (res_8742) {\n                bool res_8744;\n                float res_8745;\n                \n                res_8744 = res_8741 < res_8710;\n                if (res_8744) {\n                    res_8745 = res_8741;\n                } else {\n                    res_8745 = res_8710;\n                }\n                res_8743 = res_8745;\n            } else {\n                bool res_8746;\n                float res_8747;\n                \n                res_8746 = res_8710 < res_8741;\n                if (res_8746) {\n                    int32_t res_8748;\n                    float res_8749;\n                    \n                    res_8748 = res_8740 - 1;\n                    res_8749 = sitofp_i32_f32(res_8748);\n                    res_8747 = res_8749;\n   ",
            "             } else {\n                    res_8747 = res_8710;\n                }\n                res_8743 = res_8747;\n            }\n            res_8750 = 1.0F + res_8743;\n            res_8728 = res_8750;\n        }\n        cond_8751 = res_8399 <= res_8709;\n        res_8752 = res_8709 < res_8389;\n        x_8753 = cond_8751 && res_8752;\n        cond_8754 = res_8399 < res_8710;\n        res_8755 = res_8710 <= res_8389;\n        x_8756 = cond_8754 && res_8755;\n        cond_8757 = res_8399 <= res_8710;\n        res_8758 = res_8710 < res_8389;\n        x_8759 = cond_8757 && res_8758;\n        x_8760 = cond_8715 && x_8756;\n        x_8761 = !cond_8715;\n        y_8762 = x_8759 && x_8761;\n        res_8763 = x_8760 || y_8762;\n        x_8764 = x_8753 && res_8763;\n        y_8765 = 1.0F / res_8712;\n        \n        bool loop_while_8774;\n        float focusPoint_8777;\n        float focusPoint_8778;\n        float anchorX_8779;\n        float anchorY_8780;\n        int32_t write_index_8781;\n        \n        loop_while_8774 = x_8764;\n        focusPoint_8777 = res_8709;\n        focusPoint_8778 = res_8710;\n        anchorX_8779 = res_8727;\n        anchorY_8780 = res_8728;\n        write_index_8781 = 0;\n        while (loop_while_8774) {\n            float arg_8782 = res_8389 + focusPoint_8778;\n            int32_t res_8783 = fptosi_f32_i32(arg_8782);\n            float res_8784 = sitofp_i32_f32(res_8783);\n            bool res_8785 = 0.0F <= arg_8782;\n            float res_8786;\n            \n            if (res_8785) {\n                bool res_8787;\n                float res_8788;\n                \n                res_8787 = res_8784 < arg_8782;\n                if (res_8787) {\n                    res_8788 = res_8784;\n                } else {\n                    res_8788 = arg_8782;\n                }\n                res_8786 = res_8788;\n            } else {\n                bool res_8789;\n                float res_8790;\n                \n                res_8789 = arg_8782 < res_8784;\n                if",
            " (res_8789) {\n                    int32_t res_8791;\n                    float res_8792;\n                    \n                    res_8791 = res_8783 - 1;\n                    res_8792 = sitofp_i32_f32(res_8791);\n                    res_8790 = res_8792;\n                } else {\n                    res_8790 = arg_8782;\n                }\n                res_8786 = res_8790;\n            }\n            \n            int32_t res_8793 = fptosi_f32_i32(focusPoint_8778);\n            float res_8794 = sitofp_i32_f32(res_8793);\n            bool res_8795 = 0.0F <= focusPoint_8778;\n            float res_8796;\n            \n            if (res_8795) {\n                bool res_8797;\n                float res_8798;\n                \n                res_8797 = res_8794 < focusPoint_8778;\n                if (res_8797) {\n                    res_8798 = res_8794;\n                } else {\n                    res_8798 = focusPoint_8778;\n                }\n                res_8796 = res_8798;\n            } else {\n                bool res_8799;\n                float res_8800;\n                \n                res_8799 = focusPoint_8778 < res_8794;\n                if (res_8799) {\n                    int32_t res_8801;\n                    float res_8802;\n                    \n                    res_8801 = res_8793 - 1;\n                    res_8802 = sitofp_i32_f32(res_8801);\n                    res_8800 = res_8802;\n                } else {\n                    res_8800 = focusPoint_8778;\n                }\n                res_8796 = res_8800;\n            }\n            \n            float x_8803 = focusPoint_8778 - res_8796;\n            bool res_8804 = x_8803 == 0.0F;\n            bool x_8805 = cond_8715 && res_8804;\n            float res_8806;\n            \n            if (x_8805) {\n                float res_8807 = res_8786 - 1.0F;\n                \n                res_8806 = res_8807;\n            } else {\n                res_8806 = res_8786;\n            }\n            \n            float arg_8808 = res_8389 ",
            "+ focusPoint_8777;\n            int32_t res_8809 = fptosi_f32_i32(arg_8808);\n            float res_8810 = sitofp_i32_f32(res_8809);\n            bool res_8811 = 0.0F <= arg_8808;\n            float res_8812;\n            \n            if (res_8811) {\n                bool res_8813;\n                float res_8814;\n                \n                res_8813 = res_8810 < arg_8808;\n                if (res_8813) {\n                    res_8814 = res_8810;\n                } else {\n                    res_8814 = arg_8808;\n                }\n                res_8812 = res_8814;\n            } else {\n                bool res_8815;\n                float res_8816;\n                \n                res_8815 = arg_8808 < res_8810;\n                if (res_8815) {\n                    int32_t res_8817;\n                    float res_8818;\n                    \n                    res_8817 = res_8809 - 1;\n                    res_8818 = sitofp_i32_f32(res_8817);\n                    res_8816 = res_8818;\n                } else {\n                    res_8816 = arg_8808;\n                }\n                res_8812 = res_8816;\n            }\n            \n            float y_8819 = res_8388 * res_8806;\n            float arg_8820 = res_8812 + y_8819;\n            int32_t res_8821 = fptosi_f32_i32(arg_8820);\n            float res_8822;\n            \n            if (res_8711) {\n                res_8822 = 1.0F;\n            } else {\n                float res_8823;\n                \n                if (cond_8708) {\n                    res_8823 = 0.0F;\n                } else {\n                    float x_8824;\n                    float res_8825;\n                    \n                    x_8824 = anchorX_8779 - focusPoint_8777;\n                    res_8825 = res_8712 * x_8824;\n                    res_8823 = res_8825;\n                }\n                res_8822 = res_8823;\n            }\n            \n            float res_8826;\n            \n            if (res_8711) {\n                res_8826 = 0.0F;\n            } else",
            " {\n                float res_8827;\n                \n                if (cond_8708) {\n                    res_8827 = 1.0F;\n                } else {\n                    float x_8828;\n                    float res_8829;\n                    \n                    x_8828 = anchorY_8780 - focusPoint_8778;\n                    res_8829 = y_8765 * x_8828;\n                    res_8827 = res_8829;\n                }\n                res_8826 = res_8827;\n            }\n            \n            float res_8830 = focusPoint_8778 + res_8822;\n            float res_8831 = focusPoint_8777 + res_8826;\n            float x_8832 = anchorX_8779 - focusPoint_8777;\n            float x_8833 = fpow32(x_8832, 2.0F);\n            float x_8834 = res_8830 - focusPoint_8778;\n            float y_8835 = fpow32(x_8834, 2.0F);\n            float arg_8836 = x_8833 + y_8835;\n            float res_8837;\n            \n            res_8837 = futrts_sqrt32(arg_8836);\n            \n            float x_8838 = res_8831 - focusPoint_8777;\n            float x_8839 = fpow32(x_8838, 2.0F);\n            float x_8840 = anchorY_8780 - focusPoint_8778;\n            float y_8841 = fpow32(x_8840, 2.0F);\n            float arg_8842 = x_8839 + y_8841;\n            float res_8843;\n            \n            res_8843 = futrts_sqrt32(arg_8842);\n            \n            float res_8846;\n            float res_8847;\n            float res_8848;\n            float res_8849;\n            int32_t res_8850;\n            \n            if (cond_8708) {\n                float res_8853;\n                int32_t res_8854;\n                \n                *(__global float *) &mem_9030[(group_id_8707 * (res_8402 *\n                                                                group_sizze_8700) +\n                                               write_index_8781 *\n                                               group_sizze_8700 +\n                                               local_tid_8706) * 4] = res_8837;\n                *(__global int32_t *) &mem_9033[(group_id_",
            "8707 * (res_8402 *\n                                                                  group_sizze_8700) +\n                                                 write_index_8781 *\n                                                 group_sizze_8700 +\n                                                 local_tid_8706) * 4] =\n                    res_8821;\n                res_8853 = 1.0F + anchorX_8779;\n                res_8854 = 1 + write_index_8781;\n                res_8846 = anchorX_8779;\n                res_8847 = res_8830;\n                res_8848 = res_8853;\n                res_8849 = anchorY_8780;\n                res_8850 = res_8854;\n            } else {\n                float res_8857;\n                float res_8858;\n                float res_8859;\n                float res_8860;\n                int32_t res_8861;\n                \n                if (res_8711) {\n                    float res_8864;\n                    int32_t res_8865;\n                    \n                    *(__global float *) &mem_9030[(group_id_8707 * (res_8402 *\n                                                                    group_sizze_8700) +\n                                                   write_index_8781 *\n                                                   group_sizze_8700 +\n                                                   local_tid_8706) * 4] =\n                        res_8843;\n                    *(__global int32_t *) &mem_9033[(group_id_8707 * (res_8402 *\n                                                                      group_sizze_8700) +\n                                                     write_index_8781 *\n                                                     group_sizze_8700 +\n                                                     local_tid_8706) * 4] =\n                        res_8821;\n                    res_8864 = res_8716 + anchorY_8780;\n                    res_8865 = 1 + write_index_8781;\n                    res_8857 = res_8831;\n                    res_8858 = anchorY_8780;\n     ",
            "               res_8859 = anchorX_8779;\n                    res_8860 = res_8864;\n                    res_8861 = res_8865;\n                } else {\n                    float arg_8866;\n                    float res_8867;\n                    bool cond_8868;\n                    float res_8871;\n                    float res_8872;\n                    float res_8873;\n                    float res_8874;\n                    int32_t res_8875;\n                    \n                    arg_8866 = res_8837 - res_8843;\n                    res_8867 = (float) fabs(arg_8866);\n                    cond_8868 = 1.0e-9F < res_8867;\n                    if (cond_8868) {\n                        bool cond_8876;\n                        float res_8877;\n                        float res_8878;\n                        float res_8881;\n                        float res_8882;\n                        int32_t res_8883;\n                        \n                        cond_8876 = res_8837 < res_8843;\n                        if (cond_8876) {\n                            res_8877 = anchorX_8779;\n                        } else {\n                            res_8877 = res_8831;\n                        }\n                        if (cond_8876) {\n                            res_8878 = res_8830;\n                        } else {\n                            res_8878 = anchorY_8780;\n                        }\n                        if (cond_8876) {\n                            float res_8886;\n                            int32_t res_8887;\n                            \n                            *(__global float *) &mem_9030[(group_id_8707 *\n                                                           (res_8402 *\n                                                            group_sizze_8700) +\n                                                           write_index_8781 *\n                                                           group_sizze_8700 +\n                                                           local_tid_8706) *",
            "\n                                                          4] = res_8837;\n                            *(__global int32_t *) &mem_9033[(group_id_8707 *\n                                                             (res_8402 *\n                                                              group_sizze_8700) +\n                                                             write_index_8781 *\n                                                             group_sizze_8700 +\n                                                             local_tid_8706) *\n                                                            4] = res_8821;\n                            res_8886 = 1.0F + anchorX_8779;\n                            res_8887 = 1 + write_index_8781;\n                            res_8881 = res_8886;\n                            res_8882 = anchorY_8780;\n                            res_8883 = res_8887;\n                        } else {\n                            float res_8890;\n                            int32_t res_8891;\n                            \n                            *(__global float *) &mem_9030[(group_id_8707 *\n                                                           (res_8402 *\n                                                            group_sizze_8700) +\n                                                           write_index_8781 *\n                                                           group_sizze_8700 +\n                                                           local_tid_8706) *\n                                                          4] = res_8843;\n                            *(__global int32_t *) &mem_9033[(group_id_8707 *\n                                                             (res_8402 *\n                                                              group_sizze_8700) +\n                                                             write_index_8781 *\n                                                             group_sizze_8700 +\n                                ",
            "                             local_tid_8706) *\n                                                            4] = res_8821;\n                            res_8890 = res_8716 + anchorY_8780;\n                            res_8891 = 1 + write_index_8781;\n                            res_8881 = anchorX_8779;\n                            res_8882 = res_8890;\n                            res_8883 = res_8891;\n                        }\n                        res_8871 = res_8877;\n                        res_8872 = res_8878;\n                        res_8873 = res_8881;\n                        res_8874 = res_8882;\n                        res_8875 = res_8883;\n                    } else {\n                        float res_8894;\n                        float res_8895;\n                        int32_t res_8896;\n                        \n                        *(__global float *) &mem_9030[(group_id_8707 *\n                                                       (res_8402 *\n                                                        group_sizze_8700) +\n                                                       write_index_8781 *\n                                                       group_sizze_8700 +\n                                                       local_tid_8706) * 4] =\n                            res_8837;\n                        *(__global int32_t *) &mem_9033[(group_id_8707 *\n                                                         (res_8402 *\n                                                          group_sizze_8700) +\n                                                         write_index_8781 *\n                                                         group_sizze_8700 +\n                                                         local_tid_8706) * 4] =\n                            res_8821;\n                        res_8894 = 1.0F + anchorX_8779;\n                        res_8895 = res_8716 + anchorY_8780;\n                        res_8896 = 1 + write_index_8781;\n                        res_8871 =",
            " anchorX_8779;\n                        res_8872 = res_8830;\n                        res_8873 = res_8894;\n                        res_8874 = res_8895;\n                        res_8875 = res_8896;\n                    }\n                    res_8857 = res_8871;\n                    res_8858 = res_8872;\n                    res_8859 = res_8873;\n                    res_8860 = res_8874;\n                    res_8861 = res_8875;\n                }\n                res_8846 = res_8857;\n                res_8847 = res_8858;\n                res_8848 = res_8859;\n                res_8849 = res_8860;\n                res_8850 = res_8861;\n            }\n            \n            bool cond_8897 = res_8399 <= res_8846;\n            bool res_8898 = res_8846 < res_8389;\n            bool x_8899 = cond_8897 && res_8898;\n            bool cond_8900 = res_8399 < res_8847;\n            bool res_8901 = res_8847 <= res_8389;\n            bool x_8902 = cond_8900 && res_8901;\n            bool cond_8903 = res_8399 <= res_8847;\n            bool res_8904 = res_8847 < res_8389;\n            bool x_8905 = cond_8903 && res_8904;\n            bool x_8906 = cond_8715 && x_8902;\n            bool y_8907 = x_8761 && x_8905;\n            bool res_8908 = x_8906 || y_8907;\n            bool x_8909 = x_8899 && res_8908;\n            bool loop_while_tmp_9102 = x_8909;\n            float focusPoint_tmp_9105 = res_8846;\n            float focusPoint_tmp_9106 = res_8847;\n            float anchorX_tmp_9107 = res_8848;\n            float anchorY_tmp_9108 = res_8849;\n            int32_t write_index_tmp_9109;\n            \n            write_index_tmp_9109 = res_8850;\n            loop_while_8774 = loop_while_tmp_9102;\n            focusPoint_8777 = focusPoint_tmp_9105;\n            focusPoint_8778 = focusPoint_tmp_9106;\n            anchorX_8779 = anchorX_tmp_9107;\n            anchorY_8780 = anchorY_tmp_9108;\n            write_index_8781 = write_index_tmp_9109;\n        }\n        res_8766 = loop_while_8774;\n        res_8769 = focusPoint_8777;",
            "\n        res_8770 = focusPoint_8778;\n        res_8771 = anchorX_8779;\n        res_8772 = anchorY_8780;\n        res_8773 = write_index_8781;\n    }\n    if (thread_active_9099) {\n        for (int32_t i_9110 = 0; i_9110 < res_8402; i_9110++) {\n            *(__global float *) &mem_9055[(i_9110 * nesting_sizze_8664 +\n                                           gtid_8698) * 4] = *(__global\n                                                               float *) &mem_9030[(group_id_8707 *\n                                                                                   (res_8402 *\n                                                                                    group_sizze_8700) +\n                                                                                   i_9110 *\n                                                                                   group_sizze_8700 +\n                                                                                   local_tid_8706) *\n                                                                                  4];\n        }\n    }\n    if (thread_active_9099) {\n        for (int32_t i_9111 = 0; i_9111 < res_8402; i_9111++) {\n            *(__global int32_t *) &mem_9059[(i_9111 * nesting_sizze_8664 +\n                                             gtid_8698) * 4] = *(__global\n                                                                 int32_t *) &mem_9033[(group_id_8707 *\n                                                                                       (res_8402 *\n                                                                                        group_sizze_8700) +\n                                                                                       i_9111 *\n                                                                                       group_sizze_8700 +\n                                                                                       local_tid_8706) *\n                                                            ",
            "                          4];\n        }\n    }\n}\n__kernel void map_kernel_8944(int32_t sizze_8381, float res_8389,\n                              float res_8399, int32_t nesting_sizze_8664,\n                              __global unsigned char *rays_mem_9000, __global\n                              unsigned char *mem_9010, __global\n                              unsigned char *mem_9014, __global\n                              unsigned char *mem_9016, __global\n                              unsigned char *mem_9019, __global\n                              unsigned char *mem_9022, __global\n                              unsigned char *mem_9024, __global\n                              unsigned char *mem_9027)\n{\n    int32_t wave_sizze_9094;\n    int32_t group_sizze_9095;\n    bool thread_active_9096;\n    int32_t gtid_8937;\n    int32_t global_tid_8944;\n    int32_t local_tid_8945;\n    int32_t group_id_8946;\n    \n    global_tid_8944 = get_global_id(0);\n    local_tid_8945 = get_local_id(0);\n    group_sizze_9095 = get_local_size(0);\n    wave_sizze_9094 = LOCKSTEP_WIDTH;\n    group_id_8946 = get_group_id(0);\n    gtid_8937 = global_tid_8944;\n    thread_active_9096 = slt32(gtid_8937, nesting_sizze_8664);\n    \n    int32_t new_index_8977;\n    int32_t binop_y_8979;\n    int32_t new_index_8980;\n    float copy_p_8947;\n    float copy_p_8948;\n    float x_8949;\n    bool cond_8950;\n    float res_8951;\n    bool cond_8955;\n    float res_8956;\n    float res_8960;\n    float res_8964;\n    bool cond_8965;\n    bool res_8966;\n    bool x_8967;\n    float res_8968;\n    float res_8969;\n    float res_8973;\n    bool res_8974;\n    float y_8975;\n    float res_8976;\n    \n    if (thread_active_9096) {\n        new_index_8977 = squot32(gtid_8937, sizze_8381);\n        binop_y_8979 = sizze_8381 * new_index_8977;\n        new_index_8980 = gtid_8937 - binop_y_8979;\n        copy_p_8947 = *(__global float *) &mem_9010[(new_index_8977 *\n                                                     sizze_8381 +\n                           ",
            "                          new_index_8980) * 4];\n        copy_p_8948 = *(__global float *) &mem_9014[(new_index_8977 *\n                                                     sizze_8381 +\n                                                     new_index_8980) * 4];\n        x_8949 = *(__global float *) &rays_mem_9000[new_index_8980 * 4];\n        cond_8950 = copy_p_8947 == 0.0F;\n        if (cond_8950) {\n            res_8951 = x_8949;\n        } else {\n            float y_8952;\n            float x_8953;\n            float res_8954;\n            \n            y_8952 = res_8399 * copy_p_8948;\n            x_8953 = x_8949 - y_8952;\n            res_8954 = x_8953 / copy_p_8947;\n            res_8951 = res_8954;\n        }\n        cond_8955 = copy_p_8948 == 0.0F;\n        if (cond_8955) {\n            res_8956 = x_8949;\n        } else {\n            float y_8957;\n            float x_8958;\n            float res_8959;\n            \n            y_8957 = res_8399 * copy_p_8947;\n            x_8958 = x_8949 - y_8957;\n            res_8959 = x_8958 / copy_p_8948;\n            res_8956 = res_8959;\n        }\n        if (cond_8955) {\n            res_8960 = x_8949;\n        } else {\n            float y_8961;\n            float x_8962;\n            float res_8963;\n            \n            y_8961 = res_8389 * copy_p_8947;\n            x_8962 = x_8949 - y_8961;\n            res_8963 = x_8962 / copy_p_8948;\n            res_8960 = res_8963;\n        }\n        res_8964 = (float) fabs(res_8951);\n        cond_8965 = res_8964 <= res_8389;\n        res_8966 = !cond_8950;\n        x_8967 = cond_8965 && res_8966;\n        if (x_8967) {\n            res_8968 = res_8399;\n            res_8969 = res_8951;\n        } else {\n            bool cond_8970;\n            float res_8971;\n            float res_8972;\n            \n            cond_8970 = res_8956 <= res_8960;\n            if (cond_8970) {\n                res_8971 = res_8956;\n            } else {\n                res_8971 = res_8960;\n            }\n            if (cond_8970) {\n    ",
            "            res_8972 = res_8399;\n            } else {\n                res_8972 = res_8389;\n            }\n            res_8968 = res_8971;\n            res_8969 = res_8972;\n        }\n        res_8973 = (float) fabs(copy_p_8948);\n        res_8974 = res_8973 == 1.0F;\n        y_8975 = 0.0F - copy_p_8947;\n        res_8976 = copy_p_8948 / y_8975;\n    }\n    if (thread_active_9096) {\n        *(__global bool *) &mem_9016[gtid_8937] = cond_8955;\n    }\n    if (thread_active_9096) {\n        *(__global float *) &mem_9019[gtid_8937 * 4] = res_8968;\n    }\n    if (thread_active_9096) {\n        *(__global float *) &mem_9022[gtid_8937 * 4] = res_8969;\n    }\n    if (thread_active_9096) {\n        *(__global bool *) &mem_9024[gtid_8937] = res_8974;\n    }\n    if (thread_active_9096) {\n        *(__global float *) &mem_9027[gtid_8937 * 4] = res_8976;\n    }\n}\n",
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
static const char *size_names[] = {"group_size_8649", "group_size_8665",
                                   "group_size_8677", "group_size_8699",
                                   "group_size_8938"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size"};
static const char *size_entry_points[] = {"main", "main", "main", "main",
                                          "main"};
int futhark_get_num_sizes(void)
{
    return 5;
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
    size_t group_sizze_8649;
    size_t group_sizze_8665;
    size_t group_sizze_8677;
    size_t group_sizze_8699;
    size_t group_sizze_8938;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[5];
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    opencl_config_init(&cfg->opencl, 5, size_names, cfg->sizes, size_classes,
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
    for (int i = 0; i < 5; i++) {
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
    cl_kernel fut_kernel_map_transpose_f32;
    int fut_kernel_map_transpose_f32_total_runtime;
    int fut_kernel_map_transpose_f32_runs;
    cl_kernel fut_kernel_map_transpose_i32;
    int fut_kernel_map_transpose_i32_total_runtime;
    int fut_kernel_map_transpose_i32_runs;
    cl_kernel fut_kernel_map_transpose_lowheight_f32;
    int fut_kernel_map_transpose_lowheight_f32_total_runtime;
    int fut_kernel_map_transpose_lowheight_f32_runs;
    cl_kernel fut_kernel_map_transpose_lowheight_i32;
    int fut_kernel_map_transpose_lowheight_i32_total_runtime;
    int fut_kernel_map_transpose_lowheight_i32_runs;
    cl_kernel fut_kernel_map_transpose_lowwidth_f32;
    int fut_kernel_map_transpose_lowwidth_f32_total_runtime;
    int fut_kernel_map_transpose_lowwidth_f32_runs;
    cl_kernel fut_kernel_map_transpose_lowwidth_i32;
    int fut_kernel_map_transpose_lowwidth_i32_total_runtime;
    int fut_kernel_map_transpose_lowwidth_i32_runs;
    cl_kernel fut_kernel_map_transpose_small_f32;
    int fut_kernel_map_transpose_small_f32_total_runtime;
    int fut_kernel_map_transpose_small_f32_runs;
    cl_kernel fut_kernel_map_transpose_small_i32;
    int fut_kernel_map_transpose_small_i32_total_runtime;
    int fut_kernel_map_transpose_small_i32_runs;
    cl_kernel map_kernel_8655;
    int map_kernel_8655_total_runtime;
    int map_kernel_8655_runs;
    cl_kernel map_kernel_8671;
    int map_kernel_8671_total_runtime;
    int map_kernel_8671_runs;
    cl_kernel map_kernel_8683;
    int map_kernel_8683_total_runtime;
    int map_kernel_8683_runs;
    cl_kernel map_kernel_8705;
    int map_kernel_8705_total_runtime;
    int map_kernel_8705_runs;
    cl_kernel map_kernel_8944;
    int map_kernel_8944_total_runtime;
    int map_kernel_8944_runs;
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
    ctx->fut_kernel_map_transpose_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_f32_runs = 0;
    ctx->fut_kernel_map_transpose_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_i32_runs = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowheight_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowheight_i32_runs = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowwidth_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowwidth_i32_runs = 0;
    ctx->fut_kernel_map_transpose_small_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_small_f32_runs = 0;
    ctx->fut_kernel_map_transpose_small_i32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_small_i32_runs = 0;
    ctx->map_kernel_8655_total_runtime = 0;
    ctx->map_kernel_8655_runs = 0;
    ctx->map_kernel_8671_total_runtime = 0;
    ctx->map_kernel_8671_runs = 0;
    ctx->map_kernel_8683_total_runtime = 0;
    ctx->map_kernel_8683_runs = 0;
    ctx->map_kernel_8705_total_runtime = 0;
    ctx->map_kernel_8705_runs = 0;
    ctx->map_kernel_8944_total_runtime = 0;
    ctx->map_kernel_8944_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    
    {
        ctx->fut_kernel_map_transpose_f32 = clCreateKernel(prog,
                                                           "fut_kernel_map_transpose_f32",
                                                           &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_f32");
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
        ctx->fut_kernel_map_transpose_lowheight_f32 = clCreateKernel(prog,
                                                                     "fut_kernel_map_transpose_lowheight_f32",
                                                                     &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowheight_f32");
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
        ctx->fut_kernel_map_transpose_lowwidth_f32 = clCreateKernel(prog,
                                                                    "fut_kernel_map_transpose_lowwidth_f32",
                                                                    &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowwidth_f32");
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
        ctx->fut_kernel_map_transpose_small_f32 = clCreateKernel(prog,
                                                                 "fut_kernel_map_transpose_small_f32",
                                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_small_f32");
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
        ctx->map_kernel_8655 = clCreateKernel(prog, "map_kernel_8655", &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_8655");
    }
    {
        ctx->map_kernel_8671 = clCreateKernel(prog, "map_kernel_8671", &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_8671");
    }
    {
        ctx->map_kernel_8683 = clCreateKernel(prog, "map_kernel_8683", &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_8683");
    }
    {
        ctx->map_kernel_8705 = clCreateKernel(prog, "map_kernel_8705", &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_8705");
    }
    {
        ctx->map_kernel_8944 = clCreateKernel(prog, "map_kernel_8944", &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_8944");
    }
    ctx->sizes.group_sizze_8649 = cfg->sizes[0];
    ctx->sizes.group_sizze_8665 = cfg->sizes[1];
    ctx->sizes.group_sizze_8677 = cfg->sizes[2];
    ctx->sizes.group_sizze_8699 = cfg->sizes[3];
    ctx->sizes.group_sizze_8938 = cfg->sizes[4];
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
                "Kernel fut_kernel_map_transpose_f32           executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_f32_runs,
                (long) ctx->fut_kernel_map_transpose_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_f32_total_runtime);
        ctx->total_runtime += ctx->fut_kernel_map_transpose_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_f32_runs;
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
                "Kernel fut_kernel_map_transpose_lowheight_f32 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_lowheight_f32_runs,
                (long) ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_lowheight_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_lowheight_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_lowheight_f32_runs;
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
                "Kernel fut_kernel_map_transpose_lowwidth_f32  executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_lowwidth_f32_runs,
                (long) ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_lowwidth_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_lowwidth_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_lowwidth_f32_runs;
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
                "Kernel fut_kernel_map_transpose_small_f32     executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_small_f32_runs,
                (long) ctx->fut_kernel_map_transpose_small_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_small_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_small_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_small_f32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_small_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_small_f32_runs;
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
                "Kernel map_kernel_8655                        executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_8655_runs,
                (long) ctx->map_kernel_8655_total_runtime /
                (ctx->map_kernel_8655_runs !=
                 0 ? ctx->map_kernel_8655_runs : 1),
                (long) ctx->map_kernel_8655_total_runtime);
        ctx->total_runtime += ctx->map_kernel_8655_total_runtime;
        ctx->total_runs += ctx->map_kernel_8655_runs;
        fprintf(stderr,
                "Kernel map_kernel_8671                        executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_8671_runs,
                (long) ctx->map_kernel_8671_total_runtime /
                (ctx->map_kernel_8671_runs !=
                 0 ? ctx->map_kernel_8671_runs : 1),
                (long) ctx->map_kernel_8671_total_runtime);
        ctx->total_runtime += ctx->map_kernel_8671_total_runtime;
        ctx->total_runs += ctx->map_kernel_8671_runs;
        fprintf(stderr,
                "Kernel map_kernel_8683                        executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_8683_runs,
                (long) ctx->map_kernel_8683_total_runtime /
                (ctx->map_kernel_8683_runs !=
                 0 ? ctx->map_kernel_8683_runs : 1),
                (long) ctx->map_kernel_8683_total_runtime);
        ctx->total_runtime += ctx->map_kernel_8683_total_runtime;
        ctx->total_runs += ctx->map_kernel_8683_runs;
        fprintf(stderr,
                "Kernel map_kernel_8705                        executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_8705_runs,
                (long) ctx->map_kernel_8705_total_runtime /
                (ctx->map_kernel_8705_runs !=
                 0 ? ctx->map_kernel_8705_runs : 1),
                (long) ctx->map_kernel_8705_total_runtime);
        ctx->total_runtime += ctx->map_kernel_8705_total_runtime;
        ctx->total_runs += ctx->map_kernel_8705_runs;
        fprintf(stderr,
                "Kernel map_kernel_8944                        executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_8944_runs,
                (long) ctx->map_kernel_8944_total_runtime /
                (ctx->map_kernel_8944_runs !=
                 0 ? ctx->map_kernel_8944_runs : 1),
                (long) ctx->map_kernel_8944_total_runtime);
        ctx->total_runtime += ctx->map_kernel_8944_total_runtime;
        ctx->total_runs += ctx->map_kernel_8944_runs;
        if (ctx->debugging)
            fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_map_transpose_opencl_f32(struct futhark_context *ctx,
                                           struct memblock_device destmem_0,
                                           int32_t destoffset_1,
                                           struct memblock_device srcmem_2,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8);
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
                       int64_t *out_out_memsizze_9152,
                       struct memblock_device *out_mem_p_9153,
                       int32_t *out_out_arrsizze_9154,
                       int32_t *out_out_arrsizze_9155,
                       int64_t *out_out_memsizze_9156,
                       struct memblock_device *out_mem_p_9157,
                       int32_t *out_out_arrsizze_9158,
                       int32_t *out_out_arrsizze_9159,
                       int64_t matRow_mem_sizze_8993,
                       struct memblock_device matRow_mem_8994,
                       int64_t vect_mem_sizze_8995,
                       struct memblock_device vect_mem_8996,
                       int64_t angles_mem_sizze_8997,
                       struct memblock_device angles_mem_8998,
                       int64_t rays_mem_sizze_8999,
                       struct memblock_device rays_mem_9000, int32_t sizze_8378,
                       int32_t sizze_8379, int32_t sizze_8380,
                       int32_t sizze_8381, int32_t num_cols_8384,
                       int32_t gridsizze_8387);
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
static int futrts_map_transpose_opencl_f32(struct futhark_context *ctx,
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
            if (in_elems_7 * sizeof(float) > 0) {
                OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                             srcmem_2.mem,
                                                             destmem_0.mem,
                                                             srcoffset_3,
                                                             destoffset_1,
                                                             in_elems_7 *
                                                             sizeof(float), 0,
                                                             NULL, NULL));
                if (ctx->debugging)
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            }
        } else {
            if (sle32(x_elems_5, squot32(16, 2)) && slt32(16, y_elems_6)) {
                int32_t muly_9 = squot32(16, x_elems_5);
                int32_t new_height_10;
                
                new_height_10 = squot32(y_elems_6 + muly_9 - 1, muly_9);
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        0,
                                                        sizeof(destmem_0.mem),
                                                        &destmem_0.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        1, sizeof(destoffset_1),
                                                        &destoffset_1));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        2, sizeof(srcmem_2.mem),
                                                        &srcmem_2.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        3, sizeof(srcoffset_3),
                                                        &srcoffset_3));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        4, sizeof(x_elems_5),
                                                        &x_elems_5));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        5, sizeof(y_elems_6),
                                                        &y_elems_6));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        6, sizeof(in_elems_7),
                                                        &in_elems_7));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        7, sizeof(out_elems_8),
                                                        &out_elems_8));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        8, sizeof(muly_9),
                                                        &muly_9));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                        9, 272 * sizeof(float),
                                                        NULL));
                if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16), 16)) *
                    (new_height_10 + srem32(16 - srem32(new_height_10, 16),
                                            16)) * num_arrays_4 != 0) {
                    const size_t global_work_sizze_9112[3] = {x_elems_5 +
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
                    const size_t local_work_sizze_9116[3] = {16, 16, 1};
                    int64_t time_start_9113 = 0, time_end_9114 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_lowwidth_f32");
                        fprintf(stderr, "%zu", global_work_sizze_9112[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_9112[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_9112[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_9116[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_9116[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_9116[2]);
                        fprintf(stderr, "].\n");
                        time_start_9113 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                                    3, NULL,
                                                                    global_work_sizze_9112,
                                                                    local_work_sizze_9116,
                                                                    0, NULL,
                                                                    NULL));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_9114 = get_wall_time();
                        
                        long time_diff_9115 = time_end_9114 - time_start_9113;
                        
                        ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime +=
                            time_diff_9115;
                        ctx->fut_kernel_map_transpose_lowwidth_f32_runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "fut_kernel_map_transpose_lowwidth_f32",
                                time_diff_9115);
                    }
                }
            } else {
                if (sle32(y_elems_6, squot32(16, 2)) && slt32(16, x_elems_5)) {
                    int32_t mulx_11 = squot32(16, y_elems_6);
                    int32_t new_width_12;
                    
                    new_width_12 = squot32(x_elems_5 + mulx_11 - 1, mulx_11);
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            0,
                                                            sizeof(destmem_0.mem),
                                                            &destmem_0.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            1,
                                                            sizeof(destoffset_1),
                                                            &destoffset_1));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            2,
                                                            sizeof(srcmem_2.mem),
                                                            &srcmem_2.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            3,
                                                            sizeof(srcoffset_3),
                                                            &srcoffset_3));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            4,
                                                            sizeof(x_elems_5),
                                                            &x_elems_5));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            5,
                                                            sizeof(y_elems_6),
                                                            &y_elems_6));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            6,
                                                            sizeof(in_elems_7),
                                                            &in_elems_7));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            7,
                                                            sizeof(out_elems_8),
                                                            &out_elems_8));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            8, sizeof(mulx_11),
                                                            &mulx_11));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                            9, 272 *
                                                            sizeof(float),
                                                            NULL));
                    if (1 * (new_width_12 + srem32(16 - srem32(new_width_12,
                                                               16), 16)) *
                        (y_elems_6 + srem32(16 - srem32(y_elems_6, 16), 16)) *
                        num_arrays_4 != 0) {
                        const size_t global_work_sizze_9117[3] = {new_width_12 +
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
                        const size_t local_work_sizze_9121[3] = {16, 16, 1};
                        int64_t time_start_9118 = 0, time_end_9119 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "fut_kernel_map_transpose_lowheight_f32");
                            fprintf(stderr, "%zu", global_work_sizze_9117[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_9117[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_9117[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_9121[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_9121[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_9121[2]);
                            fprintf(stderr, "].\n");
                            time_start_9118 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->fut_kernel_map_transpose_lowheight_f32,
                                                                        3, NULL,
                                                                        global_work_sizze_9117,
                                                                        local_work_sizze_9121,
                                                                        0, NULL,
                                                                        NULL));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_9119 = get_wall_time();
                            
                            long time_diff_9120 = time_end_9119 -
                                 time_start_9118;
                            
                            ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime +=
                                time_diff_9120;
                            ctx->fut_kernel_map_transpose_lowheight_f32_runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_lowheight_f32",
                                    time_diff_9120);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, squot32(16, 2)) && sle32(y_elems_6,
                                                                  squot32(16,
                                                                          2))) {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                0,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                2,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                3,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                4,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                5,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                6,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                7,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                                8,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        if (1 * (num_arrays_4 * x_elems_5 * y_elems_6 +
                                 srem32(256 - srem32(num_arrays_4 * x_elems_5 *
                                                     y_elems_6, 256), 256)) !=
                            0) {
                            const size_t global_work_sizze_9122[1] =
                                         {num_arrays_4 * x_elems_5 * y_elems_6 +
                                         srem32(256 - srem32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256),
                                                256)};
                            const size_t local_work_sizze_9126[1] = {256};
                            int64_t time_start_9123 = 0, time_end_9124 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_small_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9122[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9126[0]);
                                fprintf(stderr, "].\n");
                                time_start_9123 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->fut_kernel_map_transpose_small_f32,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_9122,
                                                                            local_work_sizze_9126,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_9124 = get_wall_time();
                                
                                long time_diff_9125 = time_end_9124 -
                                     time_start_9123;
                                
                                ctx->fut_kernel_map_transpose_small_f32_total_runtime +=
                                    time_diff_9125;
                                ctx->fut_kernel_map_transpose_small_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_small_f32",
                                        time_diff_9125);
                            }
                        }
                    } else {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                0,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                2,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                3,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                6,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                7,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                                8, 272 *
                                                                sizeof(float),
                                                                NULL));
                        if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16),
                                                    16)) * (y_elems_6 +
                                                            srem32(16 -
                                                                   srem32(y_elems_6,
                                                                          16),
                                                                   16)) *
                            num_arrays_4 != 0) {
                            const size_t global_work_sizze_9127[3] =
                                         {x_elems_5 + srem32(16 -
                                                             srem32(x_elems_5,
                                                                    16), 16),
                                          y_elems_6 + srem32(16 -
                                                             srem32(y_elems_6,
                                                                    16), 16),
                                          num_arrays_4};
                            const size_t local_work_sizze_9131[3] = {16, 16, 1};
                            int64_t time_start_9128 = 0, time_end_9129 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9127[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9127[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9127[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9131[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9131[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9131[2]);
                                fprintf(stderr, "].\n");
                                time_start_9128 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->fut_kernel_map_transpose_f32,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_9127,
                                                                            local_work_sizze_9131,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_9129 = get_wall_time();
                                
                                long time_diff_9130 = time_end_9129 -
                                     time_start_9128;
                                
                                ctx->fut_kernel_map_transpose_f32_total_runtime +=
                                    time_diff_9130;
                                ctx->fut_kernel_map_transpose_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_f32",
                                        time_diff_9130);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
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
                    const size_t global_work_sizze_9132[3] = {x_elems_5 +
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
                    const size_t local_work_sizze_9136[3] = {16, 16, 1};
                    int64_t time_start_9133 = 0, time_end_9134 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_lowwidth_i32");
                        fprintf(stderr, "%zu", global_work_sizze_9132[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_9132[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_9132[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_9136[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_9136[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_9136[2]);
                        fprintf(stderr, "].\n");
                        time_start_9133 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->fut_kernel_map_transpose_lowwidth_i32,
                                                                    3, NULL,
                                                                    global_work_sizze_9132,
                                                                    local_work_sizze_9136,
                                                                    0, NULL,
                                                                    NULL));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_9134 = get_wall_time();
                        
                        long time_diff_9135 = time_end_9134 - time_start_9133;
                        
                        ctx->fut_kernel_map_transpose_lowwidth_i32_total_runtime +=
                            time_diff_9135;
                        ctx->fut_kernel_map_transpose_lowwidth_i32_runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "fut_kernel_map_transpose_lowwidth_i32",
                                time_diff_9135);
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
                        const size_t global_work_sizze_9137[3] = {new_width_12 +
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
                        const size_t local_work_sizze_9141[3] = {16, 16, 1};
                        int64_t time_start_9138 = 0, time_end_9139 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "fut_kernel_map_transpose_lowheight_i32");
                            fprintf(stderr, "%zu", global_work_sizze_9137[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_9137[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_9137[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_9141[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_9141[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_9141[2]);
                            fprintf(stderr, "].\n");
                            time_start_9138 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->fut_kernel_map_transpose_lowheight_i32,
                                                                        3, NULL,
                                                                        global_work_sizze_9137,
                                                                        local_work_sizze_9141,
                                                                        0, NULL,
                                                                        NULL));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_9139 = get_wall_time();
                            
                            long time_diff_9140 = time_end_9139 -
                                 time_start_9138;
                            
                            ctx->fut_kernel_map_transpose_lowheight_i32_total_runtime +=
                                time_diff_9140;
                            ctx->fut_kernel_map_transpose_lowheight_i32_runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_lowheight_i32",
                                    time_diff_9140);
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
                            const size_t global_work_sizze_9142[1] =
                                         {num_arrays_4 * x_elems_5 * y_elems_6 +
                                         srem32(256 - srem32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256),
                                                256)};
                            const size_t local_work_sizze_9146[1] = {256};
                            int64_t time_start_9143 = 0, time_end_9144 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_small_i32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9142[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9146[0]);
                                fprintf(stderr, "].\n");
                                time_start_9143 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->fut_kernel_map_transpose_small_i32,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_9142,
                                                                            local_work_sizze_9146,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_9144 = get_wall_time();
                                
                                long time_diff_9145 = time_end_9144 -
                                     time_start_9143;
                                
                                ctx->fut_kernel_map_transpose_small_i32_total_runtime +=
                                    time_diff_9145;
                                ctx->fut_kernel_map_transpose_small_i32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_small_i32",
                                        time_diff_9145);
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
                            const size_t global_work_sizze_9147[3] =
                                         {x_elems_5 + srem32(16 -
                                                             srem32(x_elems_5,
                                                                    16), 16),
                                          y_elems_6 + srem32(16 -
                                                             srem32(y_elems_6,
                                                                    16), 16),
                                          num_arrays_4};
                            const size_t local_work_sizze_9151[3] = {16, 16, 1};
                            int64_t time_start_9148 = 0, time_end_9149 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_i32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9147[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9147[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_9147[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9151[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9151[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_9151[2]);
                                fprintf(stderr, "].\n");
                                time_start_9148 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->fut_kernel_map_transpose_i32,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_9147,
                                                                            local_work_sizze_9151,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_9149 = get_wall_time();
                                
                                long time_diff_9150 = time_end_9149 -
                                     time_start_9148;
                                
                                ctx->fut_kernel_map_transpose_i32_total_runtime +=
                                    time_diff_9150;
                                ctx->fut_kernel_map_transpose_i32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_i32",
                                        time_diff_9150);
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
                       int64_t *out_out_memsizze_9152,
                       struct memblock_device *out_mem_p_9153,
                       int32_t *out_out_arrsizze_9154,
                       int32_t *out_out_arrsizze_9155,
                       int64_t *out_out_memsizze_9156,
                       struct memblock_device *out_mem_p_9157,
                       int32_t *out_out_arrsizze_9158,
                       int32_t *out_out_arrsizze_9159,
                       int64_t matRow_mem_sizze_8993,
                       struct memblock_device matRow_mem_8994,
                       int64_t vect_mem_sizze_8995,
                       struct memblock_device vect_mem_8996,
                       int64_t angles_mem_sizze_8997,
                       struct memblock_device angles_mem_8998,
                       int64_t rays_mem_sizze_8999,
                       struct memblock_device rays_mem_9000, int32_t sizze_8378,
                       int32_t sizze_8379, int32_t sizze_8380,
                       int32_t sizze_8381, int32_t num_cols_8384,
                       int32_t gridsizze_8387)
{
    int64_t out_memsizze_9078;
    struct memblock_device out_mem_9077;
    
    out_mem_9077.references = NULL;
    
    int32_t out_arrsizze_9079;
    int32_t out_arrsizze_9080;
    int64_t out_memsizze_9082;
    struct memblock_device out_mem_9081;
    
    out_mem_9081.references = NULL;
    
    int32_t out_arrsizze_9083;
    int32_t out_arrsizze_9084;
    float res_8388 = sitofp_i32_f32(gridsizze_8387);
    float res_8389 = res_8388 / 2.0F;
    int32_t group_sizze_8678;
    
    group_sizze_8678 = ctx->sizes.group_sizze_8677;
    
    int32_t y_8679 = group_sizze_8678 - 1;
    int32_t x_8680 = sizze_8380 + y_8679;
    int32_t num_groups_8681 = squot32(x_8680, group_sizze_8678);
    int32_t num_threads_8682 = group_sizze_8678 * num_groups_8681;
    int64_t binop_x_9002 = sext_i32_i64(sizze_8380);
    int64_t bytes_9001 = 4 * binop_x_9002;
    struct memblock_device mem_9003;
    
    mem_9003.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9003, bytes_9001, "mem_9003"))
        return 1;
    
    struct memblock_device mem_9006;
    
    mem_9006.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9006, bytes_9001, "mem_9006"))
        return 1;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8683, 0,
                                            sizeof(sizze_8380), &sizze_8380));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8683, 1,
                                            sizeof(angles_mem_8998.mem),
                                            &angles_mem_8998.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8683, 2,
                                            sizeof(mem_9003.mem),
                                            &mem_9003.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8683, 3,
                                            sizeof(mem_9006.mem),
                                            &mem_9006.mem));
    if (1 * (num_groups_8681 * group_sizze_8678) != 0) {
        const size_t global_work_sizze_9160[1] = {num_groups_8681 *
                     group_sizze_8678};
        const size_t local_work_sizze_9164[1] = {group_sizze_8678};
        int64_t time_start_9161 = 0, time_end_9162 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_8683");
            fprintf(stderr, "%zu", global_work_sizze_9160[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9164[0]);
            fprintf(stderr, "].\n");
            time_start_9161 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_kernel_8683, 1,
                                                        NULL,
                                                        global_work_sizze_9160,
                                                        local_work_sizze_9164,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_9162 = get_wall_time();
            
            long time_diff_9163 = time_end_9162 - time_start_9161;
            
            ctx->map_kernel_8683_total_runtime += time_diff_9163;
            ctx->map_kernel_8683_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_8683",
                    time_diff_9163);
        }
    }
    
    int32_t nesting_sizze_8664 = sizze_8380 * sizze_8381;
    int32_t group_sizze_8666;
    
    group_sizze_8666 = ctx->sizes.group_sizze_8665;
    
    int32_t y_8667 = group_sizze_8666 - 1;
    int32_t x_8668 = nesting_sizze_8664 + y_8667;
    int32_t num_groups_8669 = squot32(x_8668, group_sizze_8666);
    int32_t num_threads_8670 = group_sizze_8666 * num_groups_8669;
    int64_t binop_x_9009 = sext_i32_i64(nesting_sizze_8664);
    int64_t bytes_9007 = 4 * binop_x_9009;
    struct memblock_device mem_9010;
    
    mem_9010.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9010, bytes_9007, "mem_9010"))
        return 1;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8671, 0,
                                            sizeof(sizze_8380), &sizze_8380));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8671, 1,
                                            sizeof(sizze_8381), &sizze_8381));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8671, 2,
                                            sizeof(mem_9003.mem),
                                            &mem_9003.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8671, 3,
                                            sizeof(mem_9010.mem),
                                            &mem_9010.mem));
    if (1 * (num_groups_8669 * group_sizze_8666) != 0) {
        const size_t global_work_sizze_9165[1] = {num_groups_8669 *
                     group_sizze_8666};
        const size_t local_work_sizze_9169[1] = {group_sizze_8666};
        int64_t time_start_9166 = 0, time_end_9167 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_8671");
            fprintf(stderr, "%zu", global_work_sizze_9165[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9169[0]);
            fprintf(stderr, "].\n");
            time_start_9166 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_kernel_8671, 1,
                                                        NULL,
                                                        global_work_sizze_9165,
                                                        local_work_sizze_9169,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_9167 = get_wall_time();
            
            long time_diff_9168 = time_end_9167 - time_start_9166;
            
            ctx->map_kernel_8671_total_runtime += time_diff_9168;
            ctx->map_kernel_8671_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_8671",
                    time_diff_9168);
        }
    }
    if (memblock_unref_device(ctx, &mem_9003, "mem_9003") != 0)
        return 1;
    
    int32_t group_sizze_8650;
    
    group_sizze_8650 = ctx->sizes.group_sizze_8649;
    
    int32_t y_8651 = group_sizze_8650 - 1;
    int32_t x_8652 = y_8651 + nesting_sizze_8664;
    int32_t num_groups_8653 = squot32(x_8652, group_sizze_8650);
    int32_t num_threads_8654 = group_sizze_8650 * num_groups_8653;
    struct memblock_device mem_9014;
    
    mem_9014.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9014, bytes_9007, "mem_9014"))
        return 1;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8655, 0,
                                            sizeof(sizze_8380), &sizze_8380));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8655, 1,
                                            sizeof(sizze_8381), &sizze_8381));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8655, 2,
                                            sizeof(mem_9006.mem),
                                            &mem_9006.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8655, 3,
                                            sizeof(mem_9014.mem),
                                            &mem_9014.mem));
    if (1 * (num_groups_8653 * group_sizze_8650) != 0) {
        const size_t global_work_sizze_9170[1] = {num_groups_8653 *
                     group_sizze_8650};
        const size_t local_work_sizze_9174[1] = {group_sizze_8650};
        int64_t time_start_9171 = 0, time_end_9172 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_8655");
            fprintf(stderr, "%zu", global_work_sizze_9170[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9174[0]);
            fprintf(stderr, "].\n");
            time_start_9171 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_kernel_8655, 1,
                                                        NULL,
                                                        global_work_sizze_9170,
                                                        local_work_sizze_9174,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_9172 = get_wall_time();
            
            long time_diff_9173 = time_end_9172 - time_start_9171;
            
            ctx->map_kernel_8655_total_runtime += time_diff_9173;
            ctx->map_kernel_8655_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_8655",
                    time_diff_9173);
        }
    }
    if (memblock_unref_device(ctx, &mem_9006, "mem_9006") != 0)
        return 1;
    
    float res_8399 = 0.0F - res_8389;
    float x_8400 = 2.0F * res_8388;
    float arg_8401 = x_8400 - 1.0F;
    int32_t res_8402 = fptosi_f32_i32(arg_8401);
    int32_t group_sizze_8939;
    
    group_sizze_8939 = ctx->sizes.group_sizze_8938;
    
    int32_t y_8940 = group_sizze_8939 - 1;
    int32_t x_8941 = nesting_sizze_8664 + y_8940;
    int32_t num_groups_8942 = squot32(x_8941, group_sizze_8939);
    int32_t num_threads_8943 = group_sizze_8939 * num_groups_8942;
    struct memblock_device mem_9016;
    
    mem_9016.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9016, binop_x_9009, "mem_9016"))
        return 1;
    
    struct memblock_device mem_9019;
    
    mem_9019.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9019, bytes_9007, "mem_9019"))
        return 1;
    
    struct memblock_device mem_9022;
    
    mem_9022.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9022, bytes_9007, "mem_9022"))
        return 1;
    
    struct memblock_device mem_9024;
    
    mem_9024.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9024, binop_x_9009, "mem_9024"))
        return 1;
    
    struct memblock_device mem_9027;
    
    mem_9027.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9027, bytes_9007, "mem_9027"))
        return 1;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 0,
                                            sizeof(sizze_8381), &sizze_8381));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 1,
                                            sizeof(res_8389), &res_8389));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 2,
                                            sizeof(res_8399), &res_8399));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 3,
                                            sizeof(nesting_sizze_8664),
                                            &nesting_sizze_8664));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 4,
                                            sizeof(rays_mem_9000.mem),
                                            &rays_mem_9000.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 5,
                                            sizeof(mem_9010.mem),
                                            &mem_9010.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 6,
                                            sizeof(mem_9014.mem),
                                            &mem_9014.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 7,
                                            sizeof(mem_9016.mem),
                                            &mem_9016.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 8,
                                            sizeof(mem_9019.mem),
                                            &mem_9019.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 9,
                                            sizeof(mem_9022.mem),
                                            &mem_9022.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 10,
                                            sizeof(mem_9024.mem),
                                            &mem_9024.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8944, 11,
                                            sizeof(mem_9027.mem),
                                            &mem_9027.mem));
    if (1 * (num_groups_8942 * group_sizze_8939) != 0) {
        const size_t global_work_sizze_9175[1] = {num_groups_8942 *
                     group_sizze_8939};
        const size_t local_work_sizze_9179[1] = {group_sizze_8939};
        int64_t time_start_9176 = 0, time_end_9177 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_8944");
            fprintf(stderr, "%zu", global_work_sizze_9175[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9179[0]);
            fprintf(stderr, "].\n");
            time_start_9176 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_kernel_8944, 1,
                                                        NULL,
                                                        global_work_sizze_9175,
                                                        local_work_sizze_9179,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_9177 = get_wall_time();
            
            long time_diff_9178 = time_end_9177 - time_start_9176;
            
            ctx->map_kernel_8944_total_runtime += time_diff_9178;
            ctx->map_kernel_8944_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_8944",
                    time_diff_9178);
        }
    }
    if (memblock_unref_device(ctx, &mem_9010, "mem_9010") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9014, "mem_9014") != 0)
        return 1;
    
    int32_t group_sizze_8700;
    
    group_sizze_8700 = ctx->sizes.group_sizze_8699;
    
    int32_t y_8701 = group_sizze_8700 - 1;
    int32_t x_8702 = nesting_sizze_8664 + y_8701;
    int32_t num_groups_8703 = squot32(x_8702, group_sizze_8700);
    int32_t num_threads_8704 = group_sizze_8700 * num_groups_8703;
    int32_t convop_x_9053 = res_8402 * nesting_sizze_8664;
    int64_t binop_x_9054 = sext_i32_i64(convop_x_9053);
    int64_t bytes_9052 = 4 * binop_x_9054;
    struct memblock_device mem_9055;
    
    mem_9055.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9055, bytes_9052, "mem_9055"))
        return 1;
    
    struct memblock_device mem_9059;
    
    mem_9059.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9059, bytes_9052, "mem_9059"))
        return 1;
    
    int64_t binop_x_9029 = sext_i32_i64(res_8402);
    int64_t bytes_9028 = 4 * binop_x_9029;
    int64_t num_threads64_9074 = sext_i32_i64(num_threads_8704);
    int64_t total_sizze_9075 = bytes_9028 * num_threads64_9074;
    struct memblock_device mem_9030;
    
    mem_9030.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9030, total_sizze_9075, "mem_9030"))
        return 1;
    
    int64_t total_sizze_9076 = bytes_9028 * num_threads64_9074;
    struct memblock_device mem_9033;
    
    mem_9033.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9033, total_sizze_9076, "mem_9033"))
        return 1;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 0,
                                            sizeof(res_8388), &res_8388));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 1,
                                            sizeof(res_8389), &res_8389));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 2,
                                            sizeof(res_8399), &res_8399));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 3,
                                            sizeof(res_8402), &res_8402));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 4,
                                            sizeof(nesting_sizze_8664),
                                            &nesting_sizze_8664));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 5,
                                            sizeof(mem_9016.mem),
                                            &mem_9016.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 6,
                                            sizeof(mem_9019.mem),
                                            &mem_9019.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 7,
                                            sizeof(mem_9022.mem),
                                            &mem_9022.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 8,
                                            sizeof(mem_9024.mem),
                                            &mem_9024.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 9,
                                            sizeof(mem_9027.mem),
                                            &mem_9027.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 10,
                                            sizeof(mem_9030.mem),
                                            &mem_9030.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 11,
                                            sizeof(mem_9033.mem),
                                            &mem_9033.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 12,
                                            sizeof(mem_9055.mem),
                                            &mem_9055.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_kernel_8705, 13,
                                            sizeof(mem_9059.mem),
                                            &mem_9059.mem));
    if (1 * (num_groups_8703 * group_sizze_8700) != 0) {
        const size_t global_work_sizze_9180[1] = {num_groups_8703 *
                     group_sizze_8700};
        const size_t local_work_sizze_9184[1] = {group_sizze_8700};
        int64_t time_start_9181 = 0, time_end_9182 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_8705");
            fprintf(stderr, "%zu", global_work_sizze_9180[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9184[0]);
            fprintf(stderr, "].\n");
            time_start_9181 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_kernel_8705, 1,
                                                        NULL,
                                                        global_work_sizze_9180,
                                                        local_work_sizze_9184,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_9182 = get_wall_time();
            
            long time_diff_9183 = time_end_9182 - time_start_9181;
            
            ctx->map_kernel_8705_total_runtime += time_diff_9183;
            ctx->map_kernel_8705_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_8705",
                    time_diff_9183);
        }
    }
    if (memblock_unref_device(ctx, &mem_9016, "mem_9016") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9019, "mem_9019") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9022, "mem_9022") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9024, "mem_9024") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9027, "mem_9027") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9030, "mem_9030") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9033, "mem_9033") != 0)
        return 1;
    
    struct memblock_device mem_9063;
    
    mem_9063.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9063, bytes_9052, "mem_9063"))
        return 1;
    
    int call_ret_9185 = futrts_map_transpose_opencl_f32(ctx, mem_9063, 0,
                                                        mem_9055, 0, 1,
                                                        nesting_sizze_8664,
                                                        res_8402,
                                                        nesting_sizze_8664 *
                                                        res_8402,
                                                        nesting_sizze_8664 *
                                                        res_8402);
    
    assert(call_ret_9185 == 0);
    if (memblock_unref_device(ctx, &mem_9055, "mem_9055") != 0)
        return 1;
    
    struct memblock_device mem_9068;
    
    mem_9068.references = NULL;
    if (memblock_alloc_device(ctx, &mem_9068, bytes_9052, "mem_9068"))
        return 1;
    
    int call_ret_9186 = futrts_map_transpose_opencl_i32(ctx, mem_9068, 0,
                                                        mem_9059, 0, 1,
                                                        nesting_sizze_8664,
                                                        res_8402,
                                                        nesting_sizze_8664 *
                                                        res_8402,
                                                        nesting_sizze_8664 *
                                                        res_8402);
    
    assert(call_ret_9186 == 0);
    if (memblock_unref_device(ctx, &mem_9059, "mem_9059") != 0)
        return 1;
    out_arrsizze_9079 = nesting_sizze_8664;
    out_arrsizze_9080 = res_8402;
    out_arrsizze_9083 = nesting_sizze_8664;
    out_arrsizze_9084 = res_8402;
    out_memsizze_9078 = bytes_9052;
    if (memblock_set_device(ctx, &out_mem_9077, &mem_9063, "mem_9063") != 0)
        return 1;
    out_memsizze_9082 = bytes_9052;
    if (memblock_set_device(ctx, &out_mem_9081, &mem_9068, "mem_9068") != 0)
        return 1;
    *out_out_memsizze_9152 = out_memsizze_9078;
    (*out_mem_p_9153).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_9153, &out_mem_9077,
                            "out_mem_9077") != 0)
        return 1;
    *out_out_arrsizze_9154 = out_arrsizze_9079;
    *out_out_arrsizze_9155 = out_arrsizze_9080;
    *out_out_memsizze_9156 = out_memsizze_9082;
    (*out_mem_p_9157).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_9157, &out_mem_9081,
                            "out_mem_9081") != 0)
        return 1;
    *out_out_arrsizze_9158 = out_arrsizze_9083;
    *out_out_arrsizze_9159 = out_arrsizze_9084;
    if (memblock_unref_device(ctx, &mem_9068, "mem_9068") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9063, "mem_9063") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9033, "mem_9033") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9030, "mem_9030") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9059, "mem_9059") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9055, "mem_9055") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9027, "mem_9027") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9024, "mem_9024") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9022, "mem_9022") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9019, "mem_9019") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9016, "mem_9016") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9014, "mem_9014") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9010, "mem_9010") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9006, "mem_9006") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_9003, "mem_9003") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_9081, "out_mem_9081") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_9077, "out_mem_9077") != 0)
        return 1;
    return 0;
}
struct futhark_i32_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx,
                                          int32_t *data, int dim0, int dim1)
{
    struct futhark_i32_2d *arr = malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(int32_t),
                              "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (dim0 * dim1 * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * dim1 *
                                                      sizeof(int32_t), data + 0,
                                                      0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0,
                                              int dim1)
{
    struct futhark_i32_2d *arr = malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(int32_t),
                              "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (dim0 * dim1 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * dim1 *
                                                     sizeof(int32_t), 0, NULL,
                                                     NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_2d(struct futhark_context *ctx, struct futhark_i32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * arr->shape[1] * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     arr->shape[1] *
                                                     sizeof(int32_t), data + 0,
                                                     0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                 struct futhark_i32_2d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                              struct futhark_i32_2d *arr)
{
    return arr->shape;
}
struct futhark_f32_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1)
{
    struct futhark_f32_2d *arr = malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(float),
                              "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (dim0 * dim1 * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * dim1 *
                                                      sizeof(float), data + 0,
                                                      0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0,
                                              int dim1)
{
    struct futhark_f32_2d *arr = malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(float),
                              "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (dim0 * dim1 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * dim1 *
                                                     sizeof(float), 0, NULL,
                                                     NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * arr->shape[1] * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     arr->shape[1] *
                                                     sizeof(float), data + 0, 0,
                                                     NULL, NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                 struct futhark_f32_2d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr)
{
    return arr->shape;
}
struct futhark_f32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int dim0)
{
    struct futhark_f32_1d *arr = malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * sizeof(float),
                                                      data + 0, 0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0)
{
    struct futhark_f32_1d *arr = malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return 1;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * sizeof(float), 0,
                                                     NULL, NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     sizeof(float), data + 0, 0,
                                                     NULL, NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                 struct futhark_f32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0,
                       struct futhark_i32_2d **out1, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const int32_t in2, const
                       struct futhark_f32_1d *in3, const
                       struct futhark_f32_1d *in4, const int32_t in5)
{
    int64_t matRow_mem_sizze_8993;
    struct memblock_device matRow_mem_8994;
    
    matRow_mem_8994.references = NULL;
    
    int64_t vect_mem_sizze_8995;
    struct memblock_device vect_mem_8996;
    
    vect_mem_8996.references = NULL;
    
    int64_t angles_mem_sizze_8997;
    struct memblock_device angles_mem_8998;
    
    angles_mem_8998.references = NULL;
    
    int64_t rays_mem_sizze_8999;
    struct memblock_device rays_mem_9000;
    
    rays_mem_9000.references = NULL;
    
    int32_t sizze_8378;
    int32_t sizze_8379;
    int32_t sizze_8380;
    int32_t sizze_8381;
    int32_t num_cols_8384;
    int32_t gridsizze_8387;
    int64_t out_memsizze_9078;
    struct memblock_device out_mem_9077;
    
    out_mem_9077.references = NULL;
    
    int32_t out_arrsizze_9079;
    int32_t out_arrsizze_9080;
    int64_t out_memsizze_9082;
    struct memblock_device out_mem_9081;
    
    out_mem_9081.references = NULL;
    
    int32_t out_arrsizze_9083;
    int32_t out_arrsizze_9084;
    
    lock_lock(&ctx->lock);
    matRow_mem_8994 = in0->mem;
    matRow_mem_sizze_8993 = in0->mem.size;
    sizze_8378 = in0->shape[0];
    vect_mem_8996 = in1->mem;
    vect_mem_sizze_8995 = in1->mem.size;
    sizze_8379 = in1->shape[0];
    num_cols_8384 = in2;
    angles_mem_8998 = in3->mem;
    angles_mem_sizze_8997 = in3->mem.size;
    sizze_8380 = in3->shape[0];
    rays_mem_9000 = in4->mem;
    rays_mem_sizze_8999 = in4->mem.size;
    sizze_8381 = in4->shape[0];
    gridsizze_8387 = in5;
    
    int ret = futrts_main(ctx, &out_memsizze_9078, &out_mem_9077,
                          &out_arrsizze_9079, &out_arrsizze_9080,
                          &out_memsizze_9082, &out_mem_9081, &out_arrsizze_9083,
                          &out_arrsizze_9084, matRow_mem_sizze_8993,
                          matRow_mem_8994, vect_mem_sizze_8995, vect_mem_8996,
                          angles_mem_sizze_8997, angles_mem_8998,
                          rays_mem_sizze_8999, rays_mem_9000, sizze_8378,
                          sizze_8379, sizze_8380, sizze_8381, num_cols_8384,
                          gridsizze_8387);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_9077;
        (*out0)->shape[0] = out_arrsizze_9079;
        (*out0)->shape[1] = out_arrsizze_9080;
        assert((*out1 = malloc(sizeof(struct futhark_i32_2d))) != NULL);
        (*out1)->mem = out_mem_9081;
        (*out1)->shape[0] = out_arrsizze_9083;
        (*out1)->shape[1] = out_arrsizze_9084;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
