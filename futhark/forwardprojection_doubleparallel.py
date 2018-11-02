import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, value) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and self.device.type == device_type:
               if type(value) == str:
                   sizes[size] = self.device.get_info(getattr(cl.device_info,value))
               else:
                   sizes[size] = value
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             transpose_block_dim=16,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0
    self.free_list = {}

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}'.format(k))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            raise Exception('Unknown size class for size \'{}\': {}'.format(k, v['class']))
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DFUT_BLOCK_DIM={}".format(transpose_block_dim),
             "-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s,v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()
import pyopencl.array
import time
import argparse
synchronous = False
preferred_platform = None
preferred_device = None
fut_opencl_src = """#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))
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
#define group_sizze_10418 (group_size_10417)
#define group_sizze_10258 (group_size_10257)
#define y_10261 (group_size_10257 - 1)
#define group_sizze_10900 (group_size_10899)
#define max_num_groups_10902 (max_num_groups_10901)
#define group_sizze_10515 (group_size_10514)
__kernel void chunked_reduce_kernel_10916(__local volatile
                                          int64_t *mem_aligned_0,
                                          int32_t nesting_sizze_10105, __global
                                          unsigned char *mem_10788,
                                          int32_t num_threads_10908,
                                          int32_t per_thread_elements_10911,
                                          __global unsigned char *mem_10961)
{
    __local volatile char *restrict mem_10958 = mem_aligned_0;
    int32_t wave_sizze_11037;
    int32_t group_sizze_11038;
    bool thread_active_11039;
    int32_t gtid_10159;
    int32_t global_tid_10916;
    int32_t local_tid_10917;
    int32_t group_id_10918;
    
    global_tid_10916 = get_global_id(0);
    local_tid_10917 = get_local_id(0);
    group_sizze_11038 = get_local_size(0);
    wave_sizze_11037 = LOCKSTEP_WIDTH;
    group_id_10918 = get_group_id(0);
    gtid_10159 = global_tid_10916;
    thread_active_11039 = slt32(gtid_10159, nesting_sizze_10105);
    
    int32_t chunk_sizze_10923 = smin32(per_thread_elements_10911,
                                       squot32(nesting_sizze_10105 -
                                               global_tid_10916 +
                                               num_threads_10908 - 1,
                                               num_threads_10908));
    int32_t binop_x_10930;
    int32_t new_index_10931;
    int32_t last_offset_10932;
    int64_t binop_x_10933;
    int64_t bytes_10934;
    
    if (thread_active_11039) {
        binop_x_10930 = 4 * gtid_10159;
        new_index_10931 = 3 + binop_x_10930;
        last_offset_10932 = *(__global int32_t *) &mem_10788[new_index_10931 *
                                                             4];
        binop_x_10933 = sext_i32_i64(last_offset_10932);
        bytes_10934 = 4 * binop_x_10933;
    }
    
    int64_t max_per_thread_10925;
    int64_t final_result_10939;
    int64_t acc_10928 = 0;
    int32_t groupstream_mapaccum_dummy_chunk_sizze_10926 = 1;
    
    if (thread_active_11039) {
        for (int32_t i_10927 = 0; i_10927 < chunk_sizze_10923; i_10927++) {
            int64_t zz_10936 = smax64(acc_10928, bytes_10934);
            int64_t acc_tmp_11040 = zz_10936;
            
            acc_10928 = acc_tmp_11040;
        }
    }
    max_per_thread_10925 = acc_10928;
    for (int32_t comb_iter_11041 = 0; comb_iter_11041 <
         squot32(group_sizze_10900 + group_sizze_10900 - 1, group_sizze_10900);
         comb_iter_11041++) {
        int32_t combine_id_10921;
        int32_t flat_comb_id_11042 = comb_iter_11041 * group_sizze_10900 +
                local_tid_10917;
        
        combine_id_10921 = flat_comb_id_11042;
        if (slt32(combine_id_10921, group_sizze_10900) && 1) {
            *(__local int64_t *) &mem_10958[combine_id_10921 * 8] =
                max_per_thread_10925;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_11044;
    int32_t skip_waves_11043;
    int32_t my_index_10940;
    int32_t other_index_10941;
    int64_t x_10942;
    int64_t y_10943;
    
    my_index_10940 = local_tid_10917;
    offset_11044 = 0;
    other_index_10941 = local_tid_10917 + offset_11044;
    if (slt32(local_tid_10917, group_sizze_10900)) {
        x_10942 = *(__local int64_t *) &mem_10958[(local_tid_10917 +
                                                   offset_11044) * 8];
    }
    offset_11044 = 1;
    other_index_10941 = local_tid_10917 + offset_11044;
    while (slt32(offset_11044, wave_sizze_11037)) {
        if (slt32(other_index_10941, group_sizze_10900) && ((local_tid_10917 -
                                                             squot32(local_tid_10917,
                                                                     wave_sizze_11037) *
                                                             wave_sizze_11037) &
                                                            (2 * offset_11044 -
                                                             1)) == 0) {
            // read array element
            {
                y_10943 = *(volatile __local
                            int64_t *) &mem_10958[(local_tid_10917 +
                                                   offset_11044) * 8];
            }
            
            int64_t zz_10944;
            
            if (thread_active_11039) {
                zz_10944 = smax64(x_10942, y_10943);
            }
            x_10942 = zz_10944;
            *(volatile __local int64_t *) &mem_10958[local_tid_10917 * 8] =
                x_10942;
        }
        offset_11044 *= 2;
        other_index_10941 = local_tid_10917 + offset_11044;
    }
    skip_waves_11043 = 1;
    while (slt32(skip_waves_11043, squot32(group_sizze_10900 +
                                           wave_sizze_11037 - 1,
                                           wave_sizze_11037))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_11044 = skip_waves_11043 * wave_sizze_11037;
        other_index_10941 = local_tid_10917 + offset_11044;
        if (slt32(other_index_10941, group_sizze_10900) && ((local_tid_10917 -
                                                             squot32(local_tid_10917,
                                                                     wave_sizze_11037) *
                                                             wave_sizze_11037) ==
                                                            0 &&
                                                            (squot32(local_tid_10917,
                                                                     wave_sizze_11037) &
                                                             (2 *
                                                              skip_waves_11043 -
                                                              1)) == 0)) {
            // read array element
            {
                y_10943 = *(__local int64_t *) &mem_10958[(local_tid_10917 +
                                                           offset_11044) * 8];
            }
            
            int64_t zz_10944;
            
            if (thread_active_11039) {
                zz_10944 = smax64(x_10942, y_10943);
            }
            x_10942 = zz_10944;
            *(__local int64_t *) &mem_10958[local_tid_10917 * 8] = x_10942;
        }
        skip_waves_11043 *= 2;
    }
    final_result_10939 = x_10942;
    if (local_tid_10917 == 0) {
        *(__global int64_t *) &mem_10961[group_id_10918 * 8] =
            final_result_10939;
    }
}
__kernel void fut_kernel_map_transpose_f32(__global float *odata,
                                           uint odata_offset, __global
                                           float *idata, uint idata_offset,
                                           uint width, uint height,
                                           uint input_size, uint output_size,
                                           __local float *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(float);
    idata += idata_offset / sizeof(float);
    // Adjust the input and output arrays for the third dimension.
    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory
    x_index = get_global_id(0);
    y_index = get_global_id(1);
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < input_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Scatter the transposed matrix tile to global memory.
    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);
    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < output_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void fut_kernel_map_transpose_i32(__global int32_t *odata,
                                           uint odata_offset, __global
                                           int32_t *idata, uint idata_offset,
                                           uint width, uint height,
                                           uint input_size, uint output_size,
                                           __local int32_t *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(int32_t);
    idata += idata_offset / sizeof(int32_t);
    // Adjust the input and output arrays for the third dimension.
    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory
    x_index = get_global_id(0);
    y_index = get_global_id(1);
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < input_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Scatter the transposed matrix tile to global memory.
    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);
    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < output_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void fut_kernel_map_transpose_lowheight_f32(__global float *odata,
                                                     uint odata_offset, __global
                                                     float *idata,
                                                     uint idata_offset,
                                                     uint width, uint height,
                                                     uint input_size,
                                                     uint output_size,
                                                     uint mulx, __local
                                                     float *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(float);
    idata += idata_offset / sizeof(float);
    // Adjust the input and output arrays for the third dimension.
    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory
    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +
        get_local_id(1) % mulx * FUT_BLOCK_DIM;
    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < input_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Scatter the transposed matrix tile to global memory.
    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;
    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +
        get_local_id(0) % mulx * FUT_BLOCK_DIM;
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < output_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void fut_kernel_map_transpose_lowheight_i32(__global int32_t *odata,
                                                     uint odata_offset, __global
                                                     int32_t *idata,
                                                     uint idata_offset,
                                                     uint width, uint height,
                                                     uint input_size,
                                                     uint output_size,
                                                     uint mulx, __local
                                                     int32_t *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(int32_t);
    idata += idata_offset / sizeof(int32_t);
    // Adjust the input and output arrays for the third dimension.
    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory
    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +
        get_local_id(1) % mulx * FUT_BLOCK_DIM;
    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < input_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Scatter the transposed matrix tile to global memory.
    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;
    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +
        get_local_id(0) % mulx * FUT_BLOCK_DIM;
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < output_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void fut_kernel_map_transpose_lowwidth_f32(__global float *odata,
                                                    uint odata_offset, __global
                                                    float *idata,
                                                    uint idata_offset,
                                                    uint width, uint height,
                                                    uint input_size,
                                                    uint output_size, uint muly,
                                                    __local float *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(float);
    idata += idata_offset / sizeof(float);
    // Adjust the input and output arrays for the third dimension.
    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory
    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;
    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +
        get_local_id(0) % muly * FUT_BLOCK_DIM;
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < input_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Scatter the transposed matrix tile to global memory.
    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +
        get_local_id(1) % muly * FUT_BLOCK_DIM;
    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < output_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void fut_kernel_map_transpose_lowwidth_i32(__global int32_t *odata,
                                                    uint odata_offset, __global
                                                    int32_t *idata,
                                                    uint idata_offset,
                                                    uint width, uint height,
                                                    uint input_size,
                                                    uint output_size, uint muly,
                                                    __local int32_t *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(int32_t);
    idata += idata_offset / sizeof(int32_t);
    // Adjust the input and output arrays for the third dimension.
    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory
    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;
    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +
        get_local_id(0) % muly * FUT_BLOCK_DIM;
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < input_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Scatter the transposed matrix tile to global memory.
    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +
        get_local_id(1) % muly * FUT_BLOCK_DIM;
    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < output_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void fut_kernel_map_transpose_small_f32(__global float *odata,
                                                 uint odata_offset, __global
                                                 float *idata,
                                                 uint idata_offset,
                                                 uint num_arrays, uint width,
                                                 uint height, uint input_size,
                                                 uint output_size)
{
    uint our_array_offset = get_global_id(0) / (height * width) * (height *
                                                                   width);
    uint x_index = get_global_id(0) % (height * width) / height;
    uint y_index = get_global_id(0) % height;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(float);
    idata += idata_offset / sizeof(float);
    // Adjust the input and output arrays.
    odata += our_array_offset;
    idata += our_array_offset;
    
    uint index_in = y_index * width + x_index;
    uint index_out = x_index * height + y_index;
    
    if (get_global_id(0) < input_size)
        odata[index_out] = idata[index_in];
}
__kernel void fut_kernel_map_transpose_small_i32(__global int32_t *odata,
                                                 uint odata_offset, __global
                                                 int32_t *idata,
                                                 uint idata_offset,
                                                 uint num_arrays, uint width,
                                                 uint height, uint input_size,
                                                 uint output_size)
{
    uint our_array_offset = get_global_id(0) / (height * width) * (height *
                                                                   width);
    uint x_index = get_global_id(0) % (height * width) / height;
    uint y_index = get_global_id(0) % height;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(int32_t);
    idata += idata_offset / sizeof(int32_t);
    // Adjust the input and output arrays.
    odata += our_array_offset;
    idata += our_array_offset;
    
    uint index_in = y_index * width + x_index;
    uint index_out = x_index * height + y_index;
    
    if (get_global_id(0) < input_size)
        odata[index_out] = idata[index_in];
}
__kernel void kernel_replicate_9913(__global unsigned char *mem_10815)
{
    const uint replicate_gtid_9913 = get_global_id(0);
    
    if (replicate_gtid_9913 >= 1)
        return;
    *(__global float *) &mem_10815[replicate_gtid_9913 * 4] = 0.0F;
}
__kernel void map_kernel_10096(int32_t sizze_9810, int32_t sizze_9811, __global
                               unsigned char *mem_10714, __global
                               unsigned char *mem_10722)
{
    int32_t wave_sizze_10986;
    int32_t group_sizze_10987;
    bool thread_active_10988;
    int32_t gtid_10087;
    int32_t gtid_10088;
    int32_t global_tid_10096;
    int32_t local_tid_10097;
    int32_t group_id_10098;
    
    global_tid_10096 = get_global_id(0);
    local_tid_10097 = get_local_id(0);
    group_sizze_10987 = get_local_size(0);
    wave_sizze_10986 = LOCKSTEP_WIDTH;
    group_id_10098 = get_group_id(0);
    gtid_10087 = squot32(global_tid_10096, sizze_9811);
    gtid_10088 = global_tid_10096 - squot32(global_tid_10096, sizze_9811) *
        sizze_9811;
    thread_active_10988 = slt32(gtid_10087, sizze_9810) && slt32(gtid_10088,
                                                                 sizze_9811);
    
    float res_10099;
    
    if (thread_active_10988) {
        res_10099 = *(__global float *) &mem_10714[gtid_10087 * 4];
    }
    if (thread_active_10988) {
        *(__global float *) &mem_10722[(gtid_10087 * sizze_9811 + gtid_10088) *
                                       4] = res_10099;
    }
}
__kernel void map_kernel_10112(int32_t sizze_9810, int32_t sizze_9811, __global
                               unsigned char *mem_10711, __global
                               unsigned char *mem_10718)
{
    int32_t wave_sizze_10983;
    int32_t group_sizze_10984;
    bool thread_active_10985;
    int32_t gtid_10103;
    int32_t gtid_10104;
    int32_t global_tid_10112;
    int32_t local_tid_10113;
    int32_t group_id_10114;
    
    global_tid_10112 = get_global_id(0);
    local_tid_10113 = get_local_id(0);
    group_sizze_10984 = get_local_size(0);
    wave_sizze_10983 = LOCKSTEP_WIDTH;
    group_id_10114 = get_group_id(0);
    gtid_10103 = squot32(global_tid_10112, sizze_9811);
    gtid_10104 = global_tid_10112 - squot32(global_tid_10112, sizze_9811) *
        sizze_9811;
    thread_active_10985 = slt32(gtid_10103, sizze_9810) && slt32(gtid_10104,
                                                                 sizze_9811);
    
    float res_10115;
    
    if (thread_active_10985) {
        res_10115 = *(__global float *) &mem_10711[gtid_10103 * 4];
    }
    if (thread_active_10985) {
        *(__global float *) &mem_10718[(gtid_10103 * sizze_9811 + gtid_10104) *
                                       4] = res_10115;
    }
}
__kernel void map_kernel_10124(int32_t sizze_9810, __global
                               unsigned char *angles_mem_10704, __global
                               unsigned char *mem_10711, __global
                               unsigned char *mem_10714)
{
    int32_t wave_sizze_10980;
    int32_t group_sizze_10981;
    bool thread_active_10982;
    int32_t gtid_10117;
    int32_t global_tid_10124;
    int32_t local_tid_10125;
    int32_t group_id_10126;
    
    global_tid_10124 = get_global_id(0);
    local_tid_10125 = get_local_id(0);
    group_sizze_10981 = get_local_size(0);
    wave_sizze_10980 = LOCKSTEP_WIDTH;
    group_id_10126 = get_group_id(0);
    gtid_10117 = global_tid_10124;
    thread_active_10982 = slt32(gtid_10117, sizze_9810);
    
    float x_10127;
    float res_10128;
    float res_10129;
    
    if (thread_active_10982) {
        x_10127 = *(__global float *) &angles_mem_10704[gtid_10117 * 4];
        res_10128 = futrts_sin32(x_10127);
        res_10129 = futrts_cos32(x_10127);
    }
    if (thread_active_10982) {
        *(__global float *) &mem_10711[gtid_10117 * 4] = res_10128;
    }
    if (thread_active_10982) {
        *(__global float *) &mem_10714[gtid_10117 * 4] = res_10129;
    }
}
__kernel void map_kernel_10166(int32_t sizze_9811, float res_9821,
                               float res_9834, int32_t nesting_sizze_10105,
                               int32_t num_threads_10165, __global
                               unsigned char *rays_mem_10706, __global
                               unsigned char *mem_10718, __global
                               unsigned char *mem_10732, __global
                               unsigned char *mem_10736, __global
                               unsigned char *mem_10740, __global
                               unsigned char *mem_10751, __global
                               unsigned char *mem_10788, __global
                               unsigned char *mem_10791, __global
                               unsigned char *mem_10794, __global
                               unsigned char *mem_10803, __global
                               unsigned char *mem_10806, __global
                               unsigned char *mem_10809, __global
                               unsigned char *mem_10812)
{
    int32_t wave_sizze_11054;
    int32_t group_sizze_11055;
    bool thread_active_11056;
    int32_t gtid_10159;
    int32_t global_tid_10166;
    int32_t local_tid_10167;
    int32_t group_id_10168;
    
    global_tid_10166 = get_global_id(0);
    local_tid_10167 = get_local_id(0);
    group_sizze_11055 = get_local_size(0);
    wave_sizze_11054 = LOCKSTEP_WIDTH;
    group_id_10168 = get_group_id(0);
    gtid_10159 = global_tid_10166;
    thread_active_11056 = slt32(gtid_10159, nesting_sizze_10105);
    
    int32_t new_index_10652;
    int32_t binop_y_10654;
    int32_t new_index_10655;
    float x_10169;
    float x_10170;
    bool cond_10171;
    int32_t binop_x_10680;
    int32_t new_index_10681;
    int32_t last_offset_10177;
    float res_10210;
    float res_10211;
    float res_10212;
    float res_10213;
    
    if (thread_active_11056) {
        new_index_10652 = squot32(gtid_10159, sizze_9811);
        binop_y_10654 = sizze_9811 * new_index_10652;
        new_index_10655 = gtid_10159 - binop_y_10654;
        x_10169 = *(__global float *) &mem_10718[(new_index_10652 * sizze_9811 +
                                                  new_index_10655) * 4];
        x_10170 = *(__global float *) &rays_mem_10706[new_index_10655 * 4];
        cond_10171 = *(__global bool *) &mem_10732[gtid_10159];
        binop_x_10680 = 4 * gtid_10159;
        new_index_10681 = 3 + binop_x_10680;
        last_offset_10177 = *(__global int32_t *) &mem_10788[new_index_10681 *
                                                             4];
        for (int32_t write_iter_10184 = 0; write_iter_10184 < 4;
             write_iter_10184++) {
            int32_t new_index_10683 = write_iter_10184 + binop_x_10680;
            int32_t write_iv_10185 = *(__global
                                       int32_t *) &mem_10751[new_index_10683 *
                                                             4];
            int32_t write_iv_10186 = *(__global
                                       int32_t *) &mem_10788[new_index_10683 *
                                                             4];
            float write_iv_10187 = *(__global
                                     float *) &mem_10736[(write_iter_10184 *
                                                          nesting_sizze_10105 +
                                                          gtid_10159) * 4];
            float write_iv_10188 = *(__global
                                     float *) &mem_10740[(write_iter_10184 *
                                                          nesting_sizze_10105 +
                                                          gtid_10159) * 4];
            bool is_this_one_10195 = write_iv_10185 == 0;
            int32_t this_offset_10196 = -1 + write_iv_10186;
            int32_t total_res_10197;
            
            if (is_this_one_10195) {
                total_res_10197 = this_offset_10196;
            } else {
                total_res_10197 = -1;
            }
            
            bool less_than_zzero_10198 = slt32(total_res_10197, 0);
            bool greater_than_sizze_10199 = sle32(last_offset_10177,
                                                  total_res_10197);
            bool outside_bounds_dim_10200 = less_than_zzero_10198 ||
                 greater_than_sizze_10199;
            
            if (!outside_bounds_dim_10200) {
                *(__global float *) &mem_10791[(total_res_10197 *
                                                num_threads_10165 +
                                                global_tid_10166) * 4] =
                    write_iv_10187;
            }
            if (!outside_bounds_dim_10200) {
                *(__global float *) &mem_10794[(total_res_10197 *
                                                num_threads_10165 +
                                                global_tid_10166) * 4] =
                    write_iv_10188;
            }
        }
        if (cond_10171) {
            res_10210 = x_10170;
            res_10211 = res_9834;
            res_10212 = x_10170;
            res_10213 = res_9821;
        } else {
            bool cond_10214;
            float res_10215;
            float res_10216;
            float res_10217;
            float res_10218;
            
            cond_10214 = x_10169 == 1.0F;
            if (cond_10214) {
                res_10215 = res_9834;
                res_10216 = x_10170;
                res_10217 = res_9821;
                res_10218 = x_10170;
            } else {
                float x_10219;
                float y_10220;
                bool cond_10221;
                float res_10222;
                float res_10223;
                float res_10224;
                float res_10225;
                
                x_10219 = *(__global float *) &mem_10791[global_tid_10166 * 4];
                y_10220 = *(__global float *) &mem_10791[(num_threads_10165 +
                                                          global_tid_10166) *
                                                         4];
                cond_10221 = x_10219 < y_10220;
                if (cond_10221) {
                    res_10222 = x_10219;
                } else {
                    res_10222 = y_10220;
                }
                if (cond_10221) {
                    res_10223 = y_10220;
                } else {
                    res_10223 = x_10219;
                }
                if (cond_10221) {
                    float res_10226;
                    float res_10227;
                    
                    res_10226 = *(__global
                                  float *) &mem_10794[global_tid_10166 * 4];
                    res_10227 = *(__global
                                  float *) &mem_10794[(num_threads_10165 +
                                                       global_tid_10166) * 4];
                    res_10224 = res_10226;
                    res_10225 = res_10227;
                } else {
                    float res_10228;
                    float res_10229;
                    
                    res_10228 = *(__global
                                  float *) &mem_10794[(num_threads_10165 +
                                                       global_tid_10166) * 4];
                    res_10229 = *(__global
                                  float *) &mem_10794[global_tid_10166 * 4];
                    res_10224 = res_10228;
                    res_10225 = res_10229;
                }
                res_10215 = res_10222;
                res_10216 = res_10224;
                res_10217 = res_10223;
                res_10218 = res_10225;
            }
            res_10210 = res_10215;
            res_10211 = res_10216;
            res_10212 = res_10217;
            res_10213 = res_10218;
        }
    }
    if (thread_active_11056) {
        *(__global float *) &mem_10803[gtid_10159 * 4] = res_10210;
    }
    if (thread_active_11056) {
        *(__global float *) &mem_10806[gtid_10159 * 4] = res_10211;
    }
    if (thread_active_11056) {
        *(__global float *) &mem_10809[gtid_10159 * 4] = res_10212;
    }
    if (thread_active_11056) {
        *(__global float *) &mem_10812[gtid_10159 * 4] = res_10213;
    }
}
__kernel void map_kernel_10413(int32_t y_10321, int32_t convop_x_10734, __global
                               unsigned char *mem_10745, __global
                               unsigned char *mem_10748, __global
                               unsigned char *mem_10780, __global
                               unsigned char *mem_10783, __global
                               unsigned char *mem_10785, __global
                               unsigned char *mem_10788)
{
    int32_t wave_sizze_11034;
    int32_t group_sizze_11035;
    bool thread_active_11036;
    int32_t j_10396;
    int32_t global_tid_10413;
    int32_t local_tid_10414;
    int32_t group_id_10415;
    
    global_tid_10413 = get_global_id(0);
    local_tid_10414 = get_local_id(0);
    group_sizze_11035 = get_local_size(0);
    wave_sizze_11034 = LOCKSTEP_WIDTH;
    group_id_10415 = get_group_id(0);
    j_10396 = global_tid_10413;
    thread_active_11036 = slt32(j_10396, convop_x_10734);
    
    bool y_flag_10389;
    int32_t y_10390;
    int32_t group_id_10401;
    bool cond_10402;
    bool final_result_10405;
    int32_t final_result_10406;
    
    if (thread_active_11036) {
        y_flag_10389 = *(__global bool *) &mem_10745[j_10396];
        y_10390 = *(__global int32_t *) &mem_10748[j_10396 * 4];
        group_id_10401 = squot32(j_10396, y_10321);
        cond_10402 = 0 == group_id_10401;
        if (cond_10402) {
            final_result_10405 = y_flag_10389;
            final_result_10406 = y_10390;
        } else {
            int32_t carry_in_index_10403;
            bool x_flag_10387;
            int32_t x_10388;
            bool new_flag_10391;
            int32_t seg_lhs_10392;
            int32_t zz_10395;
            
            carry_in_index_10403 = group_id_10401 - 1;
            x_flag_10387 = *(__global bool *) &mem_10780[carry_in_index_10403];
            x_10388 = *(__global int32_t *) &mem_10783[carry_in_index_10403 *
                                                       4];
            new_flag_10391 = x_flag_10387 || y_flag_10389;
            if (y_flag_10389) {
                seg_lhs_10392 = 0;
            } else {
                seg_lhs_10392 = x_10388;
            }
            zz_10395 = y_10390 + seg_lhs_10392;
            final_result_10405 = new_flag_10391;
            final_result_10406 = zz_10395;
        }
    }
    if (thread_active_11036) {
        *(__global bool *) &mem_10785[j_10396] = final_result_10405;
    }
    if (thread_active_11036) {
        *(__global int32_t *) &mem_10788[j_10396 * 4] = final_result_10406;
    }
}
__kernel void map_kernel_10423(int32_t sizze_9811, float res_9821,
                               float res_9834, int32_t nesting_sizze_10105,
                               __global unsigned char *rays_mem_10706, __global
                               unsigned char *mem_10718, __global
                               unsigned char *mem_10722, __global
                               unsigned char *mem_10725, __global
                               unsigned char *mem_10728, __global
                               unsigned char *mem_10730, __global
                               unsigned char *mem_10732, __global
                               unsigned char *mem_10736, __global
                               unsigned char *mem_10740, __global
                               unsigned char *mem_10743)
{
    int32_t wave_sizze_10989;
    int32_t group_sizze_10990;
    bool thread_active_10991;
    int32_t gtid_10416;
    int32_t global_tid_10423;
    int32_t local_tid_10424;
    int32_t group_id_10425;
    
    global_tid_10423 = get_global_id(0);
    local_tid_10424 = get_local_id(0);
    group_sizze_10990 = get_local_size(0);
    wave_sizze_10989 = LOCKSTEP_WIDTH;
    group_id_10425 = get_group_id(0);
    gtid_10416 = global_tid_10423;
    thread_active_10991 = slt32(gtid_10416, nesting_sizze_10105);
    
    int32_t new_index_10634;
    int32_t binop_y_10636;
    int32_t new_index_10637;
    float x_10426;
    float x_10427;
    float x_10428;
    bool cond_10429;
    float res_10430;
    bool cond_10434;
    float res_10435;
    float res_10439;
    float res_10443;
    float res_10449;
    bool arr_elem_10450;
    float res_10451;
    bool arr_elem_10452;
    float res_10453;
    bool arr_elem_10454;
    float res_10455;
    bool arr_elem_10456;
    
    if (thread_active_10991) {
        new_index_10634 = squot32(gtid_10416, sizze_9811);
        binop_y_10636 = sizze_9811 * new_index_10634;
        new_index_10637 = gtid_10416 - binop_y_10636;
        x_10426 = *(__global float *) &mem_10718[(new_index_10634 * sizze_9811 +
                                                  new_index_10637) * 4];
        x_10427 = *(__global float *) &mem_10722[(new_index_10634 * sizze_9811 +
                                                  new_index_10637) * 4];
        x_10428 = *(__global float *) &rays_mem_10706[new_index_10637 * 4];
        cond_10429 = x_10426 == 0.0F;
        if (cond_10429) {
            res_10430 = x_10428;
        } else {
            float y_10431;
            float x_10432;
            float res_10433;
            
            y_10431 = res_9834 * x_10427;
            x_10432 = x_10428 - y_10431;
            res_10433 = x_10432 / x_10426;
            res_10430 = res_10433;
        }
        cond_10434 = x_10427 == 0.0F;
        if (cond_10434) {
            res_10435 = x_10428;
        } else {
            float y_10436;
            float x_10437;
            float res_10438;
            
            y_10436 = res_9834 * x_10426;
            x_10437 = x_10428 - y_10436;
            res_10438 = x_10437 / x_10427;
            res_10435 = res_10438;
        }
        if (cond_10434) {
            res_10439 = x_10428;
        } else {
            float y_10440;
            float x_10441;
            float res_10442;
            
            y_10440 = res_9821 * x_10426;
            x_10441 = x_10428 - y_10440;
            res_10442 = x_10441 / x_10427;
            res_10439 = res_10442;
        }
        if (cond_10429) {
            res_10443 = x_10428;
        } else {
            float y_10444;
            float x_10445;
            float res_10446;
            
            y_10444 = res_9821 * x_10427;
            x_10445 = x_10428 - y_10444;
            res_10446 = x_10445 / x_10426;
            res_10443 = res_10446;
        }
        *(__global float *) &mem_10725[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        local_tid_10424) * 4] = res_9834;
        *(__global float *) &mem_10725[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        group_sizze_10418 + local_tid_10424) *
                                       4] = res_10435;
        *(__global float *) &mem_10725[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        2 * group_sizze_10418 +
                                        local_tid_10424) * 4] = res_10439;
        *(__global float *) &mem_10725[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        3 * group_sizze_10418 +
                                        local_tid_10424) * 4] = res_9821;
        *(__global float *) &mem_10728[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        local_tid_10424) * 4] = res_10430;
        *(__global float *) &mem_10728[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        group_sizze_10418 + local_tid_10424) *
                                       4] = res_9834;
        *(__global float *) &mem_10728[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        2 * group_sizze_10418 +
                                        local_tid_10424) * 4] = res_9821;
        *(__global float *) &mem_10728[(group_id_10425 * (4 *
                                                          group_sizze_10418) +
                                        3 * group_sizze_10418 +
                                        local_tid_10424) * 4] = res_10443;
        res_10449 = (float) fabs(res_10430);
        arr_elem_10450 = res_10449 <= res_9821;
        res_10451 = (float) fabs(res_10435);
        arr_elem_10452 = res_10451 <= res_9821;
        res_10453 = (float) fabs(res_10439);
        arr_elem_10454 = res_10453 <= res_9821;
        res_10455 = (float) fabs(res_10443);
        arr_elem_10456 = res_10455 <= res_9821;
        *(__global bool *) &mem_10730[group_id_10425 * (4 * group_sizze_10418) +
                                      local_tid_10424] = arr_elem_10450;
        *(__global bool *) &mem_10730[group_id_10425 * (4 * group_sizze_10418) +
                                      group_sizze_10418 + local_tid_10424] =
            arr_elem_10452;
        *(__global bool *) &mem_10730[group_id_10425 * (4 * group_sizze_10418) +
                                      2 * group_sizze_10418 + local_tid_10424] =
            arr_elem_10454;
        *(__global bool *) &mem_10730[group_id_10425 * (4 * group_sizze_10418) +
                                      3 * group_sizze_10418 + local_tid_10424] =
            arr_elem_10456;
    }
    if (thread_active_10991) {
        *(__global bool *) &mem_10732[gtid_10416] = cond_10429;
    }
    if (thread_active_10991) {
        for (int32_t i_10992 = 0; i_10992 < 4; i_10992++) {
            *(__global float *) &mem_10736[(i_10992 * nesting_sizze_10105 +
                                            gtid_10416) * 4] = *(__global
                                                                 float *) &mem_10725[(group_id_10425 *
                                                                                      (4 *
                                                                                       group_sizze_10418) +
                                                                                      i_10992 *
                                                                                      group_sizze_10418 +
                                                                                      local_tid_10424) *
                                                                                     4];
        }
    }
    if (thread_active_10991) {
        for (int32_t i_10993 = 0; i_10993 < 4; i_10993++) {
            *(__global float *) &mem_10740[(i_10993 * nesting_sizze_10105 +
                                            gtid_10416) * 4] = *(__global
                                                                 float *) &mem_10728[(group_id_10425 *
                                                                                      (4 *
                                                                                       group_sizze_10418) +
                                                                                      i_10993 *
                                                                                      group_sizze_10418 +
                                                                                      local_tid_10424) *
                                                                                     4];
        }
    }
    if (thread_active_10991) {
        for (int32_t i_10994 = 0; i_10994 < 4; i_10994++) {
            *(__global bool *) &mem_10743[gtid_10416 * 4 + i_10994] = *(__global
                                                                        bool *) &mem_10730[group_id_10425 *
                                                                                           (4 *
                                                                                            group_sizze_10418) +
                                                                                           i_10994 *
                                                                                           group_sizze_10418 +
                                                                                           local_tid_10424];
        }
    }
}
__kernel void map_kernel_10478(int32_t flat_dim_9921, int32_t x_9940, __global
                               unsigned char *voxels_mem_10708, __global
                               unsigned char *mem_10855, __global
                               unsigned char *mem_10863, __global
                               unsigned char *mem_10866)
{
    int32_t wave_sizze_11077;
    int32_t group_sizze_11078;
    bool thread_active_11079;
    int32_t gtid_10471;
    int32_t global_tid_10478;
    int32_t local_tid_10479;
    int32_t group_id_10480;
    
    global_tid_10478 = get_global_id(0);
    local_tid_10479 = get_local_id(0);
    group_sizze_11078 = get_local_size(0);
    wave_sizze_11077 = LOCKSTEP_WIDTH;
    group_id_10480 = get_group_id(0);
    gtid_10471 = global_tid_10478;
    thread_active_11079 = slt32(gtid_10471, x_9940);
    
    float res_10483;
    
    if (thread_active_11079) {
        float x_10486 = 0.0F;
        
        for (int32_t chunk_offset_10485 = 0; chunk_offset_10485 < flat_dim_9921;
             chunk_offset_10485++) {
            float x_10495 = *(__global float *) &mem_10855[(chunk_offset_10485 *
                                                            x_9940 +
                                                            gtid_10471) * 4];
            int32_t x_10496 = *(__global
                                int32_t *) &mem_10863[(chunk_offset_10485 *
                                                       x_9940 + gtid_10471) *
                                                      4];
            bool cond_10498 = x_10496 == -1;
            float res_10499;
            
            if (cond_10498) {
                res_10499 = 0.0F;
            } else {
                float y_10500;
                float res_10501;
                
                y_10500 = *(__global float *) &voxels_mem_10708[x_10496 * 4];
                res_10501 = x_10495 * y_10500;
                res_10499 = res_10501;
            }
            
            float res_10503 = x_10486 + res_10499;
            float x_tmp_11080 = res_10503;
            
            x_10486 = x_tmp_11080;
        }
        res_10483 = x_10486;
    }
    if (thread_active_11079) {
        *(__global float *) &mem_10866[gtid_10471 * 4] = res_10483;
    }
}
__kernel void map_kernel_10520(int32_t res_9819, int32_t res_9820,
                               float res_9821, int32_t range_start_9915,
                               int32_t num_elems_9920, int32_t y_9923,
                               int32_t x_9940, __global
                               unsigned char *mem_10819, __global
                               unsigned char *mem_10822, __global
                               unsigned char *mem_10825, __global
                               unsigned char *mem_10828, __global
                               unsigned char *mem_10831, __global
                               unsigned char *mem_10834, __global
                               unsigned char *mem_10837, __global
                               unsigned char *mem_10842, __global
                               unsigned char *mem_10847)
{
    int32_t wave_sizze_11072;
    int32_t group_sizze_11073;
    bool thread_active_11074;
    int32_t gtid_10511;
    int32_t gtid_10512;
    int32_t global_tid_10520;
    int32_t local_tid_10521;
    int32_t group_id_10522;
    
    global_tid_10520 = get_global_id(0);
    local_tid_10521 = get_local_id(0);
    group_sizze_11073 = get_local_size(0);
    wave_sizze_11072 = LOCKSTEP_WIDTH;
    group_id_10522 = get_group_id(0);
    gtid_10511 = squot32(global_tid_10520, num_elems_9920);
    gtid_10512 = global_tid_10520 - squot32(global_tid_10520, num_elems_9920) *
        num_elems_9920;
    thread_active_11074 = slt32(gtid_10511, x_9940) && slt32(gtid_10512,
                                                             num_elems_9920);
    
    bool res_10523;
    float res_10524;
    float res_10525;
    float res_10526;
    float res_10527;
    int32_t index_primexp_10679;
    float res_10529;
    float y_10530;
    float x_10531;
    float x_10532;
    float res_10533;
    float x_10534;
    float y_10535;
    float x_10536;
    float x_10537;
    float res_10538;
    int32_t res_10539;
    float res_10540;
    bool res_10541;
    float res_10542;
    int32_t res_10549;
    int32_t res_10550;
    float res_10551;
    bool res_10552;
    float res_10553;
    int32_t res_10560;
    int32_t res_10561;
    float res_10562;
    float res_10563;
    float x_10564;
    float res_10565;
    float x_10566;
    float res_10567;
    float res_10568;
    float res_10569;
    int32_t res_10570;
    int32_t res_10571;
    int32_t res_10578;
    bool cond_10585;
    bool res_10586;
    bool x_10587;
    int32_t res_10588;
    float res_10589;
    bool cond_10592;
    bool res_10593;
    bool x_10594;
    float res_10595;
    int32_t res_10596;
    
    if (thread_active_11074) {
        res_10523 = *(__global bool *) &mem_10819[gtid_10511];
        res_10524 = *(__global float *) &mem_10822[gtid_10511 * 4];
        res_10525 = *(__global float *) &mem_10825[gtid_10511 * 4];
        res_10526 = *(__global float *) &mem_10828[gtid_10511 * 4];
        res_10527 = *(__global float *) &mem_10831[gtid_10511 * 4];
        index_primexp_10679 = range_start_9915 + gtid_10512;
        res_10529 = sitofp_i32_f32(index_primexp_10679);
        y_10530 = res_10529 - res_10524;
        x_10531 = res_10526 * y_10530;
        x_10532 = res_10525 + x_10531;
        res_10533 = res_9821 + x_10532;
        x_10534 = 1.0F + res_10529;
        y_10535 = x_10534 - res_10524;
        x_10536 = res_10526 * y_10535;
        x_10537 = res_10525 + x_10536;
        res_10538 = res_9821 + x_10537;
        res_10539 = fptosi_f32_i32(res_10533);
        res_10540 = sitofp_i32_f32(res_10539);
        res_10541 = 0.0F <= res_10533;
        if (res_10541) {
            bool res_10543;
            float res_10544;
            
            res_10543 = res_10540 < res_10533;
            if (res_10543) {
                res_10544 = res_10540;
            } else {
                res_10544 = res_10533;
            }
            res_10542 = res_10544;
        } else {
            bool res_10545;
            float res_10546;
            
            res_10545 = res_10533 < res_10540;
            if (res_10545) {
                int32_t res_10547;
                float res_10548;
                
                res_10547 = res_10539 - 1;
                res_10548 = sitofp_i32_f32(res_10547);
                res_10546 = res_10548;
            } else {
                res_10546 = res_10533;
            }
            res_10542 = res_10546;
        }
        res_10549 = fptosi_f32_i32(res_10542);
        res_10550 = fptosi_f32_i32(res_10538);
        res_10551 = sitofp_i32_f32(res_10550);
        res_10552 = 0.0F <= res_10538;
        if (res_10552) {
            bool res_10554;
            float res_10555;
            
            res_10554 = res_10551 < res_10538;
            if (res_10554) {
                res_10555 = res_10551;
            } else {
                res_10555 = res_10538;
            }
            res_10553 = res_10555;
        } else {
            bool res_10556;
            float res_10557;
            
            res_10556 = res_10538 < res_10551;
            if (res_10556) {
                int32_t res_10558;
                float res_10559;
                
                res_10558 = res_10550 - 1;
                res_10559 = sitofp_i32_f32(res_10558);
                res_10557 = res_10559;
            } else {
                res_10557 = res_10538;
            }
            res_10553 = res_10557;
        }
        res_10560 = fptosi_f32_i32(res_10553);
        res_10561 = smax32(res_10549, res_10560);
        res_10562 = res_10538 - res_10533;
        res_10563 = sitofp_i32_f32(res_10561);
        x_10564 = res_10563 - res_10533;
        res_10565 = x_10564 / res_10562;
        x_10566 = res_10538 - res_10563;
        res_10567 = x_10566 / res_10562;
        res_10568 = res_10527 * res_10565;
        res_10569 = res_10527 * res_10567;
        res_10570 = res_9820 + index_primexp_10679;
        if (res_10523) {
            int32_t x_10572;
            int32_t x_10573;
            int32_t x_10574;
            int32_t res_10575;
            
            x_10572 = res_9819 - res_10570;
            x_10573 = x_10572 - 1;
            x_10574 = res_9819 * x_10573;
            res_10575 = res_10549 + x_10574;
            res_10571 = res_10575;
        } else {
            int32_t y_10576;
            int32_t res_10577;
            
            y_10576 = res_9819 * res_10549;
            res_10577 = res_10570 + y_10576;
            res_10571 = res_10577;
        }
        if (res_10523) {
            int32_t x_10579;
            int32_t x_10580;
            int32_t x_10581;
            int32_t res_10582;
            
            x_10579 = res_9819 - res_10570;
            x_10580 = x_10579 - 1;
            x_10581 = res_9819 * x_10580;
            res_10582 = res_10560 + x_10581;
            res_10578 = res_10582;
        } else {
            int32_t y_10583;
            int32_t res_10584;
            
            y_10583 = res_9819 * res_10560;
            res_10584 = res_10570 + y_10583;
            res_10578 = res_10584;
        }
        cond_10585 = sle32(0, res_10571);
        res_10586 = slt32(res_10571, y_9923);
        x_10587 = cond_10585 && res_10586;
        if (x_10587) {
            res_10588 = res_10571;
        } else {
            res_10588 = -1;
        }
        if (x_10587) {
            bool cond_10590;
            float res_10591;
            
            cond_10590 = res_10549 == res_10560;
            if (cond_10590) {
                res_10591 = res_10527;
            } else {
                res_10591 = res_10568;
            }
            res_10589 = res_10591;
        } else {
            res_10589 = -1.0F;
        }
        cond_10592 = sle32(0, res_10578);
        res_10593 = slt32(res_10578, y_9923);
        x_10594 = cond_10592 && res_10593;
        if (x_10594) {
            bool cond_10597;
            float res_10598;
            int32_t res_10599;
            
            cond_10597 = res_10549 == res_10560;
            if (cond_10597) {
                res_10598 = -1.0F;
            } else {
                res_10598 = res_10569;
            }
            if (cond_10597) {
                res_10599 = -1;
            } else {
                res_10599 = res_10578;
            }
            res_10595 = res_10598;
            res_10596 = res_10599;
        } else {
            res_10595 = -1.0F;
            res_10596 = -1;
        }
        *(__global float *) &mem_10834[(group_id_10522 * (2 *
                                                          group_sizze_10515) +
                                        local_tid_10521) * 4] = res_10589;
        *(__global float *) &mem_10834[(group_id_10522 * (2 *
                                                          group_sizze_10515) +
                                        group_sizze_10515 + local_tid_10521) *
                                       4] = res_10595;
        *(__global int32_t *) &mem_10837[(group_id_10522 * (2 *
                                                            group_sizze_10515) +
                                          local_tid_10521) * 4] = res_10588;
        *(__global int32_t *) &mem_10837[(group_id_10522 * (2 *
                                                            group_sizze_10515) +
                                          group_sizze_10515 + local_tid_10521) *
                                         4] = res_10596;
    }
    if (thread_active_11074) {
        for (int32_t i_11075 = 0; i_11075 < 2; i_11075++) {
            *(__global float *) &mem_10842[(i_11075 * (x_9940 *
                                                       num_elems_9920) +
                                            gtid_10511 * num_elems_9920 +
                                            gtid_10512) * 4] = *(__global
                                                                 float *) &mem_10834[(group_id_10522 *
                                                                                      (2 *
                                                                                       group_sizze_10515) +
                                                                                      i_11075 *
                                                                                      group_sizze_10515 +
                                                                                      local_tid_10521) *
                                                                                     4];
        }
    }
    if (thread_active_11074) {
        for (int32_t i_11076 = 0; i_11076 < 2; i_11076++) {
            *(__global int32_t *) &mem_10847[(i_11076 * (x_9940 *
                                                         num_elems_9920) +
                                              gtid_10511 * num_elems_9920 +
                                              gtid_10512) * 4] = *(__global
                                                                   int32_t *) &mem_10837[(group_id_10522 *
                                                                                          (2 *
                                                                                           group_sizze_10515) +
                                                                                          i_11076 *
                                                                                          group_sizze_10515 +
                                                                                          local_tid_10521) *
                                                                                         4];
        }
    }
}
__kernel void map_kernel_10609(int32_t i_9938, int32_t x_9940, __global
                               unsigned char *mem_10803, __global
                               unsigned char *mem_10806, __global
                               unsigned char *mem_10809, __global
                               unsigned char *mem_10812, __global
                               unsigned char *mem_10819, __global
                               unsigned char *mem_10822, __global
                               unsigned char *mem_10825, __global
                               unsigned char *mem_10828, __global
                               unsigned char *mem_10831)
{
    int32_t wave_sizze_11069;
    int32_t group_sizze_11070;
    bool thread_active_11071;
    int32_t gtid_10602;
    int32_t global_tid_10609;
    int32_t local_tid_10610;
    int32_t group_id_10611;
    
    global_tid_10609 = get_global_id(0);
    local_tid_10610 = get_local_id(0);
    group_sizze_11070 = get_local_size(0);
    wave_sizze_11069 = LOCKSTEP_WIDTH;
    group_id_10611 = get_group_id(0);
    gtid_10602 = global_tid_10609;
    thread_active_11071 = slt32(gtid_10602, x_9940);
    
    int32_t j_p_i_t_s_10677;
    float x_10612;
    float x_10613;
    float x_10614;
    float x_10615;
    float x_10616;
    float y_10617;
    float res_10618;
    float res_10619;
    bool res_10620;
    float res_10621;
    float res_10622;
    float res_10628;
    float y_10631;
    float arg_10632;
    float res_10633;
    
    if (thread_active_11071) {
        j_p_i_t_s_10677 = i_9938 + gtid_10602;
        x_10612 = *(__global float *) &mem_10803[j_p_i_t_s_10677 * 4];
        x_10613 = *(__global float *) &mem_10806[j_p_i_t_s_10677 * 4];
        x_10614 = *(__global float *) &mem_10809[j_p_i_t_s_10677 * 4];
        x_10615 = *(__global float *) &mem_10812[j_p_i_t_s_10677 * 4];
        x_10616 = x_10615 - x_10613;
        y_10617 = x_10614 - x_10612;
        res_10618 = x_10616 / y_10617;
        res_10619 = (float) fabs(res_10618);
        res_10620 = 1.0F < res_10619;
        if (res_10620) {
            bool cond_10623;
            float res_10624;
            float res_10625;
            
            cond_10623 = res_10618 < 0.0F;
            if (cond_10623) {
                res_10624 = x_10612;
            } else {
                res_10624 = x_10614;
            }
            if (cond_10623) {
                float res_10626 = 0.0F - x_10613;
                
                res_10625 = res_10626;
            } else {
                float res_10627 = 0.0F - x_10615;
                
                res_10625 = res_10627;
            }
            res_10621 = res_10625;
            res_10622 = res_10624;
        } else {
            res_10621 = x_10612;
            res_10622 = x_10613;
        }
        if (res_10620) {
            float negate_arg_10629;
            float res_10630;
            
            negate_arg_10629 = 1.0F / res_10618;
            res_10630 = 0.0F - negate_arg_10629;
            res_10628 = res_10630;
        } else {
            res_10628 = res_10618;
        }
        y_10631 = res_10628 * res_10628;
        arg_10632 = 1.0F + y_10631;
        res_10633 = futrts_sqrt32(arg_10632);
    }
    if (thread_active_11071) {
        *(__global bool *) &mem_10819[gtid_10602] = res_10620;
    }
    if (thread_active_11071) {
        *(__global float *) &mem_10822[gtid_10602 * 4] = res_10621;
    }
    if (thread_active_11071) {
        *(__global float *) &mem_10825[gtid_10602 * 4] = res_10622;
    }
    if (thread_active_11071) {
        *(__global float *) &mem_10828[gtid_10602 * 4] = res_10628;
    }
    if (thread_active_11071) {
        *(__global float *) &mem_10831[gtid_10602 * 4] = res_10633;
    }
}
__kernel void reduce_kernel_10946(__local volatile int64_t *mem_aligned_0,
                                  int32_t num_groups_10907, __global
                                  unsigned char *mem_10961, __global
                                  unsigned char *mem_10967)
{
    __local volatile char *restrict mem_10964 = mem_aligned_0;
    int32_t wave_sizze_11046;
    int32_t group_sizze_11047;
    bool thread_active_11048;
    int32_t global_tid_10946;
    int32_t local_tid_10947;
    int32_t group_id_10948;
    
    global_tid_10946 = get_global_id(0);
    local_tid_10947 = get_local_id(0);
    group_sizze_11047 = get_local_size(0);
    wave_sizze_11046 = LOCKSTEP_WIDTH;
    group_id_10948 = get_group_id(0);
    thread_active_11048 = 1;
    
    bool in_bounds_10949;
    int64_t x_10968;
    
    if (thread_active_11048) {
        in_bounds_10949 = slt32(local_tid_10947, num_groups_10907);
        if (in_bounds_10949) {
            int64_t x_10950 = *(__global
                                int64_t *) &mem_10961[global_tid_10946 * 8];
            
            x_10968 = x_10950;
        } else {
            x_10968 = 0;
        }
    }
    
    int64_t final_result_10954;
    
    for (int32_t comb_iter_11049 = 0; comb_iter_11049 <
         squot32(max_num_groups_10902 + max_num_groups_10902 - 1,
                 max_num_groups_10902); comb_iter_11049++) {
        int32_t combine_id_10953;
        int32_t flat_comb_id_11050 = comb_iter_11049 * max_num_groups_10902 +
                local_tid_10947;
        
        combine_id_10953 = flat_comb_id_11050;
        if (slt32(combine_id_10953, max_num_groups_10902) && 1) {
            *(__local int64_t *) &mem_10964[combine_id_10953 * 8] = x_10968;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_11052;
    int32_t skip_waves_11051;
    int64_t x_10887;
    int64_t y_10888;
    int32_t my_index_10914;
    int32_t other_index_10915;
    
    my_index_10914 = local_tid_10947;
    offset_11052 = 0;
    other_index_10915 = local_tid_10947 + offset_11052;
    if (slt32(local_tid_10947, max_num_groups_10902)) {
        x_10887 = *(__local int64_t *) &mem_10964[(local_tid_10947 +
                                                   offset_11052) * 8];
    }
    offset_11052 = 1;
    other_index_10915 = local_tid_10947 + offset_11052;
    while (slt32(offset_11052, wave_sizze_11046)) {
        if (slt32(other_index_10915, max_num_groups_10902) &&
            ((local_tid_10947 - squot32(local_tid_10947, wave_sizze_11046) *
              wave_sizze_11046) & (2 * offset_11052 - 1)) == 0) {
            // read array element
            {
                y_10888 = *(volatile __local
                            int64_t *) &mem_10964[(local_tid_10947 +
                                                   offset_11052) * 8];
            }
            
            int64_t zz_10889;
            
            if (thread_active_11048) {
                zz_10889 = smax64(x_10887, y_10888);
            }
            x_10887 = zz_10889;
            *(volatile __local int64_t *) &mem_10964[local_tid_10947 * 8] =
                x_10887;
        }
        offset_11052 *= 2;
        other_index_10915 = local_tid_10947 + offset_11052;
    }
    skip_waves_11051 = 1;
    while (slt32(skip_waves_11051, squot32(max_num_groups_10902 +
                                           wave_sizze_11046 - 1,
                                           wave_sizze_11046))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_11052 = skip_waves_11051 * wave_sizze_11046;
        other_index_10915 = local_tid_10947 + offset_11052;
        if (slt32(other_index_10915, max_num_groups_10902) &&
            ((local_tid_10947 - squot32(local_tid_10947, wave_sizze_11046) *
              wave_sizze_11046) == 0 && (squot32(local_tid_10947,
                                                 wave_sizze_11046) & (2 *
                                                                      skip_waves_11051 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                y_10888 = *(__local int64_t *) &mem_10964[(local_tid_10947 +
                                                           offset_11052) * 8];
            }
            
            int64_t zz_10889;
            
            if (thread_active_11048) {
                zz_10889 = smax64(x_10887, y_10888);
            }
            x_10887 = zz_10889;
            *(__local int64_t *) &mem_10964[local_tid_10947 * 8] = x_10887;
        }
        skip_waves_11051 *= 2;
    }
    final_result_10954 = x_10887;
    if (local_tid_10947 == 0) {
        *(__global int64_t *) &mem_10967[group_id_10948 * 8] =
            final_result_10954;
    }
}
__kernel void scan1_kernel_10312(__local volatile int64_t *mem_aligned_0,
                                 __local volatile int64_t *mem_aligned_1,
                                 int32_t num_iterations_10317, int32_t y_10321,
                                 int32_t convop_x_10734, __global
                                 unsigned char *mem_10745, __global
                                 unsigned char *mem_10748, __global
                                 unsigned char *mem_10751, __global
                                 unsigned char *mem_10754, __global
                                 unsigned char *mem_10770, __global
                                 unsigned char *mem_10773)
{
    __local volatile char *restrict mem_10758 = mem_aligned_0;
    __local volatile char *restrict mem_10761 = mem_aligned_1;
    int32_t wave_sizze_10995;
    int32_t group_sizze_10996;
    bool thread_active_10997;
    int32_t global_tid_10312;
    int32_t local_tid_10313;
    int32_t group_id_10314;
    
    global_tid_10312 = get_global_id(0);
    local_tid_10313 = get_local_id(0);
    group_sizze_10996 = get_local_size(0);
    wave_sizze_10995 = LOCKSTEP_WIDTH;
    group_id_10314 = get_group_id(0);
    thread_active_10997 = 1;
    
    int32_t x_10322;
    bool is_first_thread_10345;
    bool result_10357;
    int32_t result_10358;
    
    if (thread_active_10997) {
        x_10322 = group_id_10314 * y_10321;
        is_first_thread_10345 = local_tid_10313 == 0;
        
        bool x_flag_merge_10318;
        int32_t x_merge_10319;
        
        x_flag_merge_10318 = 0;
        x_merge_10319 = 0;
        for (int32_t i_10320 = 0; i_10320 < num_iterations_10317; i_10320++) {
            int32_t y_10323 = group_sizze_10258 * i_10320;
            int32_t offset_10324 = x_10322 + y_10323;
            int32_t j_10325 = local_tid_10313 + offset_10324;
            bool cond_10326 = slt32(j_10325, convop_x_10734);
            bool foldres_10330;
            int32_t foldres_10331;
            
            if (cond_10326) {
                int32_t cmpop_x_10650;
                bool index_primexp_10651;
                int32_t new_index_10646;
                int32_t binop_y_10648;
                int32_t new_index_10649;
                bool res_r_flat_elem_10328;
                int32_t part_res_10279;
                int32_t part_res_10280;
                bool new_flag_10283;
                int32_t seg_lhs_10284;
                int32_t zz_10287;
                
                cmpop_x_10650 = srem32(j_10325, 4);
                index_primexp_10651 = cmpop_x_10650 == 0;
                new_index_10646 = squot32(j_10325, 4);
                binop_y_10648 = 4 * new_index_10646;
                new_index_10649 = j_10325 - binop_y_10648;
                res_r_flat_elem_10328 = *(__global
                                          bool *) &mem_10754[new_index_10646 *
                                                             4 +
                                                             new_index_10649];
                if (res_r_flat_elem_10328) {
                    part_res_10279 = 0;
                } else {
                    part_res_10279 = 1;
                }
                if (res_r_flat_elem_10328) {
                    part_res_10280 = 1;
                } else {
                    part_res_10280 = 0;
                }
                new_flag_10283 = x_flag_merge_10318 || index_primexp_10651;
                if (index_primexp_10651) {
                    seg_lhs_10284 = 0;
                } else {
                    seg_lhs_10284 = x_merge_10319;
                }
                zz_10287 = part_res_10280 + seg_lhs_10284;
                *(__global int32_t *) &mem_10751[j_10325 * 4] = part_res_10279;
                foldres_10330 = new_flag_10283;
                foldres_10331 = zz_10287;
            } else {
                foldres_10330 = x_flag_merge_10318;
                foldres_10331 = x_merge_10319;
            }
            for (int32_t comb_iter_11003 = 0; comb_iter_11003 <
                 squot32(group_sizze_10258 + group_sizze_10258 - 1,
                         group_sizze_10258); comb_iter_11003++) {
                int32_t combine_id_10333;
                int32_t flat_comb_id_11004 = comb_iter_11003 *
                        group_sizze_10258 + local_tid_10313;
                
                combine_id_10333 = flat_comb_id_11004;
                if (slt32(combine_id_10333, group_sizze_10258) && 1) {
                    *(__local bool *) &mem_10758[combine_id_10333] =
                        foldres_10330;
                    *(__local int32_t *) &mem_10761[combine_id_10333 * 4] =
                        foldres_10331;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t my_index_10288;
            int32_t other_index_10289;
            bool x_flag_10290;
            int32_t x_10291;
            bool y_flag_10292;
            int32_t y_10293;
            int32_t my_index_11005;
            int32_t other_index_11006;
            bool x_flag_11007;
            int32_t x_11008;
            bool y_flag_11009;
            int32_t y_11010;
            
            my_index_10288 = local_tid_10313;
            if (slt32(local_tid_10313, group_sizze_10258)) {
                y_flag_10292 = *(volatile __local
                                 bool *) &mem_10758[local_tid_10313 *
                                                    sizeof(bool)];
                y_10293 = *(volatile __local
                            int32_t *) &mem_10761[local_tid_10313 *
                                                  sizeof(int32_t)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                int32_t skip_threads_11014 = 1;
                
                while (slt32(skip_threads_11014, 32)) {
                    if (slt32(local_tid_10313, group_sizze_10258) &&
                        sle32(skip_threads_11014, local_tid_10313 -
                              squot32(local_tid_10313, 32) * 32)) {
                        // read operands
                        {
                            x_flag_10290 = *(volatile __local
                                             bool *) &mem_10758[(local_tid_10313 -
                                                                 skip_threads_11014) *
                                                                sizeof(bool)];
                            x_10291 = *(volatile __local
                                        int32_t *) &mem_10761[(local_tid_10313 -
                                                               skip_threads_11014) *
                                                              sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            bool new_flag_10294;
                            int32_t seg_lhs_10295;
                            int32_t zz_10298;
                            
                            new_flag_10294 = x_flag_10290 || y_flag_10292;
                            if (y_flag_10292) {
                                seg_lhs_10295 = 0;
                            } else {
                                seg_lhs_10295 = x_10291;
                            }
                            zz_10298 = y_10293 + seg_lhs_10295;
                            y_flag_10292 = new_flag_10294;
                            y_10293 = zz_10298;
                        }
                    }
                    if (sle32(wave_sizze_10995, skip_threads_11014)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (slt32(local_tid_10313, group_sizze_10258) &&
                        sle32(skip_threads_11014, local_tid_10313 -
                              squot32(local_tid_10313, 32) * 32)) {
                        // write result
                        {
                            *(volatile __local
                              bool *) &mem_10758[local_tid_10313 *
                                                 sizeof(bool)] = y_flag_10292;
                            *(volatile __local
                              int32_t *) &mem_10761[local_tid_10313 *
                                                    sizeof(int32_t)] = y_10293;
                        }
                    }
                    if (sle32(wave_sizze_10995, skip_threads_11014)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_11014 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_10313 - squot32(local_tid_10313, 32) * 32) ==
                    31 && slt32(local_tid_10313, group_sizze_10258)) {
                    *(volatile __local
                      bool *) &mem_10758[squot32(local_tid_10313, 32) *
                                         sizeof(bool)] = y_flag_10292;
                    *(volatile __local
                      int32_t *) &mem_10761[squot32(local_tid_10313, 32) *
                                            sizeof(int32_t)] = y_10293;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
            {
                if (squot32(local_tid_10313, 32) == 0 && slt32(local_tid_10313,
                                                               group_sizze_10258)) {
                    y_flag_11009 = *(volatile __local
                                     bool *) &mem_10758[local_tid_10313 *
                                                        sizeof(bool)];
                    y_11010 = *(volatile __local
                                int32_t *) &mem_10761[local_tid_10313 *
                                                      sizeof(int32_t)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    int32_t skip_threads_11015 = 1;
                    
                    while (slt32(skip_threads_11015, 32)) {
                        if ((squot32(local_tid_10313, 32) == 0 &&
                             slt32(local_tid_10313, group_sizze_10258)) &&
                            sle32(skip_threads_11015, local_tid_10313 -
                                  squot32(local_tid_10313, 32) * 32)) {
                            // read operands
                            {
                                x_flag_11007 = *(volatile __local
                                                 bool *) &mem_10758[(local_tid_10313 -
                                                                     skip_threads_11015) *
                                                                    sizeof(bool)];
                                x_11008 = *(volatile __local
                                            int32_t *) &mem_10761[(local_tid_10313 -
                                                                   skip_threads_11015) *
                                                                  sizeof(int32_t)];
                            }
                            // perform operation
                            {
                                bool new_flag_11011;
                                int32_t seg_lhs_11012;
                                int32_t zz_11013;
                                
                                new_flag_11011 = x_flag_11007 || y_flag_11009;
                                if (y_flag_11009) {
                                    seg_lhs_11012 = 0;
                                } else {
                                    seg_lhs_11012 = x_11008;
                                }
                                zz_11013 = y_11010 + seg_lhs_11012;
                                y_flag_11009 = new_flag_11011;
                                y_11010 = zz_11013;
                            }
                        }
                        if (sle32(wave_sizze_10995, skip_threads_11015)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if ((squot32(local_tid_10313, 32) == 0 &&
                             slt32(local_tid_10313, group_sizze_10258)) &&
                            sle32(skip_threads_11015, local_tid_10313 -
                                  squot32(local_tid_10313, 32) * 32)) {
                            // write result
                            {
                                *(volatile __local
                                  bool *) &mem_10758[local_tid_10313 *
                                                     sizeof(bool)] =
                                    y_flag_11009;
                                *(volatile __local
                                  int32_t *) &mem_10761[local_tid_10313 *
                                                        sizeof(int32_t)] =
                                    y_11010;
                            }
                        }
                        if (sle32(wave_sizze_10995, skip_threads_11015)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_11015 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_10313, 32) == 0 ||
                      !slt32(local_tid_10313, group_sizze_10258))) {
                    // read operands
                    {
                        x_flag_10290 = *(volatile __local
                                         bool *) &mem_10758[(squot32(local_tid_10313,
                                                                     32) - 1) *
                                                            sizeof(bool)];
                        x_10291 = *(volatile __local
                                    int32_t *) &mem_10761[(squot32(local_tid_10313,
                                                                   32) - 1) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        bool new_flag_10294;
                        int32_t seg_lhs_10295;
                        int32_t zz_10298;
                        
                        new_flag_10294 = x_flag_10290 || y_flag_10292;
                        if (y_flag_10292) {
                            seg_lhs_10295 = 0;
                        } else {
                            seg_lhs_10295 = x_10291;
                        }
                        zz_10298 = y_10293 + seg_lhs_10295;
                        y_flag_10292 = new_flag_10294;
                        y_10293 = zz_10298;
                    }
                    // write final result
                    {
                        *(volatile __local bool *) &mem_10758[local_tid_10313 *
                                                              sizeof(bool)] =
                            y_flag_10292;
                        *(volatile __local
                          int32_t *) &mem_10761[local_tid_10313 *
                                                sizeof(int32_t)] = y_10293;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_10313, 32) == 0) {
                    *(volatile __local bool *) &mem_10758[local_tid_10313 *
                                                          sizeof(bool)] =
                        y_flag_10292;
                    *(volatile __local int32_t *) &mem_10761[local_tid_10313 *
                                                             sizeof(int32_t)] =
                        y_10293;
                }
            }
            if (cond_10326) {
                bool scanned_elem_10339;
                int32_t scanned_elem_10340;
                
                scanned_elem_10339 = *(__local
                                       bool *) &mem_10758[local_tid_10313];
                scanned_elem_10340 = *(__local
                                       int32_t *) &mem_10761[local_tid_10313 *
                                                             4];
                *(__global bool *) &mem_10745[j_10325] = scanned_elem_10339;
                *(__global int32_t *) &mem_10748[j_10325 * 4] =
                    scanned_elem_10340;
            }
            
            bool new_scan_carry_10348;
            int32_t new_scan_carry_10349;
            
            if (is_first_thread_10345) {
                bool carry_10346;
                int32_t carry_10347;
                
                carry_10346 = *(__local bool *) &mem_10758[y_10261];
                carry_10347 = *(__local int32_t *) &mem_10761[y_10261 * 4];
                new_scan_carry_10348 = carry_10346;
                new_scan_carry_10349 = carry_10347;
            } else {
                new_scan_carry_10348 = 0;
                new_scan_carry_10349 = 0;
            }
            
            bool new_carry_sync_10352;
            int32_t new_carry_sync_10353;
            
            new_carry_sync_10352 = new_scan_carry_10348;
            new_carry_sync_10353 = new_scan_carry_10349;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_flag_merge_tmp_11001 = new_carry_sync_10352;
            int32_t x_merge_tmp_11002;
            
            x_merge_tmp_11002 = new_carry_sync_10353;
            x_flag_merge_10318 = x_flag_merge_tmp_11001;
            x_merge_10319 = x_merge_tmp_11002;
        }
        result_10357 = x_flag_merge_10318;
        result_10358 = x_merge_10319;
    }
    if (local_tid_10313 == 0) {
        *(__global bool *) &mem_10770[group_id_10314] = result_10357;
    }
    if (local_tid_10313 == 0) {
        *(__global int32_t *) &mem_10773[group_id_10314 * 4] = result_10358;
    }
}
__kernel void scan2_kernel_10370(__local volatile int64_t *mem_aligned_0,
                                 __local volatile int64_t *mem_aligned_1,
                                 int32_t num_groups_10265, __global
                                 unsigned char *mem_10770, __global
                                 unsigned char *mem_10773, __global
                                 unsigned char *mem_10780, __global
                                 unsigned char *mem_10783)
{
    __local volatile char *restrict mem_10775 = mem_aligned_0;
    __local volatile char *restrict mem_10778 = mem_aligned_1;
    int32_t wave_sizze_11018;
    int32_t group_sizze_11019;
    bool thread_active_11020;
    int32_t global_tid_10370;
    int32_t local_tid_10371;
    int32_t group_id_10372;
    
    global_tid_10370 = get_global_id(0);
    local_tid_10371 = get_local_id(0);
    group_sizze_11019 = get_local_size(0);
    wave_sizze_11018 = LOCKSTEP_WIDTH;
    group_id_10372 = get_group_id(0);
    thread_active_11020 = 1;
    for (int32_t comb_iter_11021 = 0; comb_iter_11021 <
         squot32(num_groups_10265 + num_groups_10265 - 1, num_groups_10265);
         comb_iter_11021++) {
        int32_t combine_id_10373;
        int32_t flat_comb_id_11022 = comb_iter_11021 * num_groups_10265 +
                local_tid_10371;
        
        combine_id_10373 = flat_comb_id_11022;
        if (slt32(combine_id_10373, num_groups_10265) && 1) {
            bool unused_flag_array_scan_carry_out_elem_10374 = *(__global
                                                                 bool *) &mem_10770[combine_id_10373];
            int32_t offsets_r_flat_scan_carry_out_elem_10375 = *(__global
                                                                 int32_t *) &mem_10773[combine_id_10373 *
                                                                                       4];
            
            *(__local bool *) &mem_10775[combine_id_10373] =
                unused_flag_array_scan_carry_out_elem_10374;
            *(__local int32_t *) &mem_10778[combine_id_10373 * 4] =
                offsets_r_flat_scan_carry_out_elem_10375;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t my_index_10359;
    int32_t other_index_10360;
    bool x_flag_10361;
    int32_t x_10362;
    bool y_flag_10363;
    int32_t y_10364;
    int32_t my_index_11023;
    int32_t other_index_11024;
    bool x_flag_11025;
    int32_t x_11026;
    bool y_flag_11027;
    int32_t y_11028;
    
    my_index_10359 = local_tid_10371;
    if (slt32(local_tid_10371, num_groups_10265)) {
        y_flag_10363 = *(volatile __local bool *) &mem_10775[local_tid_10371 *
                                                             sizeof(bool)];
        y_10364 = *(volatile __local int32_t *) &mem_10778[local_tid_10371 *
                                                           sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        int32_t skip_threads_11032 = 1;
        
        while (slt32(skip_threads_11032, 32)) {
            if (slt32(local_tid_10371, num_groups_10265) &&
                sle32(skip_threads_11032, local_tid_10371 -
                      squot32(local_tid_10371, 32) * 32)) {
                // read operands
                {
                    x_flag_10361 = *(volatile __local
                                     bool *) &mem_10775[(local_tid_10371 -
                                                         skip_threads_11032) *
                                                        sizeof(bool)];
                    x_10362 = *(volatile __local
                                int32_t *) &mem_10778[(local_tid_10371 -
                                                       skip_threads_11032) *
                                                      sizeof(int32_t)];
                }
                // perform operation
                {
                    bool new_flag_10365;
                    int32_t seg_lhs_10366;
                    int32_t zz_10369;
                    
                    if (thread_active_11020) {
                        new_flag_10365 = x_flag_10361 || y_flag_10363;
                        if (y_flag_10363) {
                            seg_lhs_10366 = 0;
                        } else {
                            seg_lhs_10366 = x_10362;
                        }
                        zz_10369 = y_10364 + seg_lhs_10366;
                    }
                    y_flag_10363 = new_flag_10365;
                    y_10364 = zz_10369;
                }
            }
            if (sle32(wave_sizze_11018, skip_threads_11032)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (slt32(local_tid_10371, num_groups_10265) &&
                sle32(skip_threads_11032, local_tid_10371 -
                      squot32(local_tid_10371, 32) * 32)) {
                // write result
                {
                    *(volatile __local bool *) &mem_10775[local_tid_10371 *
                                                          sizeof(bool)] =
                        y_flag_10363;
                    *(volatile __local int32_t *) &mem_10778[local_tid_10371 *
                                                             sizeof(int32_t)] =
                        y_10364;
                }
            }
            if (sle32(wave_sizze_11018, skip_threads_11032)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_11032 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_10371 - squot32(local_tid_10371, 32) * 32) == 31 &&
            slt32(local_tid_10371, num_groups_10265)) {
            *(volatile __local bool *) &mem_10775[squot32(local_tid_10371, 32) *
                                                  sizeof(bool)] = y_flag_10363;
            *(volatile __local int32_t *) &mem_10778[squot32(local_tid_10371,
                                                             32) *
                                                     sizeof(int32_t)] = y_10364;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_tid_10371, 32) == 0 && slt32(local_tid_10371,
                                                       num_groups_10265)) {
            y_flag_11027 = *(volatile __local
                             bool *) &mem_10775[local_tid_10371 * sizeof(bool)];
            y_11028 = *(volatile __local int32_t *) &mem_10778[local_tid_10371 *
                                                               sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            int32_t skip_threads_11033 = 1;
            
            while (slt32(skip_threads_11033, 32)) {
                if ((squot32(local_tid_10371, 32) == 0 && slt32(local_tid_10371,
                                                                num_groups_10265)) &&
                    sle32(skip_threads_11033, local_tid_10371 -
                          squot32(local_tid_10371, 32) * 32)) {
                    // read operands
                    {
                        x_flag_11025 = *(volatile __local
                                         bool *) &mem_10775[(local_tid_10371 -
                                                             skip_threads_11033) *
                                                            sizeof(bool)];
                        x_11026 = *(volatile __local
                                    int32_t *) &mem_10778[(local_tid_10371 -
                                                           skip_threads_11033) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        bool new_flag_11029;
                        int32_t seg_lhs_11030;
                        int32_t zz_11031;
                        
                        if (thread_active_11020) {
                            new_flag_11029 = x_flag_11025 || y_flag_11027;
                            if (y_flag_11027) {
                                seg_lhs_11030 = 0;
                            } else {
                                seg_lhs_11030 = x_11026;
                            }
                            zz_11031 = y_11028 + seg_lhs_11030;
                        }
                        y_flag_11027 = new_flag_11029;
                        y_11028 = zz_11031;
                    }
                }
                if (sle32(wave_sizze_11018, skip_threads_11033)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if ((squot32(local_tid_10371, 32) == 0 && slt32(local_tid_10371,
                                                                num_groups_10265)) &&
                    sle32(skip_threads_11033, local_tid_10371 -
                          squot32(local_tid_10371, 32) * 32)) {
                    // write result
                    {
                        *(volatile __local bool *) &mem_10775[local_tid_10371 *
                                                              sizeof(bool)] =
                            y_flag_11027;
                        *(volatile __local
                          int32_t *) &mem_10778[local_tid_10371 *
                                                sizeof(int32_t)] = y_11028;
                    }
                }
                if (sle32(wave_sizze_11018, skip_threads_11033)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_11033 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_10371, 32) == 0 || !slt32(local_tid_10371,
                                                          num_groups_10265))) {
            // read operands
            {
                x_flag_10361 = *(volatile __local
                                 bool *) &mem_10775[(squot32(local_tid_10371,
                                                             32) - 1) *
                                                    sizeof(bool)];
                x_10362 = *(volatile __local
                            int32_t *) &mem_10778[(squot32(local_tid_10371,
                                                           32) - 1) *
                                                  sizeof(int32_t)];
            }
            // perform operation
            {
                bool new_flag_10365;
                int32_t seg_lhs_10366;
                int32_t zz_10369;
                
                if (thread_active_11020) {
                    new_flag_10365 = x_flag_10361 || y_flag_10363;
                    if (y_flag_10363) {
                        seg_lhs_10366 = 0;
                    } else {
                        seg_lhs_10366 = x_10362;
                    }
                    zz_10369 = y_10364 + seg_lhs_10366;
                }
                y_flag_10363 = new_flag_10365;
                y_10364 = zz_10369;
            }
            // write final result
            {
                *(volatile __local bool *) &mem_10775[local_tid_10371 *
                                                      sizeof(bool)] =
                    y_flag_10363;
                *(volatile __local int32_t *) &mem_10778[local_tid_10371 *
                                                         sizeof(int32_t)] =
                    y_10364;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_10371, 32) == 0) {
            *(volatile __local bool *) &mem_10775[local_tid_10371 *
                                                  sizeof(bool)] = y_flag_10363;
            *(volatile __local int32_t *) &mem_10778[local_tid_10371 *
                                                     sizeof(int32_t)] = y_10364;
        }
    }
    
    bool scanned_elem_10380;
    int32_t scanned_elem_10381;
    
    if (thread_active_11020) {
        scanned_elem_10380 = *(__local bool *) &mem_10775[local_tid_10371];
        scanned_elem_10381 = *(__local int32_t *) &mem_10778[local_tid_10371 *
                                                             4];
    }
    *(__global bool *) &mem_10780[global_tid_10370] = scanned_elem_10380;
    *(__global int32_t *) &mem_10783[global_tid_10370 * 4] = scanned_elem_10381;
}
"""
# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        s = b''
        for _ in range(n):
            s += self.get_char()
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        map(f.unget_char, read[::-1])
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in string.hexdigits:
            s += c
            c = f.get_char()
        elif c == '_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16))


def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in [b'x', b'X']:
        c = f.get_char() # skip X
        s += parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == '_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
    if len(s) == 0:
        raise ValueError
    return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      s = c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      s = parse_int(f)

    return s

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    for i in range(rank):
        parse_specific_string(f, '[]')
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return None

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank-1)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if elems == None:
        # Empty array
        return np.empty([0]*rank, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype='<'+bin_fmt)
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def write_value(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[]' for _ in v.shape[1:]]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

################################################################################
### end of values.py
################################################################################
# Helper functions dealing with memory blocks.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, dim):
  return np.ctypeslib.as_array(x, shape=dim)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset, bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)
def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.exit(exitcode)
# Scalar functions.

import numpy as np
import struct

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_round64(x):
  return np.round(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_round32(x):
  return np.round(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])
class forwardprojection_doubleparallel:
  entry_points = {"main": (["[]f32", "[]f32", "[]f32", "i32"], ["[]f32"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=None, default_num_groups=None,
               default_tile_size=None, sizes={}):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width", 32),
     ("AMD Accelerated Parallel Processing", cl.device_type.GPU, "lockstep_width",
      64), ("", cl.device_type.GPU, "lockstep_width", 1), ("", cl.device_type.GPU,
                                                           "num_groups", 128), ("",
                                                                                cl.device_type.GPU,
                                                                                "group_size",
                                                                                256),
     ("", cl.device_type.GPU, "tile_size", 32), ("", cl.device_type.CPU,
                                                 "lockstep_width", 1), ("",
                                                                        cl.device_type.CPU,
                                                                        "num_groups",
                                                                        "MAX_COMPUTE_UNITS"),
     ("", cl.device_type.CPU, "group_size", 32), ("", cl.device_type.CPU,
                                                  "tile_size", 4)]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       size_heuristics=size_heuristics,
                                       required_types=["i32", "i64", "f32", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"group_size_10090": {"class": "group_size", "value": None},
                                        "group_size_10106": {"class": "group_size", "value": None},
                                        "group_size_10118": {"class": "group_size", "value": None},
                                        "group_size_10160": {"class": "group_size", "value": None},
                                        "group_size_10257": {"class": "group_size", "value": None},
                                        "max_num_groups_10259": {"class": "num_groups", "value": None},
                                        "group_size_10407": {"class": "group_size", "value": None},
                                        "group_size_10417": {"class": "group_size", "value": None},
                                        "group_size_10472": {"class": "group_size", "value": None},
                                        "group_size_10514": {"class": "group_size", "value": None},
                                        "group_size_10603": {"class": "group_size", "value": None},
                                        "group_size_10899": {"class": "group_size", "value": None},
                                        "max_num_groups_10901": {"class": "num_groups", "value": None},
                                        "group_size_11061": {"class": "group_size", "value": None}})
    self.chunked_reduce_kernel_10916_var = program.chunked_reduce_kernel_10916
    self.fut_kernel_map_transpose_f32_var = program.fut_kernel_map_transpose_f32
    self.fut_kernel_map_transpose_i32_var = program.fut_kernel_map_transpose_i32
    self.fut_kernel_map_transpose_lowheight_f32_var = program.fut_kernel_map_transpose_lowheight_f32
    self.fut_kernel_map_transpose_lowheight_i32_var = program.fut_kernel_map_transpose_lowheight_i32
    self.fut_kernel_map_transpose_lowwidth_f32_var = program.fut_kernel_map_transpose_lowwidth_f32
    self.fut_kernel_map_transpose_lowwidth_i32_var = program.fut_kernel_map_transpose_lowwidth_i32
    self.fut_kernel_map_transpose_small_f32_var = program.fut_kernel_map_transpose_small_f32
    self.fut_kernel_map_transpose_small_i32_var = program.fut_kernel_map_transpose_small_i32
    self.kernel_replicate_9913_var = program.kernel_replicate_9913
    self.map_kernel_10096_var = program.map_kernel_10096
    self.map_kernel_10112_var = program.map_kernel_10112
    self.map_kernel_10124_var = program.map_kernel_10124
    self.map_kernel_10166_var = program.map_kernel_10166
    self.map_kernel_10413_var = program.map_kernel_10413
    self.map_kernel_10423_var = program.map_kernel_10423
    self.map_kernel_10478_var = program.map_kernel_10478
    self.map_kernel_10520_var = program.map_kernel_10520
    self.map_kernel_10609_var = program.map_kernel_10609
    self.reduce_kernel_10946_var = program.reduce_kernel_10946
    self.scan1_kernel_10312_var = program.scan1_kernel_10312
    self.scan2_kernel_10370_var = program.scan2_kernel_10370
  def futhark_map_transpose_opencl_f32(self, destmem_0, destoffset_1, srcmem_2,
                                       srcoffset_3, num_arrays_4, x_elems_5,
                                       y_elems_6, in_elems_7, out_elems_8):
    if (((num_arrays_4 * x_elems_5) * y_elems_6) == np.int32(0)):
      pass
    else:
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(4))))
        if synchronous:
          self.queue.finish()
      else:
        if (sle32(x_elems_5, squot32(np.int32(16),
                                     np.int32(2))) and slt32(np.int32(16),
                                                             y_elems_6)):
          muly_9 = squot32(np.int32(16), x_elems_5)
          new_height_10 = squot32(((y_elems_6 + muly_9) - np.int32(1)), muly_9)
          if ((((1 * (x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                np.int32(16))),
                                         np.int32(16)))) * (new_height_10 + srem32((np.int32(16) - srem32(new_height_10,
                                                                                                          np.int32(16))),
                                                                                   np.int32(16)))) * num_arrays_4) != 0):
            self.fut_kernel_map_transpose_lowwidth_f32_var.set_args(destmem_0,
                                                                    np.int32(destoffset_1),
                                                                    srcmem_2,
                                                                    np.int32(srcoffset_3),
                                                                    np.int32(x_elems_5),
                                                                    np.int32(y_elems_6),
                                                                    np.int32(in_elems_7),
                                                                    np.int32(out_elems_8),
                                                                    np.int32(muly_9),
                                                                    cl.LocalMemory(np.long((np.int32(272) * np.int32(4)))))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.fut_kernel_map_transpose_lowwidth_f32_var,
                                       (np.long((x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                                           np.int32(16))),
                                                                    np.int32(16)))),
                                        np.long((new_height_10 + srem32((np.int32(16) - srem32(new_height_10,
                                                                                               np.int32(16))),
                                                                        np.int32(16)))),
                                        np.long(num_arrays_4)),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              self.queue.finish()
        else:
          if (sle32(y_elems_6, squot32(np.int32(16),
                                       np.int32(2))) and slt32(np.int32(16),
                                                               x_elems_5)):
            mulx_11 = squot32(np.int32(16), y_elems_6)
            new_width_12 = squot32(((x_elems_5 + mulx_11) - np.int32(1)),
                                   mulx_11)
            if ((((1 * (new_width_12 + srem32((np.int32(16) - srem32(new_width_12,
                                                                     np.int32(16))),
                                              np.int32(16)))) * (y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                                           np.int32(16))),
                                                                                    np.int32(16)))) * num_arrays_4) != 0):
              self.fut_kernel_map_transpose_lowheight_f32_var.set_args(destmem_0,
                                                                       np.int32(destoffset_1),
                                                                       srcmem_2,
                                                                       np.int32(srcoffset_3),
                                                                       np.int32(x_elems_5),
                                                                       np.int32(y_elems_6),
                                                                       np.int32(in_elems_7),
                                                                       np.int32(out_elems_8),
                                                                       np.int32(mulx_11),
                                                                       cl.LocalMemory(np.long((np.int32(272) * np.int32(4)))))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.fut_kernel_map_transpose_lowheight_f32_var,
                                         (np.long((new_width_12 + srem32((np.int32(16) - srem32(new_width_12,
                                                                                                np.int32(16))),
                                                                         np.int32(16)))),
                                          np.long((y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                             np.int32(16))),
                                                                      np.int32(16)))),
                                          np.long(num_arrays_4)),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                self.queue.finish()
          else:
            if (sle32(x_elems_5, squot32(np.int32(16),
                                         np.int32(2))) and sle32(y_elems_6,
                                                                 squot32(np.int32(16),
                                                                         np.int32(2)))):
              if ((1 * (((num_arrays_4 * x_elems_5) * y_elems_6) + srem32((np.int32(256) - srem32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                                                                  np.int32(256))),
                                                                          np.int32(256)))) != 0):
                self.fut_kernel_map_transpose_small_f32_var.set_args(destmem_0,
                                                                     np.int32(destoffset_1),
                                                                     srcmem_2,
                                                                     np.int32(srcoffset_3),
                                                                     np.int32(num_arrays_4),
                                                                     np.int32(x_elems_5),
                                                                     np.int32(y_elems_6),
                                                                     np.int32(in_elems_7),
                                                                     np.int32(out_elems_8))
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.fut_kernel_map_transpose_small_f32_var,
                                           (np.long((((num_arrays_4 * x_elems_5) * y_elems_6) + srem32((np.int32(256) - srem32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                                                                                               np.int32(256))),
                                                                                                       np.int32(256)))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  self.queue.finish()
            else:
              if ((((1 * (x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                    np.int32(16))),
                                             np.int32(16)))) * (y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                                          np.int32(16))),
                                                                                   np.int32(16)))) * num_arrays_4) != 0):
                self.fut_kernel_map_transpose_f32_var.set_args(destmem_0,
                                                               np.int32(destoffset_1),
                                                               srcmem_2,
                                                               np.int32(srcoffset_3),
                                                               np.int32(x_elems_5),
                                                               np.int32(y_elems_6),
                                                               np.int32(in_elems_7),
                                                               np.int32(out_elems_8),
                                                               cl.LocalMemory(np.long((np.int32(272) * np.int32(4)))))
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.fut_kernel_map_transpose_f32_var,
                                           (np.long((x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                                               np.int32(16))),
                                                                        np.int32(16)))),
                                            np.long((y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                               np.int32(16))),
                                                                        np.int32(16)))),
                                            np.long(num_arrays_4)),
                                           (np.long(np.int32(16)),
                                            np.long(np.int32(16)),
                                            np.long(np.int32(1))))
                if synchronous:
                  self.queue.finish()
    return ()
  def futhark_map_transpose_opencl_i32(self, destmem_0, destoffset_1, srcmem_2,
                                       srcoffset_3, num_arrays_4, x_elems_5,
                                       y_elems_6, in_elems_7, out_elems_8):
    if (((num_arrays_4 * x_elems_5) * y_elems_6) == np.int32(0)):
      pass
    else:
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(4))))
        if synchronous:
          self.queue.finish()
      else:
        if (sle32(x_elems_5, squot32(np.int32(16),
                                     np.int32(2))) and slt32(np.int32(16),
                                                             y_elems_6)):
          muly_9 = squot32(np.int32(16), x_elems_5)
          new_height_10 = squot32(((y_elems_6 + muly_9) - np.int32(1)), muly_9)
          if ((((1 * (x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                np.int32(16))),
                                         np.int32(16)))) * (new_height_10 + srem32((np.int32(16) - srem32(new_height_10,
                                                                                                          np.int32(16))),
                                                                                   np.int32(16)))) * num_arrays_4) != 0):
            self.fut_kernel_map_transpose_lowwidth_i32_var.set_args(destmem_0,
                                                                    np.int32(destoffset_1),
                                                                    srcmem_2,
                                                                    np.int32(srcoffset_3),
                                                                    np.int32(x_elems_5),
                                                                    np.int32(y_elems_6),
                                                                    np.int32(in_elems_7),
                                                                    np.int32(out_elems_8),
                                                                    np.int32(muly_9),
                                                                    cl.LocalMemory(np.long((np.int32(272) * np.int32(4)))))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.fut_kernel_map_transpose_lowwidth_i32_var,
                                       (np.long((x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                                           np.int32(16))),
                                                                    np.int32(16)))),
                                        np.long((new_height_10 + srem32((np.int32(16) - srem32(new_height_10,
                                                                                               np.int32(16))),
                                                                        np.int32(16)))),
                                        np.long(num_arrays_4)),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              self.queue.finish()
        else:
          if (sle32(y_elems_6, squot32(np.int32(16),
                                       np.int32(2))) and slt32(np.int32(16),
                                                               x_elems_5)):
            mulx_11 = squot32(np.int32(16), y_elems_6)
            new_width_12 = squot32(((x_elems_5 + mulx_11) - np.int32(1)),
                                   mulx_11)
            if ((((1 * (new_width_12 + srem32((np.int32(16) - srem32(new_width_12,
                                                                     np.int32(16))),
                                              np.int32(16)))) * (y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                                           np.int32(16))),
                                                                                    np.int32(16)))) * num_arrays_4) != 0):
              self.fut_kernel_map_transpose_lowheight_i32_var.set_args(destmem_0,
                                                                       np.int32(destoffset_1),
                                                                       srcmem_2,
                                                                       np.int32(srcoffset_3),
                                                                       np.int32(x_elems_5),
                                                                       np.int32(y_elems_6),
                                                                       np.int32(in_elems_7),
                                                                       np.int32(out_elems_8),
                                                                       np.int32(mulx_11),
                                                                       cl.LocalMemory(np.long((np.int32(272) * np.int32(4)))))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.fut_kernel_map_transpose_lowheight_i32_var,
                                         (np.long((new_width_12 + srem32((np.int32(16) - srem32(new_width_12,
                                                                                                np.int32(16))),
                                                                         np.int32(16)))),
                                          np.long((y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                             np.int32(16))),
                                                                      np.int32(16)))),
                                          np.long(num_arrays_4)),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                self.queue.finish()
          else:
            if (sle32(x_elems_5, squot32(np.int32(16),
                                         np.int32(2))) and sle32(y_elems_6,
                                                                 squot32(np.int32(16),
                                                                         np.int32(2)))):
              if ((1 * (((num_arrays_4 * x_elems_5) * y_elems_6) + srem32((np.int32(256) - srem32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                                                                  np.int32(256))),
                                                                          np.int32(256)))) != 0):
                self.fut_kernel_map_transpose_small_i32_var.set_args(destmem_0,
                                                                     np.int32(destoffset_1),
                                                                     srcmem_2,
                                                                     np.int32(srcoffset_3),
                                                                     np.int32(num_arrays_4),
                                                                     np.int32(x_elems_5),
                                                                     np.int32(y_elems_6),
                                                                     np.int32(in_elems_7),
                                                                     np.int32(out_elems_8))
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.fut_kernel_map_transpose_small_i32_var,
                                           (np.long((((num_arrays_4 * x_elems_5) * y_elems_6) + srem32((np.int32(256) - srem32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                                                                                               np.int32(256))),
                                                                                                       np.int32(256)))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  self.queue.finish()
            else:
              if ((((1 * (x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                    np.int32(16))),
                                             np.int32(16)))) * (y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                                          np.int32(16))),
                                                                                   np.int32(16)))) * num_arrays_4) != 0):
                self.fut_kernel_map_transpose_i32_var.set_args(destmem_0,
                                                               np.int32(destoffset_1),
                                                               srcmem_2,
                                                               np.int32(srcoffset_3),
                                                               np.int32(x_elems_5),
                                                               np.int32(y_elems_6),
                                                               np.int32(in_elems_7),
                                                               np.int32(out_elems_8),
                                                               cl.LocalMemory(np.long((np.int32(272) * np.int32(4)))))
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.fut_kernel_map_transpose_i32_var,
                                           (np.long((x_elems_5 + srem32((np.int32(16) - srem32(x_elems_5,
                                                                                               np.int32(16))),
                                                                        np.int32(16)))),
                                            np.long((y_elems_6 + srem32((np.int32(16) - srem32(y_elems_6,
                                                                                               np.int32(16))),
                                                                        np.int32(16)))),
                                            np.long(num_arrays_4)),
                                           (np.long(np.int32(16)),
                                            np.long(np.int32(16)),
                                            np.long(np.int32(1))))
                if synchronous:
                  self.queue.finish()
    return ()
  def futhark_main(self, angles_mem_sizze_10703, angles_mem_10704,
                   rays_mem_sizze_10705, rays_mem_10706, voxels_mem_sizze_10707,
                   voxels_mem_10708, sizze_9810, sizze_9811, sizze_9812,
                   stepsizze_9816):
    res_9817 = sitofp_i32_f32(sizze_9812)
    res_9818 = futhark_sqrt32(res_9817)
    res_9819 = fptosi_f32_i32(res_9818)
    res_9820 = sdiv32(res_9819, np.int32(2))
    res_9821 = sitofp_i32_f32(res_9820)
    group_sizze_10119 = self.sizes["group_size_10118"]
    y_10120 = (group_sizze_10119 - np.int32(1))
    x_10121 = (sizze_9810 + y_10120)
    num_groups_10122 = squot32(x_10121, group_sizze_10119)
    num_threads_10123 = (group_sizze_10119 * num_groups_10122)
    binop_x_10710 = sext_i32_i64(sizze_9810)
    bytes_10709 = (np.int64(4) * binop_x_10710)
    mem_10711 = opencl_alloc(self, bytes_10709, "mem_10711")
    mem_10714 = opencl_alloc(self, bytes_10709, "mem_10714")
    if ((1 * (num_groups_10122 * group_sizze_10119)) != 0):
      self.map_kernel_10124_var.set_args(np.int32(sizze_9810), angles_mem_10704,
                                         mem_10711, mem_10714)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10124_var,
                                 (np.long((num_groups_10122 * group_sizze_10119)),),
                                 (np.long(group_sizze_10119),))
      if synchronous:
        self.queue.finish()
    nesting_sizze_10105 = (sizze_9810 * sizze_9811)
    group_sizze_10107 = self.sizes["group_size_10106"]
    y_10108 = (group_sizze_10107 - np.int32(1))
    x_10109 = (nesting_sizze_10105 + y_10108)
    num_groups_10110 = squot32(x_10109, group_sizze_10107)
    num_threads_10111 = (group_sizze_10107 * num_groups_10110)
    binop_x_10717 = sext_i32_i64(nesting_sizze_10105)
    bytes_10715 = (np.int64(4) * binop_x_10717)
    mem_10718 = opencl_alloc(self, bytes_10715, "mem_10718")
    if ((1 * (num_groups_10110 * group_sizze_10107)) != 0):
      self.map_kernel_10112_var.set_args(np.int32(sizze_9810),
                                         np.int32(sizze_9811), mem_10711,
                                         mem_10718)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10112_var,
                                 (np.long((num_groups_10110 * group_sizze_10107)),),
                                 (np.long(group_sizze_10107),))
      if synchronous:
        self.queue.finish()
    mem_10711 = None
    group_sizze_10091 = self.sizes["group_size_10090"]
    y_10092 = (group_sizze_10091 - np.int32(1))
    x_10093 = (y_10092 + nesting_sizze_10105)
    num_groups_10094 = squot32(x_10093, group_sizze_10091)
    num_threads_10095 = (group_sizze_10091 * num_groups_10094)
    mem_10722 = opencl_alloc(self, bytes_10715, "mem_10722")
    if ((1 * (num_groups_10094 * group_sizze_10091)) != 0):
      self.map_kernel_10096_var.set_args(np.int32(sizze_9810),
                                         np.int32(sizze_9811), mem_10714,
                                         mem_10722)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10096_var,
                                 (np.long((num_groups_10094 * group_sizze_10091)),),
                                 (np.long(group_sizze_10091),))
      if synchronous:
        self.queue.finish()
    mem_10714 = None
    res_9834 = (np.float32(0.0) - res_9821)
    group_sizze_10418 = self.sizes["group_size_10417"]
    y_10419 = (group_sizze_10418 - np.int32(1))
    x_10420 = (nesting_sizze_10105 + y_10419)
    num_groups_10421 = squot32(x_10420, group_sizze_10418)
    num_threads_10422 = (group_sizze_10418 * num_groups_10421)
    mem_10732 = opencl_alloc(self, binop_x_10717, "mem_10732")
    convop_x_10734 = (np.int32(4) * nesting_sizze_10105)
    binop_x_10735 = sext_i32_i64(convop_x_10734)
    bytes_10733 = (np.int64(4) * binop_x_10735)
    mem_10736 = opencl_alloc(self, bytes_10733, "mem_10736")
    mem_10740 = opencl_alloc(self, bytes_10733, "mem_10740")
    mem_10743 = opencl_alloc(self, binop_x_10735, "mem_10743")
    num_threads64_10879 = sext_i32_i64(num_threads_10422)
    total_sizze_10880 = (np.int64(16) * num_threads64_10879)
    mem_10725 = opencl_alloc(self, total_sizze_10880, "mem_10725")
    total_sizze_10881 = (np.int64(16) * num_threads64_10879)
    mem_10728 = opencl_alloc(self, total_sizze_10881, "mem_10728")
    total_sizze_10882 = (np.int64(4) * num_threads64_10879)
    mem_10730 = opencl_alloc(self, total_sizze_10882, "mem_10730")
    if ((1 * (num_groups_10421 * group_sizze_10418)) != 0):
      self.map_kernel_10423_var.set_args(np.int32(sizze_9811),
                                         np.float32(res_9821),
                                         np.float32(res_9834),
                                         np.int32(nesting_sizze_10105),
                                         rays_mem_10706, mem_10718, mem_10722,
                                         mem_10725, mem_10728, mem_10730,
                                         mem_10732, mem_10736, mem_10740,
                                         mem_10743)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10423_var,
                                 (np.long((num_groups_10421 * group_sizze_10418)),),
                                 (np.long(group_sizze_10418),))
      if synchronous:
        self.queue.finish()
    mem_10722 = None
    mem_10725 = None
    mem_10728 = None
    mem_10730 = None
    group_sizze_10258 = self.sizes["group_size_10257"]
    max_num_groups_10260 = self.sizes["max_num_groups_10259"]
    y_10261 = (group_sizze_10258 - np.int32(1))
    x_10262 = (y_10261 + convop_x_10734)
    w_div_group_sizze_10263 = squot32(x_10262, group_sizze_10258)
    num_groups_maybe_zzero_10264 = smin32(max_num_groups_10260,
                                          w_div_group_sizze_10263)
    num_groups_10265 = smax32(np.int32(1), num_groups_maybe_zzero_10264)
    num_threads_10266 = (group_sizze_10258 * num_groups_10265)
    mem_10745 = opencl_alloc(self, binop_x_10735, "mem_10745")
    mem_10748 = opencl_alloc(self, bytes_10733, "mem_10748")
    mem_10751 = opencl_alloc(self, bytes_10733, "mem_10751")
    y_10315 = (num_threads_10266 - np.int32(1))
    x_10316 = (y_10315 + convop_x_10734)
    num_iterations_10317 = squot32(x_10316, num_threads_10266)
    y_10321 = (group_sizze_10258 * num_iterations_10317)
    mem_10754 = opencl_alloc(self, binop_x_10735, "mem_10754")
    if (((nesting_sizze_10105 * np.int32(4)) * np.int32(1)) != 0):
      cl.enqueue_copy(self.queue, mem_10754, mem_10743,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(0)),
                      byte_count=np.long(((nesting_sizze_10105 * np.int32(4)) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    mem_10743 = None
    bytes_10769 = sext_i32_i64(num_groups_10265)
    mem_10770 = opencl_alloc(self, bytes_10769, "mem_10770")
    bytes_10771 = (np.int64(4) * bytes_10769)
    mem_10773 = opencl_alloc(self, bytes_10771, "mem_10773")
    bytes_10757 = sext_i32_i64(group_sizze_10258)
    bytes_10759 = (np.int64(4) * bytes_10757)
    if ((1 * (num_groups_10265 * group_sizze_10258)) != 0):
      self.scan1_kernel_10312_var.set_args(cl.LocalMemory(np.long(bytes_10757)),
                                           cl.LocalMemory(np.long(bytes_10759)),
                                           np.int32(num_iterations_10317),
                                           np.int32(y_10321),
                                           np.int32(convop_x_10734), mem_10745,
                                           mem_10748, mem_10751, mem_10754,
                                           mem_10770, mem_10773)
      cl.enqueue_nd_range_kernel(self.queue, self.scan1_kernel_10312_var,
                                 (np.long((num_groups_10265 * group_sizze_10258)),),
                                 (np.long(group_sizze_10258),))
      if synchronous:
        self.queue.finish()
    mem_10754 = None
    mem_10758 = None
    mem_10761 = None
    mem_10780 = opencl_alloc(self, bytes_10769, "mem_10780")
    mem_10783 = opencl_alloc(self, bytes_10771, "mem_10783")
    if ((1 * num_groups_10265) != 0):
      self.scan2_kernel_10370_var.set_args(cl.LocalMemory(np.long(bytes_10769)),
                                           cl.LocalMemory(np.long(bytes_10771)),
                                           np.int32(num_groups_10265),
                                           mem_10770, mem_10773, mem_10780,
                                           mem_10783)
      cl.enqueue_nd_range_kernel(self.queue, self.scan2_kernel_10370_var,
                                 (np.long(num_groups_10265),),
                                 (np.long(num_groups_10265),))
      if synchronous:
        self.queue.finish()
    mem_10770 = None
    mem_10773 = None
    mem_10775 = None
    mem_10778 = None
    group_sizze_10408 = self.sizes["group_size_10407"]
    y_10409 = (group_sizze_10408 - np.int32(1))
    x_10410 = (y_10409 + convop_x_10734)
    num_groups_10411 = squot32(x_10410, group_sizze_10408)
    num_threads_10412 = (group_sizze_10408 * num_groups_10411)
    mem_10785 = opencl_alloc(self, binop_x_10735, "mem_10785")
    mem_10788 = opencl_alloc(self, bytes_10733, "mem_10788")
    if ((1 * (num_groups_10411 * group_sizze_10408)) != 0):
      self.map_kernel_10413_var.set_args(np.int32(y_10321),
                                         np.int32(convop_x_10734), mem_10745,
                                         mem_10748, mem_10780, mem_10783,
                                         mem_10785, mem_10788)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10413_var,
                                 (np.long((num_groups_10411 * group_sizze_10408)),),
                                 (np.long(group_sizze_10408),))
      if synchronous:
        self.queue.finish()
    mem_10745 = None
    mem_10748 = None
    mem_10780 = None
    mem_10783 = None
    mem_10785 = None
    group_sizze_10161 = self.sizes["group_size_10160"]
    y_10162 = (group_sizze_10161 - np.int32(1))
    x_10163 = (nesting_sizze_10105 + y_10162)
    num_groups_10164 = squot32(x_10163, group_sizze_10161)
    num_threads_10165 = (group_sizze_10161 * num_groups_10164)
    mem_10803 = opencl_alloc(self, bytes_10715, "mem_10803")
    mem_10806 = opencl_alloc(self, bytes_10715, "mem_10806")
    mem_10809 = opencl_alloc(self, bytes_10715, "mem_10809")
    mem_10812 = opencl_alloc(self, bytes_10715, "mem_10812")
    space_sizze_10891 = nesting_sizze_10105
    num_threads_10892 = sext_i32_i64(num_threads_10165)
    group_sizze_10900 = self.sizes["group_size_10899"]
    max_num_groups_10902 = self.sizes["max_num_groups_10901"]
    y_10903 = (group_sizze_10900 - np.int32(1))
    x_10904 = (nesting_sizze_10105 + y_10903)
    w_div_group_sizze_10905 = squot32(x_10904, group_sizze_10900)
    num_groups_maybe_zzero_10906 = smin32(max_num_groups_10902,
                                          w_div_group_sizze_10905)
    num_groups_10907 = smax32(np.int32(1), num_groups_maybe_zzero_10906)
    num_threads_10908 = (group_sizze_10900 * num_groups_10907)
    y_10909 = (num_threads_10908 - np.int32(1))
    x_10910 = (nesting_sizze_10105 + y_10909)
    per_thread_elements_10911 = squot32(x_10910, num_threads_10908)
    binop_x_10960 = sext_i32_i64(num_groups_10907)
    bytes_10959 = (np.int64(8) * binop_x_10960)
    mem_10961 = opencl_alloc(self, bytes_10959, "mem_10961")
    binop_x_10957 = sext_i32_i64(group_sizze_10900)
    bytes_10956 = (np.int64(8) * binop_x_10957)
    if ((1 * (num_groups_10907 * group_sizze_10900)) != 0):
      self.chunked_reduce_kernel_10916_var.set_args(cl.LocalMemory(np.long(bytes_10956)),
                                                    np.int32(nesting_sizze_10105),
                                                    mem_10788,
                                                    np.int32(num_threads_10908),
                                                    np.int32(per_thread_elements_10911),
                                                    mem_10961)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.chunked_reduce_kernel_10916_var,
                                 (np.long((num_groups_10907 * group_sizze_10900)),),
                                 (np.long(group_sizze_10900),))
      if synchronous:
        self.queue.finish()
    mem_10958 = None
    mem_10967 = opencl_alloc(self, np.int64(8), "mem_10967")
    binop_x_10963 = sext_i32_i64(max_num_groups_10902)
    bytes_10962 = (np.int64(8) * binop_x_10963)
    if ((1 * max_num_groups_10902) != 0):
      self.reduce_kernel_10946_var.set_args(cl.LocalMemory(np.long(bytes_10962)),
                                            np.int32(num_groups_10907),
                                            mem_10961, mem_10967)
      cl.enqueue_nd_range_kernel(self.queue, self.reduce_kernel_10946_var,
                                 (np.long(max_num_groups_10902),),
                                 (np.long(max_num_groups_10902),))
      if synchronous:
        self.queue.finish()
    mem_10961 = None
    mem_10964 = None
    read_res_11082 = np.empty(1, dtype=ct.c_int64)
    cl.enqueue_copy(self.queue, read_res_11082, mem_10967,
                    device_offset=np.long(np.int32(0)), is_blocking=True)
    max_per_thread_10893 = read_res_11082[0]
    mem_10967 = None
    sizze_sum_10955 = (num_threads_10892 * max_per_thread_10893)
    mem_10794 = opencl_alloc(self, sizze_sum_10955, "mem_10794")
    mem_10791 = opencl_alloc(self, sizze_sum_10955, "mem_10791")
    if ((1 * (num_groups_10164 * group_sizze_10161)) != 0):
      self.map_kernel_10166_var.set_args(np.int32(sizze_9811),
                                         np.float32(res_9821),
                                         np.float32(res_9834),
                                         np.int32(nesting_sizze_10105),
                                         np.int32(num_threads_10165),
                                         rays_mem_10706, mem_10718, mem_10732,
                                         mem_10736, mem_10740, mem_10751,
                                         mem_10788, mem_10791, mem_10794,
                                         mem_10803, mem_10806, mem_10809,
                                         mem_10812)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10166_var,
                                 (np.long((num_groups_10164 * group_sizze_10161)),),
                                 (np.long(group_sizze_10161),))
      if synchronous:
        self.queue.finish()
    mem_10718 = None
    mem_10732 = None
    mem_10736 = None
    mem_10740 = None
    mem_10751 = None
    mem_10788 = None
    mem_10791 = None
    mem_10794 = None
    res_9912 = sdiv32(nesting_sizze_10105, stepsizze_9816)
    mem_10815 = opencl_alloc(self, np.int64(4), "mem_10815")
    group_sizze_11061 = self.sizes["group_size_11061"]
    num_groups_11062 = squot32(((np.int32(1) + sext_i32_i32(group_sizze_11061)) - np.int32(1)),
                               sext_i32_i32(group_sizze_11061))
    if ((1 * (num_groups_11062 * group_sizze_11061)) != 0):
      self.kernel_replicate_9913_var.set_args(mem_10815)
      cl.enqueue_nd_range_kernel(self.queue, self.kernel_replicate_9913_var,
                                 (np.long((num_groups_11062 * group_sizze_11061)),),
                                 (np.long(group_sizze_11061),))
      if synchronous:
        self.queue.finish()
    loop_cond_9914 = slt32(np.int32(0), res_9912)
    range_start_9915 = (np.int32(0) - res_9820)
    range_end_9916 = (res_9820 - np.int32(1))
    bounds_invalid_upwards_9917 = slt32(range_end_9916, range_start_9915)
    distance_upwards_exclusive_9918 = (range_end_9916 - range_start_9915)
    distance_9919 = (np.int32(1) + distance_upwards_exclusive_9918)
    if bounds_invalid_upwards_9917:
      num_elems_9920 = np.int32(0)
    else:
      num_elems_9920 = distance_9919
    flat_dim_9921 = (np.int32(2) * num_elems_9920)
    y_9923 = pow32(res_9819, np.int32(2))
    group_sizze_10604 = self.sizes["group_size_10603"]
    y_10605 = (group_sizze_10604 - np.int32(1))
    group_sizze_10515 = self.sizes["group_size_10514"]
    y_10516 = (group_sizze_10515 - np.int32(1))
    group_sizze_10473 = self.sizes["group_size_10472"]
    y_10474 = (group_sizze_10473 - np.int32(1))
    sizze_9928 = np.int32(1)
    output_mem_sizze_10816 = np.int64(4)
    output_mem_10817 = mem_10815
    loop_while_9929 = loop_cond_9914
    run_9931 = np.int32(0)
    while loop_while_9929:
      x_9932 = (np.int32(1) + run_9931)
      x_9933 = (stepsizze_9816 * x_9932)
      cond_9934 = sle32(nesting_sizze_10105, x_9933)
      if cond_9934:
        y_9936 = (stepsizze_9816 * run_9931)
        res_9937 = (nesting_sizze_10105 - y_9936)
        res_9935 = res_9937
      else:
        res_9935 = stepsizze_9816
      i_9938 = (stepsizze_9816 * run_9931)
      j_9939 = (res_9935 + i_9938)
      x_9940 = abs(res_9935)
      empty_slice_9941 = (x_9940 == np.int32(0))
      m_9942 = (x_9940 - np.int32(1))
      i_p_m_t_s_9943 = (i_9938 + m_9942)
      zzero_leq_i_p_m_t_s_9944 = sle32(np.int32(0), i_p_m_t_s_9943)
      i_p_m_t_s_leq_w_9945 = slt32(i_p_m_t_s_9943, nesting_sizze_10105)
      zzero_lte_i_9946 = sle32(np.int32(0), i_9938)
      i_lte_j_9947 = sle32(i_9938, j_9939)
      y_9948 = (i_p_m_t_s_leq_w_9945 and zzero_lte_i_9946)
      y_9949 = (zzero_leq_i_p_m_t_s_9944 and y_9948)
      y_9950 = (i_lte_j_9947 and y_9949)
      forwards_ok_9951 = (zzero_lte_i_9946 and y_9950)
      ok_or_empty_9952 = (empty_slice_9941 or forwards_ok_9951)
      index_certs_9953 = True
      assert ok_or_empty_9952, ("Error at forwardprojection_doubleparallel.fut:32:1-36:70 -> forwardprojection_doubleparallel.fut:36:11-36:70 -> projection_lib.fut:244:67-244:113: %s%d%s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                       i_9938,
                                                                                                                                                                                                       ":",
                                                                                                                                                                                                       j_9939,
                                                                                                                                                                                                       "] out of bounds for array of shape [",
                                                                                                                                                                                                       nesting_sizze_10105,
                                                                                                                                                                                                       "]."))
      x_10606 = (x_9940 + y_10605)
      num_groups_10607 = squot32(x_10606, group_sizze_10604)
      num_threads_10608 = (group_sizze_10604 * num_groups_10607)
      bytes_10818 = sext_i32_i64(x_9940)
      mem_10819 = opencl_alloc(self, bytes_10818, "mem_10819")
      bytes_10820 = (np.int64(4) * bytes_10818)
      mem_10822 = opencl_alloc(self, bytes_10820, "mem_10822")
      mem_10825 = opencl_alloc(self, bytes_10820, "mem_10825")
      mem_10828 = opencl_alloc(self, bytes_10820, "mem_10828")
      mem_10831 = opencl_alloc(self, bytes_10820, "mem_10831")
      if ((1 * (num_groups_10607 * group_sizze_10604)) != 0):
        self.map_kernel_10609_var.set_args(np.int32(i_9938), np.int32(x_9940),
                                           mem_10803, mem_10806, mem_10809,
                                           mem_10812, mem_10819, mem_10822,
                                           mem_10825, mem_10828, mem_10831)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10609_var,
                                   (np.long((num_groups_10607 * group_sizze_10604)),),
                                   (np.long(group_sizze_10604),))
        if synchronous:
          self.queue.finish()
      nesting_sizze_10513 = (num_elems_9920 * x_9940)
      x_10517 = (nesting_sizze_10513 + y_10516)
      num_groups_10518 = squot32(x_10517, group_sizze_10515)
      num_threads_10519 = (group_sizze_10515 * num_groups_10518)
      binop_x_10839 = (np.int32(2) * x_9940)
      convop_x_10840 = (num_elems_9920 * binop_x_10839)
      binop_x_10841 = sext_i32_i64(convop_x_10840)
      bytes_10838 = (np.int64(4) * binop_x_10841)
      mem_10842 = opencl_alloc(self, bytes_10838, "mem_10842")
      mem_10847 = opencl_alloc(self, bytes_10838, "mem_10847")
      num_threads64_10973 = sext_i32_i64(num_threads_10519)
      total_sizze_10974 = (np.int64(8) * num_threads64_10973)
      mem_10834 = opencl_alloc(self, total_sizze_10974, "mem_10834")
      total_sizze_10975 = (np.int64(8) * num_threads64_10973)
      mem_10837 = opencl_alloc(self, total_sizze_10975, "mem_10837")
      if ((1 * (num_groups_10518 * group_sizze_10515)) != 0):
        self.map_kernel_10520_var.set_args(np.int32(res_9819),
                                           np.int32(res_9820),
                                           np.float32(res_9821),
                                           np.int32(range_start_9915),
                                           np.int32(num_elems_9920),
                                           np.int32(y_9923), np.int32(x_9940),
                                           mem_10819, mem_10822, mem_10825,
                                           mem_10828, mem_10831, mem_10834,
                                           mem_10837, mem_10842, mem_10847)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10520_var,
                                   (np.long((num_groups_10518 * group_sizze_10515)),),
                                   (np.long(group_sizze_10515),))
        if synchronous:
          self.queue.finish()
      mem_10819 = None
      mem_10822 = None
      mem_10825 = None
      mem_10828 = None
      mem_10831 = None
      mem_10834 = None
      mem_10837 = None
      x_10475 = (x_9940 + y_10474)
      num_groups_10476 = squot32(x_10475, group_sizze_10473)
      num_threads_10477 = (group_sizze_10473 * num_groups_10476)
      convop_x_10849 = (flat_dim_9921 * x_9940)
      binop_x_10850 = sext_i32_i64(convop_x_10849)
      bytes_10848 = (np.int64(4) * binop_x_10850)
      mem_10851 = opencl_alloc(self, bytes_10848, "mem_10851")
      self.futhark_map_transpose_opencl_f32(mem_10851, np.int32(0), mem_10842,
                                            np.int32(0), np.int32(1),
                                            (x_9940 * num_elems_9920),
                                            np.int32(2),
                                            ((x_9940 * num_elems_9920) * np.int32(2)),
                                            (x_9940 * flat_dim_9921))
      mem_10842 = None
      mem_10855 = opencl_alloc(self, bytes_10848, "mem_10855")
      self.futhark_map_transpose_opencl_f32(mem_10855, np.int32(0), mem_10851,
                                            np.int32(0), np.int32(1),
                                            flat_dim_9921, x_9940,
                                            (x_9940 * flat_dim_9921),
                                            (x_9940 * flat_dim_9921))
      mem_10851 = None
      mem_10859 = opencl_alloc(self, bytes_10848, "mem_10859")
      self.futhark_map_transpose_opencl_i32(mem_10859, np.int32(0), mem_10847,
                                            np.int32(0), np.int32(1),
                                            (x_9940 * num_elems_9920),
                                            np.int32(2),
                                            ((x_9940 * num_elems_9920) * np.int32(2)),
                                            (x_9940 * flat_dim_9921))
      mem_10847 = None
      mem_10863 = opencl_alloc(self, bytes_10848, "mem_10863")
      self.futhark_map_transpose_opencl_i32(mem_10863, np.int32(0), mem_10859,
                                            np.int32(0), np.int32(1),
                                            flat_dim_9921, x_9940,
                                            (x_9940 * flat_dim_9921),
                                            (x_9940 * flat_dim_9921))
      mem_10859 = None
      mem_10866 = opencl_alloc(self, bytes_10820, "mem_10866")
      if ((1 * (num_groups_10476 * group_sizze_10473)) != 0):
        self.map_kernel_10478_var.set_args(np.int32(flat_dim_9921),
                                           np.int32(x_9940), voxels_mem_10708,
                                           mem_10855, mem_10863, mem_10866)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10478_var,
                                   (np.long((num_groups_10476 * group_sizze_10473)),),
                                   (np.long(group_sizze_10473),))
        if synchronous:
          self.queue.finish()
      mem_10855 = None
      mem_10863 = None
      conc_tmp_10069 = (sizze_9928 + x_9940)
      binop_x_10868 = sext_i32_i64(conc_tmp_10069)
      bytes_10867 = (np.int64(4) * binop_x_10868)
      mem_10869 = opencl_alloc(self, bytes_10867, "mem_10869")
      tmp_offs_11081 = np.int32(0)
      if ((sizze_9928 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10869, output_mem_10817,
                        dest_offset=np.long((tmp_offs_11081 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sizze_9928 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_11081 = (tmp_offs_11081 + sizze_9928)
      if ((x_9940 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10869, mem_10866,
                        dest_offset=np.long((tmp_offs_11081 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((x_9940 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_11081 = (tmp_offs_11081 + x_9940)
      mem_10866 = None
      loop_cond_10071 = slt32(x_9932, res_9912)
      sizze_tmp_11063 = conc_tmp_10069
      output_mem_sizze_tmp_11064 = bytes_10867
      output_mem_tmp_11065 = mem_10869
      loop_while_tmp_11066 = loop_cond_10071
      run_tmp_11068 = x_9932
      sizze_9928 = sizze_tmp_11063
      output_mem_sizze_10816 = output_mem_sizze_tmp_11064
      output_mem_10817 = output_mem_tmp_11065
      loop_while_9929 = loop_while_tmp_11066
      run_9931 = run_tmp_11068
    sizze_9924 = sizze_9928
    res_mem_sizze_10870 = output_mem_sizze_10816
    res_mem_10871 = output_mem_10817
    res_9925 = loop_while_9929
    res_9927 = run_9931
    mem_10803 = None
    mem_10806 = None
    mem_10809 = None
    mem_10812 = None
    mem_10815 = None
    j_m_i_10072 = (sizze_9924 - np.int32(1))
    x_10073 = abs(j_m_i_10072)
    empty_slice_10074 = (x_10073 == np.int32(0))
    m_10075 = (x_10073 - np.int32(1))
    i_p_m_t_s_10076 = (np.int32(1) + m_10075)
    zzero_leq_i_p_m_t_s_10077 = sle32(np.int32(0), i_p_m_t_s_10076)
    i_p_m_t_s_leq_w_10078 = slt32(i_p_m_t_s_10076, sizze_9924)
    i_lte_j_10079 = sle32(np.int32(1), sizze_9924)
    y_10080 = (zzero_leq_i_p_m_t_s_10077 and i_p_m_t_s_leq_w_10078)
    y_10081 = (i_lte_j_10079 and y_10080)
    ok_or_empty_10082 = (empty_slice_10074 or y_10081)
    index_certs_10083 = True
    assert ok_or_empty_10082, ("Error at forwardprojection_doubleparallel.fut:32:1-36:70 -> forwardprojection_doubleparallel.fut:36:11-36:70 -> projection_lib.fut:247:20-247:31 -> /futlib/array.fut:21:29-21:33: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                                                  np.int32(1),
                                                                                                                                                                                                                                  "] out of bounds for array of shape [",
                                                                                                                                                                                                                                  sizze_9924,
                                                                                                                                                                                                                                  "]."))
    binop_x_10873 = sext_i32_i64(x_10073)
    bytes_10872 = (np.int64(4) * binop_x_10873)
    mem_10874 = opencl_alloc(self, bytes_10872, "mem_10874")
    if ((x_10073 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_10874, res_mem_10871,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(4)),
                      byte_count=np.long((x_10073 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    res_mem_10871 = None
    out_arrsizze_10979 = x_10073
    out_memsizze_10978 = bytes_10872
    out_mem_10977 = mem_10874
    return (out_memsizze_10978, out_mem_10977, out_arrsizze_10979)
  def main(self, angles_mem_10704_ext, rays_mem_10706_ext, voxels_mem_10708_ext,
           stepsizze_9816_ext):
    try:
      assert ((type(angles_mem_10704_ext) in [np.ndarray,
                                              cl.array.Array]) and (angles_mem_10704_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9810 = np.int32(angles_mem_10704_ext.shape[0])
      angles_mem_sizze_10703 = np.int64(angles_mem_10704_ext.nbytes)
      if (type(angles_mem_10704_ext) == cl.array.Array):
        angles_mem_10704 = angles_mem_10704_ext.data
      else:
        angles_mem_10704 = opencl_alloc(self, angles_mem_sizze_10703,
                                        "angles_mem_10704")
        if (angles_mem_sizze_10703 != 0):
          cl.enqueue_copy(self.queue, angles_mem_10704,
                          normaliseArray(angles_mem_10704_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(angles_mem_10704_ext),
                                                                                                                            angles_mem_10704_ext))
    try:
      assert ((type(rays_mem_10706_ext) in [np.ndarray,
                                            cl.array.Array]) and (rays_mem_10706_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9811 = np.int32(rays_mem_10706_ext.shape[0])
      rays_mem_sizze_10705 = np.int64(rays_mem_10706_ext.nbytes)
      if (type(rays_mem_10706_ext) == cl.array.Array):
        rays_mem_10706 = rays_mem_10706_ext.data
      else:
        rays_mem_10706 = opencl_alloc(self, rays_mem_sizze_10705,
                                      "rays_mem_10706")
        if (rays_mem_sizze_10705 != 0):
          cl.enqueue_copy(self.queue, rays_mem_10706,
                          normaliseArray(rays_mem_10706_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(rays_mem_10706_ext),
                                                                                                                            rays_mem_10706_ext))
    try:
      assert ((type(voxels_mem_10708_ext) in [np.ndarray,
                                              cl.array.Array]) and (voxels_mem_10708_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9812 = np.int32(voxels_mem_10708_ext.shape[0])
      voxels_mem_sizze_10707 = np.int64(voxels_mem_10708_ext.nbytes)
      if (type(voxels_mem_10708_ext) == cl.array.Array):
        voxels_mem_10708 = voxels_mem_10708_ext.data
      else:
        voxels_mem_10708 = opencl_alloc(self, voxels_mem_sizze_10707,
                                        "voxels_mem_10708")
        if (voxels_mem_sizze_10707 != 0):
          cl.enqueue_copy(self.queue, voxels_mem_10708,
                          normaliseArray(voxels_mem_10708_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(voxels_mem_10708_ext),
                                                                                                                            voxels_mem_10708_ext))
    try:
      stepsizze_9816 = np.int32(ct.c_int32(stepsizze_9816_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(stepsizze_9816_ext),
                                                                                                                            stepsizze_9816_ext))
    (out_memsizze_10978, out_mem_10977,
     out_arrsizze_10979) = self.futhark_main(angles_mem_sizze_10703,
                                             angles_mem_10704,
                                             rays_mem_sizze_10705,
                                             rays_mem_10706,
                                             voxels_mem_sizze_10707,
                                             voxels_mem_10708, sizze_9810,
                                             sizze_9811, sizze_9812,
                                             stepsizze_9816)
    return cl.array.Array(self.queue, (out_arrsizze_10979,), ct.c_float,
                          data=out_mem_10977)