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
#define group_sizze_8501 (group_size_8500)
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
__kernel void map_kernel_8410(int32_t sizze_8234, int32_t sizze_8235, __global
                              unsigned char *mem_8684, __global
                              unsigned char *mem_8692)
{
    int32_t wave_sizze_8759;
    int32_t group_sizze_8760;
    bool thread_active_8761;
    int32_t gtid_8401;
    int32_t gtid_8402;
    int32_t global_tid_8410;
    int32_t local_tid_8411;
    int32_t group_id_8412;
    
    global_tid_8410 = get_global_id(0);
    local_tid_8411 = get_local_id(0);
    group_sizze_8760 = get_local_size(0);
    wave_sizze_8759 = LOCKSTEP_WIDTH;
    group_id_8412 = get_group_id(0);
    gtid_8401 = squot32(global_tid_8410, sizze_8235);
    gtid_8402 = global_tid_8410 - squot32(global_tid_8410, sizze_8235) *
        sizze_8235;
    thread_active_8761 = slt32(gtid_8401, sizze_8234) && slt32(gtid_8402,
                                                               sizze_8235);
    
    float res_8413;
    
    if (thread_active_8761) {
        res_8413 = *(__global float *) &mem_8684[gtid_8401 * 4];
    }
    if (thread_active_8761) {
        *(__global float *) &mem_8692[(gtid_8401 * sizze_8235 + gtid_8402) *
                                      4] = res_8413;
    }
}
__kernel void map_kernel_8426(int32_t sizze_8234, int32_t sizze_8235, __global
                              unsigned char *mem_8681, __global
                              unsigned char *mem_8688)
{
    int32_t wave_sizze_8756;
    int32_t group_sizze_8757;
    bool thread_active_8758;
    int32_t gtid_8417;
    int32_t gtid_8418;
    int32_t global_tid_8426;
    int32_t local_tid_8427;
    int32_t group_id_8428;
    
    global_tid_8426 = get_global_id(0);
    local_tid_8427 = get_local_id(0);
    group_sizze_8757 = get_local_size(0);
    wave_sizze_8756 = LOCKSTEP_WIDTH;
    group_id_8428 = get_group_id(0);
    gtid_8417 = squot32(global_tid_8426, sizze_8235);
    gtid_8418 = global_tid_8426 - squot32(global_tid_8426, sizze_8235) *
        sizze_8235;
    thread_active_8758 = slt32(gtid_8417, sizze_8234) && slt32(gtid_8418,
                                                               sizze_8235);
    
    float res_8429;
    
    if (thread_active_8758) {
        res_8429 = *(__global float *) &mem_8681[gtid_8417 * 4];
    }
    if (thread_active_8758) {
        *(__global float *) &mem_8688[(gtid_8417 * sizze_8235 + gtid_8418) *
                                      4] = res_8429;
    }
}
__kernel void map_kernel_8438(int32_t sizze_8234, __global
                              unsigned char *angles_mem_8674, __global
                              unsigned char *mem_8681, __global
                              unsigned char *mem_8684)
{
    int32_t wave_sizze_8753;
    int32_t group_sizze_8754;
    bool thread_active_8755;
    int32_t gtid_8431;
    int32_t global_tid_8438;
    int32_t local_tid_8439;
    int32_t group_id_8440;
    
    global_tid_8438 = get_global_id(0);
    local_tid_8439 = get_local_id(0);
    group_sizze_8754 = get_local_size(0);
    wave_sizze_8753 = LOCKSTEP_WIDTH;
    group_id_8440 = get_group_id(0);
    gtid_8431 = global_tid_8438;
    thread_active_8755 = slt32(gtid_8431, sizze_8234);
    
    float x_8441;
    float res_8442;
    float res_8443;
    
    if (thread_active_8755) {
        x_8441 = *(__global float *) &angles_mem_8674[gtid_8431 * 4];
        res_8442 = futrts_sin32(x_8441);
        res_8443 = futrts_cos32(x_8441);
    }
    if (thread_active_8755) {
        *(__global float *) &mem_8681[gtid_8431 * 4] = res_8442;
    }
    if (thread_active_8755) {
        *(__global float *) &mem_8684[gtid_8431 * 4] = res_8443;
    }
}
__kernel void map_kernel_8464(int32_t flat_dim_8258, int32_t nesting_sizze_8419,
                              __global unsigned char *voxels_mem_8678, __global
                              unsigned char *mem_8730, __global
                              unsigned char *mem_8738, __global
                              unsigned char *mem_8741)
{
    int32_t wave_sizze_8770;
    int32_t group_sizze_8771;
    bool thread_active_8772;
    int32_t gtid_8457;
    int32_t global_tid_8464;
    int32_t local_tid_8465;
    int32_t group_id_8466;
    
    global_tid_8464 = get_global_id(0);
    local_tid_8465 = get_local_id(0);
    group_sizze_8771 = get_local_size(0);
    wave_sizze_8770 = LOCKSTEP_WIDTH;
    group_id_8466 = get_group_id(0);
    gtid_8457 = global_tid_8464;
    thread_active_8772 = slt32(gtid_8457, nesting_sizze_8419);
    
    float res_8469;
    
    if (thread_active_8772) {
        float x_8472 = 0.0F;
        
        for (int32_t chunk_offset_8471 = 0; chunk_offset_8471 < flat_dim_8258;
             chunk_offset_8471++) {
            float x_8481 = *(__global float *) &mem_8730[(chunk_offset_8471 *
                                                          nesting_sizze_8419 +
                                                          gtid_8457) * 4];
            int32_t x_8482 = *(__global
                               int32_t *) &mem_8738[(chunk_offset_8471 *
                                                     nesting_sizze_8419 +
                                                     gtid_8457) * 4];
            bool cond_8484 = x_8482 == -1;
            float res_8485;
            
            if (cond_8484) {
                res_8485 = 0.0F;
            } else {
                float y_8486;
                float res_8487;
                
                y_8486 = *(__global float *) &voxels_mem_8678[x_8482 * 4];
                res_8487 = x_8481 * y_8486;
                res_8485 = res_8487;
            }
            
            float res_8489 = x_8472 + res_8485;
            float x_tmp_8773 = res_8489;
            
            x_8472 = x_tmp_8773;
        }
        res_8469 = x_8472;
    }
    if (thread_active_8772) {
        *(__global float *) &mem_8741[gtid_8457 * 4] = res_8469;
    }
}
__kernel void map_kernel_8506(int32_t sizze_8236, int32_t res_8241,
                              float res_8242, int32_t range_start_8252,
                              int32_t num_elems_8257, int32_t y_8260,
                              int32_t nesting_sizze_8419, __global
                              unsigned char *mem_8694, __global
                              unsigned char *mem_8697, __global
                              unsigned char *mem_8700, __global
                              unsigned char *mem_8703, __global
                              unsigned char *mem_8706, __global
                              unsigned char *mem_8709, __global
                              unsigned char *mem_8712, __global
                              unsigned char *mem_8717, __global
                              unsigned char *mem_8722)
{
    int32_t wave_sizze_8765;
    int32_t group_sizze_8766;
    bool thread_active_8767;
    int32_t gtid_8497;
    int32_t gtid_8498;
    int32_t global_tid_8506;
    int32_t local_tid_8507;
    int32_t group_id_8508;
    
    global_tid_8506 = get_global_id(0);
    local_tid_8507 = get_local_id(0);
    group_sizze_8766 = get_local_size(0);
    wave_sizze_8765 = LOCKSTEP_WIDTH;
    group_id_8508 = get_group_id(0);
    gtid_8497 = squot32(global_tid_8506, num_elems_8257);
    gtid_8498 = global_tid_8506 - squot32(global_tid_8506, num_elems_8257) *
        num_elems_8257;
    thread_active_8767 = slt32(gtid_8497, nesting_sizze_8419) &&
        slt32(gtid_8498, num_elems_8257);
    
    bool res_8509;
    float res_8510;
    float res_8511;
    float res_8512;
    float res_8513;
    int32_t index_primexp_8656;
    float res_8515;
    float y_8516;
    float x_8517;
    float x_8518;
    float res_8519;
    float x_8520;
    float y_8521;
    float x_8522;
    float x_8523;
    float res_8524;
    int32_t res_8525;
    float res_8526;
    bool res_8527;
    float res_8528;
    int32_t res_8535;
    int32_t res_8536;
    float res_8537;
    bool res_8538;
    float res_8539;
    int32_t res_8546;
    int32_t res_8547;
    float res_8548;
    float res_8549;
    float x_8550;
    float res_8551;
    float x_8552;
    float res_8553;
    float res_8554;
    float res_8555;
    int32_t res_8556;
    int32_t res_8557;
    int32_t res_8562;
    bool cond_8567;
    bool res_8568;
    bool x_8569;
    int32_t res_8570;
    float res_8571;
    bool cond_8574;
    bool res_8575;
    bool x_8576;
    float res_8577;
    int32_t res_8578;
    
    if (thread_active_8767) {
        res_8509 = *(__global bool *) &mem_8694[gtid_8497];
        res_8510 = *(__global float *) &mem_8697[gtid_8497 * 4];
        res_8511 = *(__global float *) &mem_8700[gtid_8497 * 4];
        res_8512 = *(__global float *) &mem_8703[gtid_8497 * 4];
        res_8513 = *(__global float *) &mem_8706[gtid_8497 * 4];
        index_primexp_8656 = range_start_8252 + gtid_8498;
        res_8515 = sitofp_i32_f32(index_primexp_8656);
        y_8516 = res_8515 - res_8511;
        x_8517 = res_8510 * y_8516;
        x_8518 = res_8512 + x_8517;
        res_8519 = res_8242 + x_8518;
        x_8520 = 1.0F + res_8515;
        y_8521 = x_8520 - res_8511;
        x_8522 = res_8510 * y_8521;
        x_8523 = res_8512 + x_8522;
        res_8524 = res_8242 + x_8523;
        res_8525 = fptosi_f32_i32(res_8519);
        res_8526 = sitofp_i32_f32(res_8525);
        res_8527 = 0.0F <= res_8519;
        if (res_8527) {
            bool res_8529;
            float res_8530;
            
            res_8529 = res_8526 < res_8519;
            if (res_8529) {
                res_8530 = res_8526;
            } else {
                res_8530 = res_8519;
            }
            res_8528 = res_8530;
        } else {
            bool res_8531;
            float res_8532;
            
            res_8531 = res_8519 < res_8526;
            if (res_8531) {
                int32_t res_8533;
                float res_8534;
                
                res_8533 = res_8525 - 1;
                res_8534 = sitofp_i32_f32(res_8533);
                res_8532 = res_8534;
            } else {
                res_8532 = res_8519;
            }
            res_8528 = res_8532;
        }
        res_8535 = fptosi_f32_i32(res_8528);
        res_8536 = fptosi_f32_i32(res_8524);
        res_8537 = sitofp_i32_f32(res_8536);
        res_8538 = 0.0F <= res_8524;
        if (res_8538) {
            bool res_8540;
            float res_8541;
            
            res_8540 = res_8537 < res_8524;
            if (res_8540) {
                res_8541 = res_8537;
            } else {
                res_8541 = res_8524;
            }
            res_8539 = res_8541;
        } else {
            bool res_8542;
            float res_8543;
            
            res_8542 = res_8524 < res_8537;
            if (res_8542) {
                int32_t res_8544;
                float res_8545;
                
                res_8544 = res_8536 - 1;
                res_8545 = sitofp_i32_f32(res_8544);
                res_8543 = res_8545;
            } else {
                res_8543 = res_8524;
            }
            res_8539 = res_8543;
        }
        res_8546 = fptosi_f32_i32(res_8539);
        res_8547 = smax32(res_8535, res_8546);
        res_8548 = res_8524 - res_8519;
        res_8549 = sitofp_i32_f32(res_8547);
        x_8550 = res_8549 - res_8519;
        res_8551 = x_8550 / res_8548;
        x_8552 = res_8524 - res_8549;
        res_8553 = x_8552 / res_8548;
        res_8554 = res_8513 * res_8551;
        res_8555 = res_8513 * res_8553;
        res_8556 = res_8241 + index_primexp_8656;
        if (res_8509) {
            int32_t x_8558;
            int32_t res_8559;
            
            x_8558 = sizze_8236 * res_8556;
            res_8559 = res_8535 + x_8558;
            res_8557 = res_8559;
        } else {
            int32_t y_8560;
            int32_t res_8561;
            
            y_8560 = sizze_8236 * res_8535;
            res_8561 = res_8556 + y_8560;
            res_8557 = res_8561;
        }
        if (res_8509) {
            int32_t x_8563;
            int32_t res_8564;
            
            x_8563 = sizze_8236 * res_8556;
            res_8564 = res_8546 + x_8563;
            res_8562 = res_8564;
        } else {
            int32_t y_8565;
            int32_t res_8566;
            
            y_8565 = sizze_8236 * res_8546;
            res_8566 = res_8556 + y_8565;
            res_8562 = res_8566;
        }
        cond_8567 = sle32(0, res_8557);
        res_8568 = slt32(res_8557, y_8260);
        x_8569 = cond_8567 && res_8568;
        if (x_8569) {
            res_8570 = res_8557;
        } else {
            res_8570 = -1;
        }
        if (x_8569) {
            bool cond_8572;
            float res_8573;
            
            cond_8572 = res_8535 == res_8546;
            if (cond_8572) {
                res_8573 = res_8513;
            } else {
                res_8573 = res_8554;
            }
            res_8571 = res_8573;
        } else {
            res_8571 = -1.0F;
        }
        cond_8574 = sle32(0, res_8562);
        res_8575 = slt32(res_8562, y_8260);
        x_8576 = cond_8574 && res_8575;
        if (x_8576) {
            bool cond_8579;
            float res_8580;
            int32_t res_8581;
            
            cond_8579 = res_8535 == res_8546;
            if (cond_8579) {
                res_8580 = -1.0F;
            } else {
                res_8580 = res_8555;
            }
            if (cond_8579) {
                res_8581 = -1;
            } else {
                res_8581 = res_8562;
            }
            res_8577 = res_8580;
            res_8578 = res_8581;
        } else {
            res_8577 = -1.0F;
            res_8578 = -1;
        }
        *(__global float *) &mem_8709[(group_id_8508 * (2 * group_sizze_8501) +
                                       local_tid_8507) * 4] = res_8571;
        *(__global float *) &mem_8709[(group_id_8508 * (2 * group_sizze_8501) +
                                       group_sizze_8501 + local_tid_8507) * 4] =
            res_8577;
        *(__global int32_t *) &mem_8712[(group_id_8508 * (2 *
                                                          group_sizze_8501) +
                                         local_tid_8507) * 4] = res_8570;
        *(__global int32_t *) &mem_8712[(group_id_8508 * (2 *
                                                          group_sizze_8501) +
                                         group_sizze_8501 + local_tid_8507) *
                                        4] = res_8578;
    }
    if (thread_active_8767) {
        for (int32_t i_8768 = 0; i_8768 < 2; i_8768++) {
            *(__global float *) &mem_8717[(i_8768 * (nesting_sizze_8419 *
                                                     num_elems_8257) +
                                           gtid_8497 * num_elems_8257 +
                                           gtid_8498) * 4] = *(__global
                                                               float *) &mem_8709[(group_id_8508 *
                                                                                   (2 *
                                                                                    group_sizze_8501) +
                                                                                   i_8768 *
                                                                                   group_sizze_8501 +
                                                                                   local_tid_8507) *
                                                                                  4];
        }
    }
    if (thread_active_8767) {
        for (int32_t i_8769 = 0; i_8769 < 2; i_8769++) {
            *(__global int32_t *) &mem_8722[(i_8769 * (nesting_sizze_8419 *
                                                       num_elems_8257) +
                                             gtid_8497 * num_elems_8257 +
                                             gtid_8498) * 4] = *(__global
                                                                 int32_t *) &mem_8712[(group_id_8508 *
                                                                                       (2 *
                                                                                        group_sizze_8501) +
                                                                                       i_8769 *
                                                                                       group_sizze_8501 +
                                                                                       local_tid_8507) *
                                                                                      4];
        }
    }
}
__kernel void map_kernel_8591(int32_t sizze_8235, float res_8242,
                              float res_8251, int32_t nesting_sizze_8419,
                              __global unsigned char *rays_mem_8676, __global
                              unsigned char *mem_8688, __global
                              unsigned char *mem_8692, __global
                              unsigned char *mem_8694, __global
                              unsigned char *mem_8697, __global
                              unsigned char *mem_8700, __global
                              unsigned char *mem_8703, __global
                              unsigned char *mem_8706)
{
    int32_t wave_sizze_8762;
    int32_t group_sizze_8763;
    bool thread_active_8764;
    int32_t gtid_8584;
    int32_t global_tid_8591;
    int32_t local_tid_8592;
    int32_t group_id_8593;
    
    global_tid_8591 = get_global_id(0);
    local_tid_8592 = get_local_id(0);
    group_sizze_8763 = get_local_size(0);
    wave_sizze_8762 = LOCKSTEP_WIDTH;
    group_id_8593 = get_group_id(0);
    gtid_8584 = global_tid_8591;
    thread_active_8764 = slt32(gtid_8584, nesting_sizze_8419);
    
    int32_t new_index_8643;
    int32_t binop_y_8645;
    int32_t new_index_8646;
    float x_8594;
    float x_8595;
    float x_8596;
    bool cond_8597;
    float res_8598;
    bool cond_8602;
    float res_8603;
    float res_8607;
    float res_8611;
    float res_8615;
    bool cond_8616;
    bool res_8617;
    bool x_8618;
    float res_8619;
    float res_8620;
    float res_8624;
    bool cond_8625;
    bool x_8626;
    float res_8627;
    float res_8628;
    float x_8632;
    float y_8633;
    float res_8634;
    bool res_8635;
    float res_8636;
    float res_8638;
    float res_8639;
    float y_8640;
    float arg_8641;
    float res_8642;
    
    if (thread_active_8764) {
        new_index_8643 = squot32(gtid_8584, sizze_8235);
        binop_y_8645 = sizze_8235 * new_index_8643;
        new_index_8646 = gtid_8584 - binop_y_8645;
        x_8594 = *(__global float *) &mem_8688[(new_index_8643 * sizze_8235 +
                                                new_index_8646) * 4];
        x_8595 = *(__global float *) &mem_8692[(new_index_8643 * sizze_8235 +
                                                new_index_8646) * 4];
        x_8596 = *(__global float *) &rays_mem_8676[new_index_8646 * 4];
        cond_8597 = x_8594 == 0.0F;
        if (cond_8597) {
            res_8598 = x_8596;
        } else {
            float y_8599;
            float x_8600;
            float res_8601;
            
            y_8599 = res_8251 * x_8595;
            x_8600 = x_8596 - y_8599;
            res_8601 = x_8600 / x_8594;
            res_8598 = res_8601;
        }
        cond_8602 = x_8595 == 0.0F;
        if (cond_8602) {
            res_8603 = x_8596;
        } else {
            float y_8604;
            float x_8605;
            float res_8606;
            
            y_8604 = res_8251 * x_8594;
            x_8605 = x_8596 - y_8604;
            res_8606 = x_8605 / x_8595;
            res_8603 = res_8606;
        }
        if (cond_8602) {
            res_8607 = x_8596;
        } else {
            float y_8608;
            float x_8609;
            float res_8610;
            
            y_8608 = res_8242 * x_8594;
            x_8609 = x_8596 - y_8608;
            res_8610 = x_8609 / x_8595;
            res_8607 = res_8610;
        }
        if (cond_8597) {
            res_8611 = x_8596;
        } else {
            float y_8612;
            float x_8613;
            float res_8614;
            
            y_8612 = res_8242 * x_8595;
            x_8613 = x_8596 - y_8612;
            res_8614 = x_8613 / x_8594;
            res_8611 = res_8614;
        }
        res_8615 = (float) fabs(res_8598);
        cond_8616 = res_8615 <= res_8242;
        res_8617 = !cond_8597;
        x_8618 = cond_8616 && res_8617;
        if (x_8618) {
            res_8619 = res_8251;
            res_8620 = res_8598;
        } else {
            bool cond_8621;
            float res_8622;
            float res_8623;
            
            cond_8621 = res_8603 <= res_8607;
            if (cond_8621) {
                res_8622 = res_8603;
            } else {
                res_8622 = res_8607;
            }
            if (cond_8621) {
                res_8623 = res_8251;
            } else {
                res_8623 = res_8242;
            }
            res_8619 = res_8622;
            res_8620 = res_8623;
        }
        res_8624 = (float) fabs(res_8611);
        cond_8625 = res_8624 <= res_8242;
        x_8626 = res_8617 && cond_8625;
        if (x_8626) {
            res_8627 = res_8242;
            res_8628 = res_8611;
        } else {
            bool cond_8629;
            float res_8630;
            float res_8631;
            
            cond_8629 = res_8603 <= res_8607;
            if (cond_8629) {
                res_8630 = res_8607;
            } else {
                res_8630 = res_8603;
            }
            if (cond_8629) {
                res_8631 = res_8242;
            } else {
                res_8631 = res_8251;
            }
            res_8627 = res_8630;
            res_8628 = res_8631;
        }
        x_8632 = res_8628 - res_8620;
        y_8633 = res_8627 - res_8619;
        res_8634 = x_8632 / y_8633;
        res_8635 = 1.0F < res_8634;
        if (res_8635) {
            float res_8637 = 1.0F / res_8634;
            
            res_8636 = res_8637;
        } else {
            res_8636 = res_8634;
        }
        if (res_8635) {
            res_8638 = res_8620;
        } else {
            res_8638 = res_8619;
        }
        if (res_8635) {
            res_8639 = res_8619;
        } else {
            res_8639 = res_8620;
        }
        y_8640 = fpow32(res_8636, 2.0F);
        arg_8641 = 1.0F + y_8640;
        res_8642 = futrts_sqrt32(arg_8641);
    }
    if (thread_active_8764) {
        *(__global bool *) &mem_8694[gtid_8584] = res_8635;
    }
    if (thread_active_8764) {
        *(__global float *) &mem_8697[gtid_8584 * 4] = res_8636;
    }
    if (thread_active_8764) {
        *(__global float *) &mem_8700[gtid_8584 * 4] = res_8638;
    }
    if (thread_active_8764) {
        *(__global float *) &mem_8703[gtid_8584 * 4] = res_8639;
    }
    if (thread_active_8764) {
        *(__global float *) &mem_8706[gtid_8584 * 4] = res_8642;
    }
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
                                       required_types=["i32", "f32", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"group_size_8404": {"class": "group_size", "value": None},
                                        "group_size_8420": {"class": "group_size", "value": None},
                                        "group_size_8432": {"class": "group_size", "value": None},
                                        "group_size_8458": {"class": "group_size", "value": None},
                                        "group_size_8500": {"class": "group_size", "value": None},
                                        "group_size_8585": {"class": "group_size", "value": None}})
    self.fut_kernel_map_transpose_f32_var = program.fut_kernel_map_transpose_f32
    self.fut_kernel_map_transpose_i32_var = program.fut_kernel_map_transpose_i32
    self.fut_kernel_map_transpose_lowheight_f32_var = program.fut_kernel_map_transpose_lowheight_f32
    self.fut_kernel_map_transpose_lowheight_i32_var = program.fut_kernel_map_transpose_lowheight_i32
    self.fut_kernel_map_transpose_lowwidth_f32_var = program.fut_kernel_map_transpose_lowwidth_f32
    self.fut_kernel_map_transpose_lowwidth_i32_var = program.fut_kernel_map_transpose_lowwidth_i32
    self.fut_kernel_map_transpose_small_f32_var = program.fut_kernel_map_transpose_small_f32
    self.fut_kernel_map_transpose_small_i32_var = program.fut_kernel_map_transpose_small_i32
    self.map_kernel_8410_var = program.map_kernel_8410
    self.map_kernel_8426_var = program.map_kernel_8426
    self.map_kernel_8438_var = program.map_kernel_8438
    self.map_kernel_8464_var = program.map_kernel_8464
    self.map_kernel_8506_var = program.map_kernel_8506
    self.map_kernel_8591_var = program.map_kernel_8591
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
  def futhark_main(self, angles_mem_sizze_8673, angles_mem_8674,
                   rays_mem_sizze_8675, rays_mem_8676, voxels_mem_sizze_8677,
                   voxels_mem_8678, sizze_8234, sizze_8235, sizze_8236,
                   stepsizze_8240):
    res_8241 = sdiv32(sizze_8236, np.int32(2))
    res_8242 = sitofp_i32_f32(res_8241)
    group_sizze_8433 = self.sizes["group_size_8432"]
    y_8434 = (group_sizze_8433 - np.int32(1))
    x_8435 = (sizze_8234 + y_8434)
    num_groups_8436 = squot32(x_8435, group_sizze_8433)
    num_threads_8437 = (group_sizze_8433 * num_groups_8436)
    binop_x_8680 = sext_i32_i64(sizze_8234)
    bytes_8679 = (np.int64(4) * binop_x_8680)
    mem_8681 = opencl_alloc(self, bytes_8679, "mem_8681")
    mem_8684 = opencl_alloc(self, bytes_8679, "mem_8684")
    if ((1 * (num_groups_8436 * group_sizze_8433)) != 0):
      self.map_kernel_8438_var.set_args(np.int32(sizze_8234), angles_mem_8674,
                                        mem_8681, mem_8684)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8438_var,
                                 (np.long((num_groups_8436 * group_sizze_8433)),),
                                 (np.long(group_sizze_8433),))
      if synchronous:
        self.queue.finish()
    nesting_sizze_8419 = (sizze_8234 * sizze_8235)
    group_sizze_8421 = self.sizes["group_size_8420"]
    y_8422 = (group_sizze_8421 - np.int32(1))
    x_8423 = (nesting_sizze_8419 + y_8422)
    num_groups_8424 = squot32(x_8423, group_sizze_8421)
    num_threads_8425 = (group_sizze_8421 * num_groups_8424)
    binop_x_8687 = sext_i32_i64(nesting_sizze_8419)
    bytes_8685 = (np.int64(4) * binop_x_8687)
    mem_8688 = opencl_alloc(self, bytes_8685, "mem_8688")
    if ((1 * (num_groups_8424 * group_sizze_8421)) != 0):
      self.map_kernel_8426_var.set_args(np.int32(sizze_8234),
                                        np.int32(sizze_8235), mem_8681,
                                        mem_8688)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8426_var,
                                 (np.long((num_groups_8424 * group_sizze_8421)),),
                                 (np.long(group_sizze_8421),))
      if synchronous:
        self.queue.finish()
    mem_8681 = None
    group_sizze_8405 = self.sizes["group_size_8404"]
    y_8406 = (group_sizze_8405 - np.int32(1))
    x_8407 = (y_8406 + nesting_sizze_8419)
    num_groups_8408 = squot32(x_8407, group_sizze_8405)
    num_threads_8409 = (group_sizze_8405 * num_groups_8408)
    mem_8692 = opencl_alloc(self, bytes_8685, "mem_8692")
    if ((1 * (num_groups_8408 * group_sizze_8405)) != 0):
      self.map_kernel_8410_var.set_args(np.int32(sizze_8234),
                                        np.int32(sizze_8235), mem_8684,
                                        mem_8692)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8410_var,
                                 (np.long((num_groups_8408 * group_sizze_8405)),),
                                 (np.long(group_sizze_8405),))
      if synchronous:
        self.queue.finish()
    mem_8684 = None
    res_8251 = (np.float32(0.0) - res_8242)
    range_start_8252 = (np.int32(0) - res_8241)
    range_end_8253 = (res_8241 - np.int32(1))
    bounds_invalid_upwards_8254 = slt32(range_end_8253, range_start_8252)
    distance_upwards_exclusive_8255 = (range_end_8253 - range_start_8252)
    distance_8256 = (np.int32(1) + distance_upwards_exclusive_8255)
    if bounds_invalid_upwards_8254:
      num_elems_8257 = np.int32(0)
    else:
      num_elems_8257 = distance_8256
    flat_dim_8258 = (np.int32(2) * num_elems_8257)
    y_8260 = pow32(sizze_8236, np.int32(2))
    group_sizze_8586 = self.sizes["group_size_8585"]
    y_8587 = (group_sizze_8586 - np.int32(1))
    x_8588 = (nesting_sizze_8419 + y_8587)
    num_groups_8589 = squot32(x_8588, group_sizze_8586)
    num_threads_8590 = (group_sizze_8586 * num_groups_8589)
    mem_8694 = opencl_alloc(self, binop_x_8687, "mem_8694")
    mem_8697 = opencl_alloc(self, bytes_8685, "mem_8697")
    mem_8700 = opencl_alloc(self, bytes_8685, "mem_8700")
    mem_8703 = opencl_alloc(self, bytes_8685, "mem_8703")
    mem_8706 = opencl_alloc(self, bytes_8685, "mem_8706")
    if ((1 * (num_groups_8589 * group_sizze_8586)) != 0):
      self.map_kernel_8591_var.set_args(np.int32(sizze_8235),
                                        np.float32(res_8242),
                                        np.float32(res_8251),
                                        np.int32(nesting_sizze_8419),
                                        rays_mem_8676, mem_8688, mem_8692,
                                        mem_8694, mem_8697, mem_8700, mem_8703,
                                        mem_8706)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8591_var,
                                 (np.long((num_groups_8589 * group_sizze_8586)),),
                                 (np.long(group_sizze_8586),))
      if synchronous:
        self.queue.finish()
    mem_8688 = None
    mem_8692 = None
    nesting_sizze_8499 = (num_elems_8257 * nesting_sizze_8419)
    group_sizze_8501 = self.sizes["group_size_8500"]
    y_8502 = (group_sizze_8501 - np.int32(1))
    x_8503 = (nesting_sizze_8499 + y_8502)
    num_groups_8504 = squot32(x_8503, group_sizze_8501)
    num_threads_8505 = (group_sizze_8501 * num_groups_8504)
    binop_x_8714 = (np.int32(2) * nesting_sizze_8419)
    convop_x_8715 = (num_elems_8257 * binop_x_8714)
    binop_x_8716 = sext_i32_i64(convop_x_8715)
    bytes_8713 = (np.int64(4) * binop_x_8716)
    mem_8717 = opencl_alloc(self, bytes_8713, "mem_8717")
    mem_8722 = opencl_alloc(self, bytes_8713, "mem_8722")
    num_threads64_8746 = sext_i32_i64(num_threads_8505)
    total_sizze_8747 = (np.int64(8) * num_threads64_8746)
    mem_8709 = opencl_alloc(self, total_sizze_8747, "mem_8709")
    total_sizze_8748 = (np.int64(8) * num_threads64_8746)
    mem_8712 = opencl_alloc(self, total_sizze_8748, "mem_8712")
    if ((1 * (num_groups_8504 * group_sizze_8501)) != 0):
      self.map_kernel_8506_var.set_args(np.int32(sizze_8236),
                                        np.int32(res_8241),
                                        np.float32(res_8242),
                                        np.int32(range_start_8252),
                                        np.int32(num_elems_8257),
                                        np.int32(y_8260),
                                        np.int32(nesting_sizze_8419), mem_8694,
                                        mem_8697, mem_8700, mem_8703, mem_8706,
                                        mem_8709, mem_8712, mem_8717, mem_8722)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8506_var,
                                 (np.long((num_groups_8504 * group_sizze_8501)),),
                                 (np.long(group_sizze_8501),))
      if synchronous:
        self.queue.finish()
    mem_8694 = None
    mem_8697 = None
    mem_8700 = None
    mem_8703 = None
    mem_8706 = None
    mem_8709 = None
    mem_8712 = None
    group_sizze_8459 = self.sizes["group_size_8458"]
    y_8460 = (group_sizze_8459 - np.int32(1))
    x_8461 = (nesting_sizze_8419 + y_8460)
    num_groups_8462 = squot32(x_8461, group_sizze_8459)
    num_threads_8463 = (group_sizze_8459 * num_groups_8462)
    convop_x_8724 = (flat_dim_8258 * nesting_sizze_8419)
    binop_x_8725 = sext_i32_i64(convop_x_8724)
    bytes_8723 = (np.int64(4) * binop_x_8725)
    mem_8726 = opencl_alloc(self, bytes_8723, "mem_8726")
    self.futhark_map_transpose_opencl_f32(mem_8726, np.int32(0), mem_8717,
                                          np.int32(0), np.int32(1),
                                          (nesting_sizze_8419 * num_elems_8257),
                                          np.int32(2),
                                          ((nesting_sizze_8419 * num_elems_8257) * np.int32(2)),
                                          (nesting_sizze_8419 * flat_dim_8258))
    mem_8717 = None
    mem_8730 = opencl_alloc(self, bytes_8723, "mem_8730")
    self.futhark_map_transpose_opencl_f32(mem_8730, np.int32(0), mem_8726,
                                          np.int32(0), np.int32(1),
                                          flat_dim_8258, nesting_sizze_8419,
                                          (nesting_sizze_8419 * flat_dim_8258),
                                          (nesting_sizze_8419 * flat_dim_8258))
    mem_8726 = None
    mem_8734 = opencl_alloc(self, bytes_8723, "mem_8734")
    self.futhark_map_transpose_opencl_i32(mem_8734, np.int32(0), mem_8722,
                                          np.int32(0), np.int32(1),
                                          (nesting_sizze_8419 * num_elems_8257),
                                          np.int32(2),
                                          ((nesting_sizze_8419 * num_elems_8257) * np.int32(2)),
                                          (nesting_sizze_8419 * flat_dim_8258))
    mem_8722 = None
    mem_8738 = opencl_alloc(self, bytes_8723, "mem_8738")
    self.futhark_map_transpose_opencl_i32(mem_8738, np.int32(0), mem_8734,
                                          np.int32(0), np.int32(1),
                                          flat_dim_8258, nesting_sizze_8419,
                                          (nesting_sizze_8419 * flat_dim_8258),
                                          (nesting_sizze_8419 * flat_dim_8258))
    mem_8734 = None
    mem_8741 = opencl_alloc(self, bytes_8685, "mem_8741")
    if ((1 * (num_groups_8462 * group_sizze_8459)) != 0):
      self.map_kernel_8464_var.set_args(np.int32(flat_dim_8258),
                                        np.int32(nesting_sizze_8419),
                                        voxels_mem_8678, mem_8730, mem_8738,
                                        mem_8741)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8464_var,
                                 (np.long((num_groups_8462 * group_sizze_8459)),),
                                 (np.long(group_sizze_8459),))
      if synchronous:
        self.queue.finish()
    mem_8730 = None
    mem_8738 = None
    out_arrsizze_8752 = nesting_sizze_8419
    out_memsizze_8751 = bytes_8685
    out_mem_8750 = mem_8741
    return (out_memsizze_8751, out_mem_8750, out_arrsizze_8752)
  def main(self, angles_mem_8674_ext, rays_mem_8676_ext, voxels_mem_8678_ext,
           stepsizze_8240_ext):
    try:
      assert ((type(angles_mem_8674_ext) in [np.ndarray,
                                             cl.array.Array]) and (angles_mem_8674_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_8234 = np.int32(angles_mem_8674_ext.shape[0])
      angles_mem_sizze_8673 = np.int64(angles_mem_8674_ext.nbytes)
      if (type(angles_mem_8674_ext) == cl.array.Array):
        angles_mem_8674 = angles_mem_8674_ext.data
      else:
        angles_mem_8674 = opencl_alloc(self, angles_mem_sizze_8673,
                                       "angles_mem_8674")
        if (angles_mem_sizze_8673 != 0):
          cl.enqueue_copy(self.queue, angles_mem_8674,
                          normaliseArray(angles_mem_8674_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(angles_mem_8674_ext),
                                                                                                                            angles_mem_8674_ext))
    try:
      assert ((type(rays_mem_8676_ext) in [np.ndarray,
                                           cl.array.Array]) and (rays_mem_8676_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_8235 = np.int32(rays_mem_8676_ext.shape[0])
      rays_mem_sizze_8675 = np.int64(rays_mem_8676_ext.nbytes)
      if (type(rays_mem_8676_ext) == cl.array.Array):
        rays_mem_8676 = rays_mem_8676_ext.data
      else:
        rays_mem_8676 = opencl_alloc(self, rays_mem_sizze_8675, "rays_mem_8676")
        if (rays_mem_sizze_8675 != 0):
          cl.enqueue_copy(self.queue, rays_mem_8676,
                          normaliseArray(rays_mem_8676_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(rays_mem_8676_ext),
                                                                                                                            rays_mem_8676_ext))
    try:
      assert ((type(voxels_mem_8678_ext) in [np.ndarray,
                                             cl.array.Array]) and (voxels_mem_8678_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_8236 = np.int32(voxels_mem_8678_ext.shape[0])
      voxels_mem_sizze_8677 = np.int64(voxels_mem_8678_ext.nbytes)
      if (type(voxels_mem_8678_ext) == cl.array.Array):
        voxels_mem_8678 = voxels_mem_8678_ext.data
      else:
        voxels_mem_8678 = opencl_alloc(self, voxels_mem_sizze_8677,
                                       "voxels_mem_8678")
        if (voxels_mem_sizze_8677 != 0):
          cl.enqueue_copy(self.queue, voxels_mem_8678,
                          normaliseArray(voxels_mem_8678_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(voxels_mem_8678_ext),
                                                                                                                            voxels_mem_8678_ext))
    try:
      stepsizze_8240 = np.int32(ct.c_int32(stepsizze_8240_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(stepsizze_8240_ext),
                                                                                                                            stepsizze_8240_ext))
    (out_memsizze_8751, out_mem_8750,
     out_arrsizze_8752) = self.futhark_main(angles_mem_sizze_8673,
                                            angles_mem_8674,
                                            rays_mem_sizze_8675, rays_mem_8676,
                                            voxels_mem_sizze_8677,
                                            voxels_mem_8678, sizze_8234,
                                            sizze_8235, sizze_8236,
                                            stepsizze_8240)
    return cl.array.Array(self.queue, (out_arrsizze_8752,), ct.c_float,
                          data=out_mem_8750)