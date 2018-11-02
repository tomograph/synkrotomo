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
#define group_sizze_9291 (group_size_9290)
#define group_sizze_9131 (group_size_9130)
#define y_9134 (group_size_9130 - 1)
#define group_sizze_10185 (group_size_10184)
#define max_num_groups_10187 (max_num_groups_10186)
#define group_sizze_9746 (group_size_9745)
#define group_sizze_9363 (group_size_9362)
#define group_sizze_9363 (group_size_9362)
#define group_sizze_9497 (group_size_9496)
#define group_sizze_9497 (group_size_9496)
#define group_sizze_9363 (group_size_9362)
__kernel void chunked_reduce_kernel_10201(__local volatile
                                          int64_t *mem_aligned_0,
                                          int32_t nesting_sizze_8978, __global
                                          unsigned char *mem_10047,
                                          int32_t num_threads_10193,
                                          int32_t per_thread_elements_10196,
                                          __global unsigned char *mem_10246)
{
    __local volatile char *restrict mem_10243 = mem_aligned_0;
    int32_t wave_sizze_10325;
    int32_t group_sizze_10326;
    bool thread_active_10327;
    int32_t gtid_9032;
    int32_t global_tid_10201;
    int32_t local_tid_10202;
    int32_t group_id_10203;
    
    global_tid_10201 = get_global_id(0);
    local_tid_10202 = get_local_id(0);
    group_sizze_10326 = get_local_size(0);
    wave_sizze_10325 = LOCKSTEP_WIDTH;
    group_id_10203 = get_group_id(0);
    gtid_9032 = global_tid_10201;
    thread_active_10327 = slt32(gtid_9032, nesting_sizze_8978);
    
    int32_t chunk_sizze_10208 = smin32(per_thread_elements_10196,
                                       squot32(nesting_sizze_8978 -
                                               global_tid_10201 +
                                               num_threads_10193 - 1,
                                               num_threads_10193));
    int32_t binop_x_10215;
    int32_t new_index_10216;
    int32_t last_offset_10217;
    int64_t binop_x_10218;
    int64_t bytes_10219;
    
    if (thread_active_10327) {
        binop_x_10215 = 4 * gtid_9032;
        new_index_10216 = 3 + binop_x_10215;
        last_offset_10217 = *(__global int32_t *) &mem_10047[new_index_10216 *
                                                             4];
        binop_x_10218 = sext_i32_i64(last_offset_10217);
        bytes_10219 = 4 * binop_x_10218;
    }
    
    int64_t max_per_thread_10210;
    int64_t final_result_10224;
    int64_t acc_10213 = 0;
    int32_t groupstream_mapaccum_dummy_chunk_sizze_10211 = 1;
    
    if (thread_active_10327) {
        for (int32_t i_10212 = 0; i_10212 < chunk_sizze_10208; i_10212++) {
            int64_t zz_10221 = smax64(acc_10213, bytes_10219);
            int64_t acc_tmp_10328 = zz_10221;
            
            acc_10213 = acc_tmp_10328;
        }
    }
    max_per_thread_10210 = acc_10213;
    for (int32_t comb_iter_10329 = 0; comb_iter_10329 <
         squot32(group_sizze_10185 + group_sizze_10185 - 1, group_sizze_10185);
         comb_iter_10329++) {
        int32_t combine_id_10206;
        int32_t flat_comb_id_10330 = comb_iter_10329 * group_sizze_10185 +
                local_tid_10202;
        
        combine_id_10206 = flat_comb_id_10330;
        if (slt32(combine_id_10206, group_sizze_10185) && 1) {
            *(__local int64_t *) &mem_10243[combine_id_10206 * 8] =
                max_per_thread_10210;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_10332;
    int32_t skip_waves_10331;
    int32_t my_index_10225;
    int32_t other_index_10226;
    int64_t x_10227;
    int64_t y_10228;
    
    my_index_10225 = local_tid_10202;
    offset_10332 = 0;
    other_index_10226 = local_tid_10202 + offset_10332;
    if (slt32(local_tid_10202, group_sizze_10185)) {
        x_10227 = *(__local int64_t *) &mem_10243[(local_tid_10202 +
                                                   offset_10332) * 8];
    }
    offset_10332 = 1;
    other_index_10226 = local_tid_10202 + offset_10332;
    while (slt32(offset_10332, wave_sizze_10325)) {
        if (slt32(other_index_10226, group_sizze_10185) && ((local_tid_10202 -
                                                             squot32(local_tid_10202,
                                                                     wave_sizze_10325) *
                                                             wave_sizze_10325) &
                                                            (2 * offset_10332 -
                                                             1)) == 0) {
            // read array element
            {
                y_10228 = *(volatile __local
                            int64_t *) &mem_10243[(local_tid_10202 +
                                                   offset_10332) * 8];
            }
            
            int64_t zz_10229;
            
            if (thread_active_10327) {
                zz_10229 = smax64(x_10227, y_10228);
            }
            x_10227 = zz_10229;
            *(volatile __local int64_t *) &mem_10243[local_tid_10202 * 8] =
                x_10227;
        }
        offset_10332 *= 2;
        other_index_10226 = local_tid_10202 + offset_10332;
    }
    skip_waves_10331 = 1;
    while (slt32(skip_waves_10331, squot32(group_sizze_10185 +
                                           wave_sizze_10325 - 1,
                                           wave_sizze_10325))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_10332 = skip_waves_10331 * wave_sizze_10325;
        other_index_10226 = local_tid_10202 + offset_10332;
        if (slt32(other_index_10226, group_sizze_10185) && ((local_tid_10202 -
                                                             squot32(local_tid_10202,
                                                                     wave_sizze_10325) *
                                                             wave_sizze_10325) ==
                                                            0 &&
                                                            (squot32(local_tid_10202,
                                                                     wave_sizze_10325) &
                                                             (2 *
                                                              skip_waves_10331 -
                                                              1)) == 0)) {
            // read array element
            {
                y_10228 = *(__local int64_t *) &mem_10243[(local_tid_10202 +
                                                           offset_10332) * 8];
            }
            
            int64_t zz_10229;
            
            if (thread_active_10327) {
                zz_10229 = smax64(x_10227, y_10228);
            }
            x_10227 = zz_10229;
            *(__local int64_t *) &mem_10243[local_tid_10202 * 8] = x_10227;
        }
        skip_waves_10331 *= 2;
    }
    final_result_10224 = x_10227;
    if (local_tid_10202 == 0) {
        *(__global int64_t *) &mem_10246[group_id_10203 * 8] =
            final_result_10224;
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
__kernel void kernel_replicate_8791(__global unsigned char *mem_10074)
{
    const uint replicate_gtid_8791 = get_global_id(0);
    
    if (replicate_gtid_8791 >= 1)
        return;
    *(__global float *) &mem_10074[replicate_gtid_8791 * 4] = 0.0F;
}
__kernel void map_kernel_8969(int32_t sizze_8688, int32_t sizze_8689, __global
                              unsigned char *mem_9973, __global
                              unsigned char *mem_9981)
{
    int32_t wave_sizze_10274;
    int32_t group_sizze_10275;
    bool thread_active_10276;
    int32_t gtid_8960;
    int32_t gtid_8961;
    int32_t global_tid_8969;
    int32_t local_tid_8970;
    int32_t group_id_8971;
    
    global_tid_8969 = get_global_id(0);
    local_tid_8970 = get_local_id(0);
    group_sizze_10275 = get_local_size(0);
    wave_sizze_10274 = LOCKSTEP_WIDTH;
    group_id_8971 = get_group_id(0);
    gtid_8960 = squot32(global_tid_8969, sizze_8689);
    gtid_8961 = global_tid_8969 - squot32(global_tid_8969, sizze_8689) *
        sizze_8689;
    thread_active_10276 = slt32(gtid_8960, sizze_8688) && slt32(gtid_8961,
                                                                sizze_8689);
    
    float res_8972;
    
    if (thread_active_10276) {
        res_8972 = *(__global float *) &mem_9973[gtid_8960 * 4];
    }
    if (thread_active_10276) {
        *(__global float *) &mem_9981[(gtid_8960 * sizze_8689 + gtid_8961) *
                                      4] = res_8972;
    }
}
__kernel void map_kernel_8985(int32_t sizze_8688, int32_t sizze_8689, __global
                              unsigned char *mem_9970, __global
                              unsigned char *mem_9977)
{
    int32_t wave_sizze_10271;
    int32_t group_sizze_10272;
    bool thread_active_10273;
    int32_t gtid_8976;
    int32_t gtid_8977;
    int32_t global_tid_8985;
    int32_t local_tid_8986;
    int32_t group_id_8987;
    
    global_tid_8985 = get_global_id(0);
    local_tid_8986 = get_local_id(0);
    group_sizze_10272 = get_local_size(0);
    wave_sizze_10271 = LOCKSTEP_WIDTH;
    group_id_8987 = get_group_id(0);
    gtid_8976 = squot32(global_tid_8985, sizze_8689);
    gtid_8977 = global_tid_8985 - squot32(global_tid_8985, sizze_8689) *
        sizze_8689;
    thread_active_10273 = slt32(gtid_8976, sizze_8688) && slt32(gtid_8977,
                                                                sizze_8689);
    
    float res_8988;
    
    if (thread_active_10273) {
        res_8988 = *(__global float *) &mem_9970[gtid_8976 * 4];
    }
    if (thread_active_10273) {
        *(__global float *) &mem_9977[(gtid_8976 * sizze_8689 + gtid_8977) *
                                      4] = res_8988;
    }
}
__kernel void map_kernel_8997(int32_t sizze_8688, __global
                              unsigned char *angles_mem_9963, __global
                              unsigned char *mem_9970, __global
                              unsigned char *mem_9973)
{
    int32_t wave_sizze_10268;
    int32_t group_sizze_10269;
    bool thread_active_10270;
    int32_t gtid_8990;
    int32_t global_tid_8997;
    int32_t local_tid_8998;
    int32_t group_id_8999;
    
    global_tid_8997 = get_global_id(0);
    local_tid_8998 = get_local_id(0);
    group_sizze_10269 = get_local_size(0);
    wave_sizze_10268 = LOCKSTEP_WIDTH;
    group_id_8999 = get_group_id(0);
    gtid_8990 = global_tid_8997;
    thread_active_10270 = slt32(gtid_8990, sizze_8688);
    
    float x_9000;
    float res_9001;
    float res_9002;
    
    if (thread_active_10270) {
        x_9000 = *(__global float *) &angles_mem_9963[gtid_8990 * 4];
        res_9001 = futrts_sin32(x_9000);
        res_9002 = futrts_cos32(x_9000);
    }
    if (thread_active_10270) {
        *(__global float *) &mem_9970[gtid_8990 * 4] = res_9001;
    }
    if (thread_active_10270) {
        *(__global float *) &mem_9973[gtid_8990 * 4] = res_9002;
    }
}
__kernel void map_kernel_9039(int32_t sizze_8689, float res_8699,
                              float res_8712, int32_t nesting_sizze_8978,
                              int32_t num_threads_9038, __global
                              unsigned char *rays_mem_9965, __global
                              unsigned char *mem_9977, __global
                              unsigned char *mem_9991, __global
                              unsigned char *mem_9995, __global
                              unsigned char *mem_9999, __global
                              unsigned char *mem_10010, __global
                              unsigned char *mem_10047, __global
                              unsigned char *mem_10050, __global
                              unsigned char *mem_10053, __global
                              unsigned char *mem_10062, __global
                              unsigned char *mem_10065, __global
                              unsigned char *mem_10068, __global
                              unsigned char *mem_10071)
{
    int32_t wave_sizze_10342;
    int32_t group_sizze_10343;
    bool thread_active_10344;
    int32_t gtid_9032;
    int32_t global_tid_9039;
    int32_t local_tid_9040;
    int32_t group_id_9041;
    
    global_tid_9039 = get_global_id(0);
    local_tid_9040 = get_local_id(0);
    group_sizze_10343 = get_local_size(0);
    wave_sizze_10342 = LOCKSTEP_WIDTH;
    group_id_9041 = get_group_id(0);
    gtid_9032 = global_tid_9039;
    thread_active_10344 = slt32(gtid_9032, nesting_sizze_8978);
    
    int32_t new_index_9885;
    int32_t binop_y_9887;
    int32_t new_index_9888;
    float x_9042;
    float x_9043;
    bool cond_9044;
    int32_t binop_x_9934;
    int32_t new_index_9935;
    int32_t last_offset_9050;
    float res_9083;
    float res_9084;
    float res_9085;
    float res_9086;
    
    if (thread_active_10344) {
        new_index_9885 = squot32(gtid_9032, sizze_8689);
        binop_y_9887 = sizze_8689 * new_index_9885;
        new_index_9888 = gtid_9032 - binop_y_9887;
        x_9042 = *(__global float *) &mem_9977[(new_index_9885 * sizze_8689 +
                                                new_index_9888) * 4];
        x_9043 = *(__global float *) &rays_mem_9965[new_index_9888 * 4];
        cond_9044 = *(__global bool *) &mem_9991[gtid_9032];
        binop_x_9934 = 4 * gtid_9032;
        new_index_9935 = 3 + binop_x_9934;
        last_offset_9050 = *(__global int32_t *) &mem_10047[new_index_9935 * 4];
        for (int32_t write_iter_9057 = 0; write_iter_9057 < 4;
             write_iter_9057++) {
            int32_t new_index_9937 = write_iter_9057 + binop_x_9934;
            int32_t write_iv_9058 = *(__global
                                      int32_t *) &mem_10010[new_index_9937 * 4];
            int32_t write_iv_9059 = *(__global
                                      int32_t *) &mem_10047[new_index_9937 * 4];
            float write_iv_9060 = *(__global
                                    float *) &mem_9995[(write_iter_9057 *
                                                        nesting_sizze_8978 +
                                                        gtid_9032) * 4];
            float write_iv_9061 = *(__global
                                    float *) &mem_9999[(write_iter_9057 *
                                                        nesting_sizze_8978 +
                                                        gtid_9032) * 4];
            bool is_this_one_9068 = write_iv_9058 == 0;
            int32_t this_offset_9069 = -1 + write_iv_9059;
            int32_t total_res_9070;
            
            if (is_this_one_9068) {
                total_res_9070 = this_offset_9069;
            } else {
                total_res_9070 = -1;
            }
            
            bool less_than_zzero_9071 = slt32(total_res_9070, 0);
            bool greater_than_sizze_9072 = sle32(last_offset_9050,
                                                 total_res_9070);
            bool outside_bounds_dim_9073 = less_than_zzero_9071 ||
                 greater_than_sizze_9072;
            
            if (!outside_bounds_dim_9073) {
                *(__global float *) &mem_10050[(total_res_9070 *
                                                num_threads_9038 +
                                                global_tid_9039) * 4] =
                    write_iv_9060;
            }
            if (!outside_bounds_dim_9073) {
                *(__global float *) &mem_10053[(total_res_9070 *
                                                num_threads_9038 +
                                                global_tid_9039) * 4] =
                    write_iv_9061;
            }
        }
        if (cond_9044) {
            res_9083 = x_9043;
            res_9084 = res_8712;
            res_9085 = x_9043;
            res_9086 = res_8699;
        } else {
            bool cond_9087;
            float res_9088;
            float res_9089;
            float res_9090;
            float res_9091;
            
            cond_9087 = x_9042 == 1.0F;
            if (cond_9087) {
                res_9088 = res_8712;
                res_9089 = x_9043;
                res_9090 = res_8699;
                res_9091 = x_9043;
            } else {
                float x_9092;
                float y_9093;
                bool cond_9094;
                float res_9095;
                float res_9096;
                float res_9097;
                float res_9098;
                
                x_9092 = *(__global float *) &mem_10050[global_tid_9039 * 4];
                y_9093 = *(__global float *) &mem_10050[(num_threads_9038 +
                                                         global_tid_9039) * 4];
                cond_9094 = x_9092 < y_9093;
                if (cond_9094) {
                    res_9095 = x_9092;
                } else {
                    res_9095 = y_9093;
                }
                if (cond_9094) {
                    res_9096 = y_9093;
                } else {
                    res_9096 = x_9092;
                }
                if (cond_9094) {
                    float res_9099;
                    float res_9100;
                    
                    res_9099 = *(__global float *) &mem_10053[global_tid_9039 *
                                                              4];
                    res_9100 = *(__global
                                 float *) &mem_10053[(num_threads_9038 +
                                                      global_tid_9039) * 4];
                    res_9097 = res_9099;
                    res_9098 = res_9100;
                } else {
                    float res_9101;
                    float res_9102;
                    
                    res_9101 = *(__global
                                 float *) &mem_10053[(num_threads_9038 +
                                                      global_tid_9039) * 4];
                    res_9102 = *(__global float *) &mem_10053[global_tid_9039 *
                                                              4];
                    res_9097 = res_9101;
                    res_9098 = res_9102;
                }
                res_9088 = res_9095;
                res_9089 = res_9097;
                res_9090 = res_9096;
                res_9091 = res_9098;
            }
            res_9083 = res_9088;
            res_9084 = res_9089;
            res_9085 = res_9090;
            res_9086 = res_9091;
        }
    }
    if (thread_active_10344) {
        *(__global float *) &mem_10062[gtid_9032 * 4] = res_9083;
    }
    if (thread_active_10344) {
        *(__global float *) &mem_10065[gtid_9032 * 4] = res_9084;
    }
    if (thread_active_10344) {
        *(__global float *) &mem_10068[gtid_9032 * 4] = res_9085;
    }
    if (thread_active_10344) {
        *(__global float *) &mem_10071[gtid_9032 * 4] = res_9086;
    }
}
__kernel void map_kernel_9286(int32_t y_9194, int32_t convop_x_9993, __global
                              unsigned char *mem_10004, __global
                              unsigned char *mem_10007, __global
                              unsigned char *mem_10039, __global
                              unsigned char *mem_10042, __global
                              unsigned char *mem_10044, __global
                              unsigned char *mem_10047)
{
    int32_t wave_sizze_10322;
    int32_t group_sizze_10323;
    bool thread_active_10324;
    int32_t j_9269;
    int32_t global_tid_9286;
    int32_t local_tid_9287;
    int32_t group_id_9288;
    
    global_tid_9286 = get_global_id(0);
    local_tid_9287 = get_local_id(0);
    group_sizze_10323 = get_local_size(0);
    wave_sizze_10322 = LOCKSTEP_WIDTH;
    group_id_9288 = get_group_id(0);
    j_9269 = global_tid_9286;
    thread_active_10324 = slt32(j_9269, convop_x_9993);
    
    bool y_flag_9262;
    int32_t y_9263;
    int32_t group_id_9274;
    bool cond_9275;
    bool final_result_9278;
    int32_t final_result_9279;
    
    if (thread_active_10324) {
        y_flag_9262 = *(__global bool *) &mem_10004[j_9269];
        y_9263 = *(__global int32_t *) &mem_10007[j_9269 * 4];
        group_id_9274 = squot32(j_9269, y_9194);
        cond_9275 = 0 == group_id_9274;
        if (cond_9275) {
            final_result_9278 = y_flag_9262;
            final_result_9279 = y_9263;
        } else {
            int32_t carry_in_index_9276;
            bool x_flag_9260;
            int32_t x_9261;
            bool new_flag_9264;
            int32_t seg_lhs_9265;
            int32_t zz_9268;
            
            carry_in_index_9276 = group_id_9274 - 1;
            x_flag_9260 = *(__global bool *) &mem_10039[carry_in_index_9276];
            x_9261 = *(__global int32_t *) &mem_10042[carry_in_index_9276 * 4];
            new_flag_9264 = x_flag_9260 || y_flag_9262;
            if (y_flag_9262) {
                seg_lhs_9265 = 0;
            } else {
                seg_lhs_9265 = x_9261;
            }
            zz_9268 = y_9263 + seg_lhs_9265;
            final_result_9278 = new_flag_9264;
            final_result_9279 = zz_9268;
        }
    }
    if (thread_active_10324) {
        *(__global bool *) &mem_10044[j_9269] = final_result_9278;
    }
    if (thread_active_10324) {
        *(__global int32_t *) &mem_10047[j_9269 * 4] = final_result_9279;
    }
}
__kernel void map_kernel_9296(int32_t sizze_8689, float res_8699,
                              float res_8712, int32_t nesting_sizze_8978,
                              __global unsigned char *rays_mem_9965, __global
                              unsigned char *mem_9977, __global
                              unsigned char *mem_9981, __global
                              unsigned char *mem_9984, __global
                              unsigned char *mem_9987, __global
                              unsigned char *mem_9989, __global
                              unsigned char *mem_9991, __global
                              unsigned char *mem_9995, __global
                              unsigned char *mem_9999, __global
                              unsigned char *mem_10002)
{
    int32_t wave_sizze_10277;
    int32_t group_sizze_10278;
    bool thread_active_10279;
    int32_t gtid_9289;
    int32_t global_tid_9296;
    int32_t local_tid_9297;
    int32_t group_id_9298;
    
    global_tid_9296 = get_global_id(0);
    local_tid_9297 = get_local_id(0);
    group_sizze_10278 = get_local_size(0);
    wave_sizze_10277 = LOCKSTEP_WIDTH;
    group_id_9298 = get_group_id(0);
    gtid_9289 = global_tid_9296;
    thread_active_10279 = slt32(gtid_9289, nesting_sizze_8978);
    
    int32_t new_index_9867;
    int32_t binop_y_9869;
    int32_t new_index_9870;
    float x_9299;
    float x_9300;
    float x_9301;
    bool cond_9302;
    float res_9303;
    bool cond_9307;
    float res_9308;
    float res_9312;
    float res_9316;
    float res_9322;
    bool arr_elem_9323;
    float res_9324;
    bool arr_elem_9325;
    float res_9326;
    bool arr_elem_9327;
    float res_9328;
    bool arr_elem_9329;
    
    if (thread_active_10279) {
        new_index_9867 = squot32(gtid_9289, sizze_8689);
        binop_y_9869 = sizze_8689 * new_index_9867;
        new_index_9870 = gtid_9289 - binop_y_9869;
        x_9299 = *(__global float *) &mem_9977[(new_index_9867 * sizze_8689 +
                                                new_index_9870) * 4];
        x_9300 = *(__global float *) &mem_9981[(new_index_9867 * sizze_8689 +
                                                new_index_9870) * 4];
        x_9301 = *(__global float *) &rays_mem_9965[new_index_9870 * 4];
        cond_9302 = x_9299 == 0.0F;
        if (cond_9302) {
            res_9303 = x_9301;
        } else {
            float y_9304;
            float x_9305;
            float res_9306;
            
            y_9304 = res_8712 * x_9300;
            x_9305 = x_9301 - y_9304;
            res_9306 = x_9305 / x_9299;
            res_9303 = res_9306;
        }
        cond_9307 = x_9300 == 0.0F;
        if (cond_9307) {
            res_9308 = x_9301;
        } else {
            float y_9309;
            float x_9310;
            float res_9311;
            
            y_9309 = res_8712 * x_9299;
            x_9310 = x_9301 - y_9309;
            res_9311 = x_9310 / x_9300;
            res_9308 = res_9311;
        }
        if (cond_9307) {
            res_9312 = x_9301;
        } else {
            float y_9313;
            float x_9314;
            float res_9315;
            
            y_9313 = res_8699 * x_9299;
            x_9314 = x_9301 - y_9313;
            res_9315 = x_9314 / x_9300;
            res_9312 = res_9315;
        }
        if (cond_9302) {
            res_9316 = x_9301;
        } else {
            float y_9317;
            float x_9318;
            float res_9319;
            
            y_9317 = res_8699 * x_9300;
            x_9318 = x_9301 - y_9317;
            res_9319 = x_9318 / x_9299;
            res_9316 = res_9319;
        }
        *(__global float *) &mem_9984[(group_id_9298 * (4 * group_sizze_9291) +
                                       local_tid_9297) * 4] = res_8712;
        *(__global float *) &mem_9984[(group_id_9298 * (4 * group_sizze_9291) +
                                       group_sizze_9291 + local_tid_9297) * 4] =
            res_9308;
        *(__global float *) &mem_9984[(group_id_9298 * (4 * group_sizze_9291) +
                                       2 * group_sizze_9291 + local_tid_9297) *
                                      4] = res_9312;
        *(__global float *) &mem_9984[(group_id_9298 * (4 * group_sizze_9291) +
                                       3 * group_sizze_9291 + local_tid_9297) *
                                      4] = res_8699;
        *(__global float *) &mem_9987[(group_id_9298 * (4 * group_sizze_9291) +
                                       local_tid_9297) * 4] = res_9303;
        *(__global float *) &mem_9987[(group_id_9298 * (4 * group_sizze_9291) +
                                       group_sizze_9291 + local_tid_9297) * 4] =
            res_8712;
        *(__global float *) &mem_9987[(group_id_9298 * (4 * group_sizze_9291) +
                                       2 * group_sizze_9291 + local_tid_9297) *
                                      4] = res_8699;
        *(__global float *) &mem_9987[(group_id_9298 * (4 * group_sizze_9291) +
                                       3 * group_sizze_9291 + local_tid_9297) *
                                      4] = res_9316;
        res_9322 = (float) fabs(res_9303);
        arr_elem_9323 = res_9322 <= res_8699;
        res_9324 = (float) fabs(res_9308);
        arr_elem_9325 = res_9324 <= res_8699;
        res_9326 = (float) fabs(res_9312);
        arr_elem_9327 = res_9326 <= res_8699;
        res_9328 = (float) fabs(res_9316);
        arr_elem_9329 = res_9328 <= res_8699;
        *(__global bool *) &mem_9989[group_id_9298 * (4 * group_sizze_9291) +
                                     local_tid_9297] = arr_elem_9323;
        *(__global bool *) &mem_9989[group_id_9298 * (4 * group_sizze_9291) +
                                     group_sizze_9291 + local_tid_9297] =
            arr_elem_9325;
        *(__global bool *) &mem_9989[group_id_9298 * (4 * group_sizze_9291) +
                                     2 * group_sizze_9291 + local_tid_9297] =
            arr_elem_9327;
        *(__global bool *) &mem_9989[group_id_9298 * (4 * group_sizze_9291) +
                                     3 * group_sizze_9291 + local_tid_9297] =
            arr_elem_9329;
    }
    if (thread_active_10279) {
        *(__global bool *) &mem_9991[gtid_9289] = cond_9302;
    }
    if (thread_active_10279) {
        for (int32_t i_10280 = 0; i_10280 < 4; i_10280++) {
            *(__global float *) &mem_9995[(i_10280 * nesting_sizze_8978 +
                                           gtid_9289) * 4] = *(__global
                                                               float *) &mem_9984[(group_id_9298 *
                                                                                   (4 *
                                                                                    group_sizze_9291) +
                                                                                   i_10280 *
                                                                                   group_sizze_9291 +
                                                                                   local_tid_9297) *
                                                                                  4];
        }
    }
    if (thread_active_10279) {
        for (int32_t i_10281 = 0; i_10281 < 4; i_10281++) {
            *(__global float *) &mem_9999[(i_10281 * nesting_sizze_8978 +
                                           gtid_9289) * 4] = *(__global
                                                               float *) &mem_9987[(group_id_9298 *
                                                                                   (4 *
                                                                                    group_sizze_9291) +
                                                                                   i_10281 *
                                                                                   group_sizze_9291 +
                                                                                   local_tid_9297) *
                                                                                  4];
        }
    }
    if (thread_active_10279) {
        for (int32_t i_10282 = 0; i_10282 < 4; i_10282++) {
            *(__global bool *) &mem_10002[gtid_9289 * 4 + i_10282] = *(__global
                                                                       bool *) &mem_9989[group_id_9298 *
                                                                                         (4 *
                                                                                          group_sizze_9291) +
                                                                                         i_10282 *
                                                                                         group_sizze_9291 +
                                                                                         local_tid_9297];
        }
    }
}
__kernel void map_kernel_9751(int32_t res_8697, int32_t res_8698,
                              float res_8699, int32_t range_start_8793,
                              int32_t num_elems_8798, int32_t y_8800,
                              int32_t x_8818, __global
                              unsigned char *voxels_mem_9967, __global
                              unsigned char *mem_10078, __global
                              unsigned char *mem_10081, __global
                              unsigned char *mem_10084, __global
                              unsigned char *mem_10087, __global
                              unsigned char *mem_10090, __global
                              unsigned char *mem_10093, __global
                              unsigned char *mem_10098)
{
    int32_t wave_sizze_10360;
    int32_t group_sizze_10361;
    bool thread_active_10362;
    int32_t gtid_9742;
    int32_t gtid_9743;
    int32_t global_tid_9751;
    int32_t local_tid_9752;
    int32_t group_id_9753;
    
    global_tid_9751 = get_global_id(0);
    local_tid_9752 = get_local_id(0);
    group_sizze_10361 = get_local_size(0);
    wave_sizze_10360 = LOCKSTEP_WIDTH;
    group_id_9753 = get_group_id(0);
    gtid_9742 = squot32(global_tid_9751, num_elems_8798);
    gtid_9743 = global_tid_9751 - squot32(global_tid_9751, num_elems_8798) *
        num_elems_8798;
    thread_active_10362 = slt32(gtid_9742, x_8818) && slt32(gtid_9743,
                                                            num_elems_8798);
    
    bool res_9754;
    float res_9755;
    float res_9756;
    float res_9757;
    float res_9758;
    int32_t index_primexp_9912;
    float res_9760;
    float y_9761;
    float x_9762;
    float x_9763;
    float res_9764;
    float x_9765;
    float y_9766;
    float x_9767;
    float x_9768;
    float res_9769;
    int32_t res_9770;
    float res_9771;
    bool res_9772;
    float res_9773;
    int32_t res_9780;
    int32_t res_9781;
    float res_9782;
    bool res_9783;
    float res_9784;
    int32_t res_9791;
    int32_t res_9792;
    float res_9793;
    float res_9794;
    float x_9795;
    float res_9796;
    float x_9797;
    float res_9798;
    int32_t res_9799;
    int32_t res_9800;
    int32_t res_9807;
    float res_9814;
    float res_9815;
    bool cond_9816;
    bool res_9817;
    bool x_9818;
    float res_9819;
    bool cond_9826;
    bool res_9827;
    bool x_9828;
    float res_9829;
    
    if (thread_active_10362) {
        res_9754 = *(__global bool *) &mem_10078[gtid_9742];
        res_9755 = *(__global float *) &mem_10081[gtid_9742 * 4];
        res_9756 = *(__global float *) &mem_10084[gtid_9742 * 4];
        res_9757 = *(__global float *) &mem_10087[gtid_9742 * 4];
        res_9758 = *(__global float *) &mem_10090[gtid_9742 * 4];
        index_primexp_9912 = range_start_8793 + gtid_9743;
        res_9760 = sitofp_i32_f32(index_primexp_9912);
        y_9761 = res_9760 - res_9755;
        x_9762 = res_9757 * y_9761;
        x_9763 = res_9756 + x_9762;
        res_9764 = res_8699 + x_9763;
        x_9765 = 1.0F + res_9760;
        y_9766 = x_9765 - res_9755;
        x_9767 = res_9757 * y_9766;
        x_9768 = res_9756 + x_9767;
        res_9769 = res_8699 + x_9768;
        res_9770 = fptosi_f32_i32(res_9764);
        res_9771 = sitofp_i32_f32(res_9770);
        res_9772 = 0.0F <= res_9764;
        if (res_9772) {
            bool res_9774;
            float res_9775;
            
            res_9774 = res_9771 < res_9764;
            if (res_9774) {
                res_9775 = res_9771;
            } else {
                res_9775 = res_9764;
            }
            res_9773 = res_9775;
        } else {
            bool res_9776;
            float res_9777;
            
            res_9776 = res_9764 < res_9771;
            if (res_9776) {
                int32_t res_9778;
                float res_9779;
                
                res_9778 = res_9770 - 1;
                res_9779 = sitofp_i32_f32(res_9778);
                res_9777 = res_9779;
            } else {
                res_9777 = res_9764;
            }
            res_9773 = res_9777;
        }
        res_9780 = fptosi_f32_i32(res_9773);
        res_9781 = fptosi_f32_i32(res_9769);
        res_9782 = sitofp_i32_f32(res_9781);
        res_9783 = 0.0F <= res_9769;
        if (res_9783) {
            bool res_9785;
            float res_9786;
            
            res_9785 = res_9782 < res_9769;
            if (res_9785) {
                res_9786 = res_9782;
            } else {
                res_9786 = res_9769;
            }
            res_9784 = res_9786;
        } else {
            bool res_9787;
            float res_9788;
            
            res_9787 = res_9769 < res_9782;
            if (res_9787) {
                int32_t res_9789;
                float res_9790;
                
                res_9789 = res_9781 - 1;
                res_9790 = sitofp_i32_f32(res_9789);
                res_9788 = res_9790;
            } else {
                res_9788 = res_9769;
            }
            res_9784 = res_9788;
        }
        res_9791 = fptosi_f32_i32(res_9784);
        res_9792 = smax32(res_9780, res_9791);
        res_9793 = res_9769 - res_9764;
        res_9794 = sitofp_i32_f32(res_9792);
        x_9795 = res_9794 - res_9764;
        res_9796 = x_9795 / res_9793;
        x_9797 = res_9769 - res_9794;
        res_9798 = x_9797 / res_9793;
        res_9799 = res_8698 + index_primexp_9912;
        if (res_9754) {
            int32_t x_9801;
            int32_t x_9802;
            int32_t x_9803;
            int32_t res_9804;
            
            x_9801 = res_8697 - res_9799;
            x_9802 = x_9801 - 1;
            x_9803 = res_8697 * x_9802;
            res_9804 = res_9780 + x_9803;
            res_9800 = res_9804;
        } else {
            int32_t y_9805;
            int32_t res_9806;
            
            y_9805 = res_8697 * res_9780;
            res_9806 = res_9799 + y_9805;
            res_9800 = res_9806;
        }
        if (res_9754) {
            int32_t x_9808;
            int32_t x_9809;
            int32_t x_9810;
            int32_t res_9811;
            
            x_9808 = res_8697 - res_9799;
            x_9809 = x_9808 - 1;
            x_9810 = res_8697 * x_9809;
            res_9811 = res_9791 + x_9810;
            res_9807 = res_9811;
        } else {
            int32_t y_9812;
            int32_t res_9813;
            
            y_9812 = res_8697 * res_9791;
            res_9813 = res_9799 + y_9812;
            res_9807 = res_9813;
        }
        res_9814 = res_9758 * res_9796;
        res_9815 = res_9758 * res_9798;
        cond_9816 = sle32(0, res_9800);
        res_9817 = slt32(res_9800, y_8800);
        x_9818 = cond_9816 && res_9817;
        if (x_9818) {
            bool cond_9820;
            float res_9821;
            
            cond_9820 = res_9780 == res_9791;
            if (cond_9820) {
                float y_9822;
                float res_9823;
                
                y_9822 = *(__global float *) &voxels_mem_9967[res_9800 * 4];
                res_9823 = res_9758 * y_9822;
                res_9821 = res_9823;
            } else {
                float y_9824;
                float res_9825;
                
                y_9824 = *(__global float *) &voxels_mem_9967[res_9800 * 4];
                res_9825 = res_9814 * y_9824;
                res_9821 = res_9825;
            }
            res_9819 = res_9821;
        } else {
            res_9819 = 0.0F;
        }
        cond_9826 = sle32(0, res_9807);
        res_9827 = slt32(res_9807, y_8800);
        x_9828 = cond_9826 && res_9827;
        if (x_9828) {
            bool cond_9830;
            float res_9831;
            
            cond_9830 = res_9780 == res_9791;
            if (cond_9830) {
                res_9831 = 0.0F;
            } else {
                float y_9832;
                float res_9833;
                
                y_9832 = *(__global float *) &voxels_mem_9967[res_9800 * 4];
                res_9833 = res_9815 * y_9832;
                res_9831 = res_9833;
            }
            res_9829 = res_9831;
        } else {
            res_9829 = 0.0F;
        }
        *(__global float *) &mem_10093[(group_id_9753 * (2 * group_sizze_9746) +
                                        local_tid_9752) * 4] = res_9819;
        *(__global float *) &mem_10093[(group_id_9753 * (2 * group_sizze_9746) +
                                        group_sizze_9746 + local_tid_9752) *
                                       4] = res_9829;
    }
    if (thread_active_10362) {
        for (int32_t i_10363 = 0; i_10363 < 2; i_10363++) {
            *(__global float *) &mem_10098[(i_10363 * (x_8818 *
                                                       num_elems_8798) +
                                            gtid_9742 * num_elems_8798 +
                                            gtid_9743) * 4] = *(__global
                                                                float *) &mem_10093[(group_id_9753 *
                                                                                     (2 *
                                                                                      group_sizze_9746) +
                                                                                     i_10363 *
                                                                                     group_sizze_9746 +
                                                                                     local_tid_9752) *
                                                                                    4];
        }
    }
}
__kernel void map_kernel_9842(int32_t i_8816, int32_t x_8818, __global
                              unsigned char *mem_10062, __global
                              unsigned char *mem_10065, __global
                              unsigned char *mem_10068, __global
                              unsigned char *mem_10071, __global
                              unsigned char *mem_10078, __global
                              unsigned char *mem_10081, __global
                              unsigned char *mem_10084, __global
                              unsigned char *mem_10087, __global
                              unsigned char *mem_10090)
{
    int32_t wave_sizze_10357;
    int32_t group_sizze_10358;
    bool thread_active_10359;
    int32_t gtid_9835;
    int32_t global_tid_9842;
    int32_t local_tid_9843;
    int32_t group_id_9844;
    
    global_tid_9842 = get_global_id(0);
    local_tid_9843 = get_local_id(0);
    group_sizze_10358 = get_local_size(0);
    wave_sizze_10357 = LOCKSTEP_WIDTH;
    group_id_9844 = get_group_id(0);
    gtid_9835 = global_tid_9842;
    thread_active_10359 = slt32(gtid_9835, x_8818);
    
    int32_t j_p_i_t_s_9910;
    float x_9845;
    float x_9846;
    float x_9847;
    float x_9848;
    float x_9849;
    float y_9850;
    float res_9851;
    float res_9852;
    bool res_9853;
    float res_9854;
    float res_9855;
    float res_9861;
    float y_9864;
    float arg_9865;
    float res_9866;
    
    if (thread_active_10359) {
        j_p_i_t_s_9910 = i_8816 + gtid_9835;
        x_9845 = *(__global float *) &mem_10062[j_p_i_t_s_9910 * 4];
        x_9846 = *(__global float *) &mem_10065[j_p_i_t_s_9910 * 4];
        x_9847 = *(__global float *) &mem_10068[j_p_i_t_s_9910 * 4];
        x_9848 = *(__global float *) &mem_10071[j_p_i_t_s_9910 * 4];
        x_9849 = x_9848 - x_9846;
        y_9850 = x_9847 - x_9845;
        res_9851 = x_9849 / y_9850;
        res_9852 = (float) fabs(res_9851);
        res_9853 = 1.0F < res_9852;
        if (res_9853) {
            bool cond_9856;
            float res_9857;
            float res_9858;
            
            cond_9856 = res_9851 < 0.0F;
            if (cond_9856) {
                res_9857 = x_9845;
            } else {
                res_9857 = x_9847;
            }
            if (cond_9856) {
                float res_9859 = 0.0F - x_9846;
                
                res_9858 = res_9859;
            } else {
                float res_9860 = 0.0F - x_9848;
                
                res_9858 = res_9860;
            }
            res_9854 = res_9858;
            res_9855 = res_9857;
        } else {
            res_9854 = x_9845;
            res_9855 = x_9846;
        }
        if (res_9853) {
            float negate_arg_9862;
            float res_9863;
            
            negate_arg_9862 = 1.0F / res_9851;
            res_9863 = 0.0F - negate_arg_9862;
            res_9861 = res_9863;
        } else {
            res_9861 = res_9851;
        }
        y_9864 = res_9861 * res_9861;
        arg_9865 = 1.0F + y_9864;
        res_9866 = futrts_sqrt32(arg_9865);
    }
    if (thread_active_10359) {
        *(__global bool *) &mem_10078[gtid_9835] = res_9853;
    }
    if (thread_active_10359) {
        *(__global float *) &mem_10081[gtid_9835 * 4] = res_9854;
    }
    if (thread_active_10359) {
        *(__global float *) &mem_10084[gtid_9835 * 4] = res_9855;
    }
    if (thread_active_10359) {
        *(__global float *) &mem_10087[gtid_9835 * 4] = res_9861;
    }
    if (thread_active_10359) {
        *(__global float *) &mem_10090[gtid_9835 * 4] = res_9866;
    }
}
__kernel void reduce_kernel_10231(__local volatile int64_t *mem_aligned_0,
                                  int32_t num_groups_10192, __global
                                  unsigned char *mem_10246, __global
                                  unsigned char *mem_10252)
{
    __local volatile char *restrict mem_10249 = mem_aligned_0;
    int32_t wave_sizze_10334;
    int32_t group_sizze_10335;
    bool thread_active_10336;
    int32_t global_tid_10231;
    int32_t local_tid_10232;
    int32_t group_id_10233;
    
    global_tid_10231 = get_global_id(0);
    local_tid_10232 = get_local_id(0);
    group_sizze_10335 = get_local_size(0);
    wave_sizze_10334 = LOCKSTEP_WIDTH;
    group_id_10233 = get_group_id(0);
    thread_active_10336 = 1;
    
    bool in_bounds_10234;
    int64_t x_10253;
    
    if (thread_active_10336) {
        in_bounds_10234 = slt32(local_tid_10232, num_groups_10192);
        if (in_bounds_10234) {
            int64_t x_10235 = *(__global
                                int64_t *) &mem_10246[global_tid_10231 * 8];
            
            x_10253 = x_10235;
        } else {
            x_10253 = 0;
        }
    }
    
    int64_t final_result_10239;
    
    for (int32_t comb_iter_10337 = 0; comb_iter_10337 <
         squot32(max_num_groups_10187 + max_num_groups_10187 - 1,
                 max_num_groups_10187); comb_iter_10337++) {
        int32_t combine_id_10238;
        int32_t flat_comb_id_10338 = comb_iter_10337 * max_num_groups_10187 +
                local_tid_10232;
        
        combine_id_10238 = flat_comb_id_10338;
        if (slt32(combine_id_10238, max_num_groups_10187) && 1) {
            *(__local int64_t *) &mem_10249[combine_id_10238 * 8] = x_10253;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_10340;
    int32_t skip_waves_10339;
    int64_t x_10172;
    int64_t y_10173;
    int32_t my_index_10199;
    int32_t other_index_10200;
    
    my_index_10199 = local_tid_10232;
    offset_10340 = 0;
    other_index_10200 = local_tid_10232 + offset_10340;
    if (slt32(local_tid_10232, max_num_groups_10187)) {
        x_10172 = *(__local int64_t *) &mem_10249[(local_tid_10232 +
                                                   offset_10340) * 8];
    }
    offset_10340 = 1;
    other_index_10200 = local_tid_10232 + offset_10340;
    while (slt32(offset_10340, wave_sizze_10334)) {
        if (slt32(other_index_10200, max_num_groups_10187) &&
            ((local_tid_10232 - squot32(local_tid_10232, wave_sizze_10334) *
              wave_sizze_10334) & (2 * offset_10340 - 1)) == 0) {
            // read array element
            {
                y_10173 = *(volatile __local
                            int64_t *) &mem_10249[(local_tid_10232 +
                                                   offset_10340) * 8];
            }
            
            int64_t zz_10174;
            
            if (thread_active_10336) {
                zz_10174 = smax64(x_10172, y_10173);
            }
            x_10172 = zz_10174;
            *(volatile __local int64_t *) &mem_10249[local_tid_10232 * 8] =
                x_10172;
        }
        offset_10340 *= 2;
        other_index_10200 = local_tid_10232 + offset_10340;
    }
    skip_waves_10339 = 1;
    while (slt32(skip_waves_10339, squot32(max_num_groups_10187 +
                                           wave_sizze_10334 - 1,
                                           wave_sizze_10334))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_10340 = skip_waves_10339 * wave_sizze_10334;
        other_index_10200 = local_tid_10232 + offset_10340;
        if (slt32(other_index_10200, max_num_groups_10187) &&
            ((local_tid_10232 - squot32(local_tid_10232, wave_sizze_10334) *
              wave_sizze_10334) == 0 && (squot32(local_tid_10232,
                                                 wave_sizze_10334) & (2 *
                                                                      skip_waves_10339 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                y_10173 = *(__local int64_t *) &mem_10249[(local_tid_10232 +
                                                           offset_10340) * 8];
            }
            
            int64_t zz_10174;
            
            if (thread_active_10336) {
                zz_10174 = smax64(x_10172, y_10173);
            }
            x_10172 = zz_10174;
            *(__local int64_t *) &mem_10249[local_tid_10232 * 8] = x_10172;
        }
        skip_waves_10339 *= 2;
    }
    final_result_10239 = x_10172;
    if (local_tid_10232 == 0) {
        *(__global int64_t *) &mem_10252[group_id_10233 * 8] =
            final_result_10239;
    }
}
__kernel void scan1_kernel_9185(__local volatile int64_t *mem_aligned_0,
                                __local volatile int64_t *mem_aligned_1,
                                int32_t num_iterations_9190, int32_t y_9194,
                                int32_t convop_x_9993, __global
                                unsigned char *mem_10004, __global
                                unsigned char *mem_10007, __global
                                unsigned char *mem_10010, __global
                                unsigned char *mem_10013, __global
                                unsigned char *mem_10029, __global
                                unsigned char *mem_10032)
{
    __local volatile char *restrict mem_10017 = mem_aligned_0;
    __local volatile char *restrict mem_10020 = mem_aligned_1;
    int32_t wave_sizze_10283;
    int32_t group_sizze_10284;
    bool thread_active_10285;
    int32_t global_tid_9185;
    int32_t local_tid_9186;
    int32_t group_id_9187;
    
    global_tid_9185 = get_global_id(0);
    local_tid_9186 = get_local_id(0);
    group_sizze_10284 = get_local_size(0);
    wave_sizze_10283 = LOCKSTEP_WIDTH;
    group_id_9187 = get_group_id(0);
    thread_active_10285 = 1;
    
    int32_t x_9195;
    bool is_first_thread_9218;
    bool result_9230;
    int32_t result_9231;
    
    if (thread_active_10285) {
        x_9195 = group_id_9187 * y_9194;
        is_first_thread_9218 = local_tid_9186 == 0;
        
        bool x_flag_merge_9191;
        int32_t x_merge_9192;
        
        x_flag_merge_9191 = 0;
        x_merge_9192 = 0;
        for (int32_t i_9193 = 0; i_9193 < num_iterations_9190; i_9193++) {
            int32_t y_9196 = group_sizze_9131 * i_9193;
            int32_t offset_9197 = x_9195 + y_9196;
            int32_t j_9198 = local_tid_9186 + offset_9197;
            bool cond_9199 = slt32(j_9198, convop_x_9993);
            bool foldres_9203;
            int32_t foldres_9204;
            
            if (cond_9199) {
                int32_t cmpop_x_9883;
                bool index_primexp_9884;
                int32_t new_index_9879;
                int32_t binop_y_9881;
                int32_t new_index_9882;
                bool res_r_flat_elem_9201;
                int32_t part_res_9152;
                int32_t part_res_9153;
                bool new_flag_9156;
                int32_t seg_lhs_9157;
                int32_t zz_9160;
                
                cmpop_x_9883 = srem32(j_9198, 4);
                index_primexp_9884 = cmpop_x_9883 == 0;
                new_index_9879 = squot32(j_9198, 4);
                binop_y_9881 = 4 * new_index_9879;
                new_index_9882 = j_9198 - binop_y_9881;
                res_r_flat_elem_9201 = *(__global
                                         bool *) &mem_10013[new_index_9879 * 4 +
                                                            new_index_9882];
                if (res_r_flat_elem_9201) {
                    part_res_9152 = 0;
                } else {
                    part_res_9152 = 1;
                }
                if (res_r_flat_elem_9201) {
                    part_res_9153 = 1;
                } else {
                    part_res_9153 = 0;
                }
                new_flag_9156 = x_flag_merge_9191 || index_primexp_9884;
                if (index_primexp_9884) {
                    seg_lhs_9157 = 0;
                } else {
                    seg_lhs_9157 = x_merge_9192;
                }
                zz_9160 = part_res_9153 + seg_lhs_9157;
                *(__global int32_t *) &mem_10010[j_9198 * 4] = part_res_9152;
                foldres_9203 = new_flag_9156;
                foldres_9204 = zz_9160;
            } else {
                foldres_9203 = x_flag_merge_9191;
                foldres_9204 = x_merge_9192;
            }
            for (int32_t comb_iter_10291 = 0; comb_iter_10291 <
                 squot32(group_sizze_9131 + group_sizze_9131 - 1,
                         group_sizze_9131); comb_iter_10291++) {
                int32_t combine_id_9206;
                int32_t flat_comb_id_10292 = comb_iter_10291 *
                        group_sizze_9131 + local_tid_9186;
                
                combine_id_9206 = flat_comb_id_10292;
                if (slt32(combine_id_9206, group_sizze_9131) && 1) {
                    *(__local bool *) &mem_10017[combine_id_9206] =
                        foldres_9203;
                    *(__local int32_t *) &mem_10020[combine_id_9206 * 4] =
                        foldres_9204;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t my_index_9161;
            int32_t other_index_9162;
            bool x_flag_9163;
            int32_t x_9164;
            bool y_flag_9165;
            int32_t y_9166;
            int32_t my_index_10293;
            int32_t other_index_10294;
            bool x_flag_10295;
            int32_t x_10296;
            bool y_flag_10297;
            int32_t y_10298;
            
            my_index_9161 = local_tid_9186;
            if (slt32(local_tid_9186, group_sizze_9131)) {
                y_flag_9165 = *(volatile __local
                                bool *) &mem_10017[local_tid_9186 *
                                                   sizeof(bool)];
                y_9166 = *(volatile __local
                           int32_t *) &mem_10020[local_tid_9186 *
                                                 sizeof(int32_t)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                int32_t skip_threads_10302 = 1;
                
                while (slt32(skip_threads_10302, 32)) {
                    if (slt32(local_tid_9186, group_sizze_9131) &&
                        sle32(skip_threads_10302, local_tid_9186 -
                              squot32(local_tid_9186, 32) * 32)) {
                        // read operands
                        {
                            x_flag_9163 = *(volatile __local
                                            bool *) &mem_10017[(local_tid_9186 -
                                                                skip_threads_10302) *
                                                               sizeof(bool)];
                            x_9164 = *(volatile __local
                                       int32_t *) &mem_10020[(local_tid_9186 -
                                                              skip_threads_10302) *
                                                             sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            bool new_flag_9167;
                            int32_t seg_lhs_9168;
                            int32_t zz_9171;
                            
                            new_flag_9167 = x_flag_9163 || y_flag_9165;
                            if (y_flag_9165) {
                                seg_lhs_9168 = 0;
                            } else {
                                seg_lhs_9168 = x_9164;
                            }
                            zz_9171 = y_9166 + seg_lhs_9168;
                            y_flag_9165 = new_flag_9167;
                            y_9166 = zz_9171;
                        }
                    }
                    if (sle32(wave_sizze_10283, skip_threads_10302)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (slt32(local_tid_9186, group_sizze_9131) &&
                        sle32(skip_threads_10302, local_tid_9186 -
                              squot32(local_tid_9186, 32) * 32)) {
                        // write result
                        {
                            *(volatile __local
                              bool *) &mem_10017[local_tid_9186 *
                                                 sizeof(bool)] = y_flag_9165;
                            *(volatile __local
                              int32_t *) &mem_10020[local_tid_9186 *
                                                    sizeof(int32_t)] = y_9166;
                        }
                    }
                    if (sle32(wave_sizze_10283, skip_threads_10302)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_10302 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_9186 - squot32(local_tid_9186, 32) * 32) == 31 &&
                    slt32(local_tid_9186, group_sizze_9131)) {
                    *(volatile __local
                      bool *) &mem_10017[squot32(local_tid_9186, 32) *
                                         sizeof(bool)] = y_flag_9165;
                    *(volatile __local
                      int32_t *) &mem_10020[squot32(local_tid_9186, 32) *
                                            sizeof(int32_t)] = y_9166;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
            {
                if (squot32(local_tid_9186, 32) == 0 && slt32(local_tid_9186,
                                                              group_sizze_9131)) {
                    y_flag_10297 = *(volatile __local
                                     bool *) &mem_10017[local_tid_9186 *
                                                        sizeof(bool)];
                    y_10298 = *(volatile __local
                                int32_t *) &mem_10020[local_tid_9186 *
                                                      sizeof(int32_t)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    int32_t skip_threads_10303 = 1;
                    
                    while (slt32(skip_threads_10303, 32)) {
                        if ((squot32(local_tid_9186, 32) == 0 &&
                             slt32(local_tid_9186, group_sizze_9131)) &&
                            sle32(skip_threads_10303, local_tid_9186 -
                                  squot32(local_tid_9186, 32) * 32)) {
                            // read operands
                            {
                                x_flag_10295 = *(volatile __local
                                                 bool *) &mem_10017[(local_tid_9186 -
                                                                     skip_threads_10303) *
                                                                    sizeof(bool)];
                                x_10296 = *(volatile __local
                                            int32_t *) &mem_10020[(local_tid_9186 -
                                                                   skip_threads_10303) *
                                                                  sizeof(int32_t)];
                            }
                            // perform operation
                            {
                                bool new_flag_10299;
                                int32_t seg_lhs_10300;
                                int32_t zz_10301;
                                
                                new_flag_10299 = x_flag_10295 || y_flag_10297;
                                if (y_flag_10297) {
                                    seg_lhs_10300 = 0;
                                } else {
                                    seg_lhs_10300 = x_10296;
                                }
                                zz_10301 = y_10298 + seg_lhs_10300;
                                y_flag_10297 = new_flag_10299;
                                y_10298 = zz_10301;
                            }
                        }
                        if (sle32(wave_sizze_10283, skip_threads_10303)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if ((squot32(local_tid_9186, 32) == 0 &&
                             slt32(local_tid_9186, group_sizze_9131)) &&
                            sle32(skip_threads_10303, local_tid_9186 -
                                  squot32(local_tid_9186, 32) * 32)) {
                            // write result
                            {
                                *(volatile __local
                                  bool *) &mem_10017[local_tid_9186 *
                                                     sizeof(bool)] =
                                    y_flag_10297;
                                *(volatile __local
                                  int32_t *) &mem_10020[local_tid_9186 *
                                                        sizeof(int32_t)] =
                                    y_10298;
                            }
                        }
                        if (sle32(wave_sizze_10283, skip_threads_10303)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_10303 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_9186, 32) == 0 || !slt32(local_tid_9186,
                                                                 group_sizze_9131))) {
                    // read operands
                    {
                        x_flag_9163 = *(volatile __local
                                        bool *) &mem_10017[(squot32(local_tid_9186,
                                                                    32) - 1) *
                                                           sizeof(bool)];
                        x_9164 = *(volatile __local
                                   int32_t *) &mem_10020[(squot32(local_tid_9186,
                                                                  32) - 1) *
                                                         sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        bool new_flag_9167;
                        int32_t seg_lhs_9168;
                        int32_t zz_9171;
                        
                        new_flag_9167 = x_flag_9163 || y_flag_9165;
                        if (y_flag_9165) {
                            seg_lhs_9168 = 0;
                        } else {
                            seg_lhs_9168 = x_9164;
                        }
                        zz_9171 = y_9166 + seg_lhs_9168;
                        y_flag_9165 = new_flag_9167;
                        y_9166 = zz_9171;
                    }
                    // write final result
                    {
                        *(volatile __local bool *) &mem_10017[local_tid_9186 *
                                                              sizeof(bool)] =
                            y_flag_9165;
                        *(volatile __local
                          int32_t *) &mem_10020[local_tid_9186 *
                                                sizeof(int32_t)] = y_9166;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_9186, 32) == 0) {
                    *(volatile __local bool *) &mem_10017[local_tid_9186 *
                                                          sizeof(bool)] =
                        y_flag_9165;
                    *(volatile __local int32_t *) &mem_10020[local_tid_9186 *
                                                             sizeof(int32_t)] =
                        y_9166;
                }
            }
            if (cond_9199) {
                bool scanned_elem_9212;
                int32_t scanned_elem_9213;
                
                scanned_elem_9212 = *(__local
                                      bool *) &mem_10017[local_tid_9186];
                scanned_elem_9213 = *(__local
                                      int32_t *) &mem_10020[local_tid_9186 * 4];
                *(__global bool *) &mem_10004[j_9198] = scanned_elem_9212;
                *(__global int32_t *) &mem_10007[j_9198 * 4] =
                    scanned_elem_9213;
            }
            
            bool new_scan_carry_9221;
            int32_t new_scan_carry_9222;
            
            if (is_first_thread_9218) {
                bool carry_9219;
                int32_t carry_9220;
                
                carry_9219 = *(__local bool *) &mem_10017[y_9134];
                carry_9220 = *(__local int32_t *) &mem_10020[y_9134 * 4];
                new_scan_carry_9221 = carry_9219;
                new_scan_carry_9222 = carry_9220;
            } else {
                new_scan_carry_9221 = 0;
                new_scan_carry_9222 = 0;
            }
            
            bool new_carry_sync_9225;
            int32_t new_carry_sync_9226;
            
            new_carry_sync_9225 = new_scan_carry_9221;
            new_carry_sync_9226 = new_scan_carry_9222;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_flag_merge_tmp_10289 = new_carry_sync_9225;
            int32_t x_merge_tmp_10290;
            
            x_merge_tmp_10290 = new_carry_sync_9226;
            x_flag_merge_9191 = x_flag_merge_tmp_10289;
            x_merge_9192 = x_merge_tmp_10290;
        }
        result_9230 = x_flag_merge_9191;
        result_9231 = x_merge_9192;
    }
    if (local_tid_9186 == 0) {
        *(__global bool *) &mem_10029[group_id_9187] = result_9230;
    }
    if (local_tid_9186 == 0) {
        *(__global int32_t *) &mem_10032[group_id_9187 * 4] = result_9231;
    }
}
__kernel void scan2_kernel_9243(__local volatile int64_t *mem_aligned_0,
                                __local volatile int64_t *mem_aligned_1,
                                int32_t num_groups_9138, __global
                                unsigned char *mem_10029, __global
                                unsigned char *mem_10032, __global
                                unsigned char *mem_10039, __global
                                unsigned char *mem_10042)
{
    __local volatile char *restrict mem_10034 = mem_aligned_0;
    __local volatile char *restrict mem_10037 = mem_aligned_1;
    int32_t wave_sizze_10306;
    int32_t group_sizze_10307;
    bool thread_active_10308;
    int32_t global_tid_9243;
    int32_t local_tid_9244;
    int32_t group_id_9245;
    
    global_tid_9243 = get_global_id(0);
    local_tid_9244 = get_local_id(0);
    group_sizze_10307 = get_local_size(0);
    wave_sizze_10306 = LOCKSTEP_WIDTH;
    group_id_9245 = get_group_id(0);
    thread_active_10308 = 1;
    for (int32_t comb_iter_10309 = 0; comb_iter_10309 <
         squot32(num_groups_9138 + num_groups_9138 - 1, num_groups_9138);
         comb_iter_10309++) {
        int32_t combine_id_9246;
        int32_t flat_comb_id_10310 = comb_iter_10309 * num_groups_9138 +
                local_tid_9244;
        
        combine_id_9246 = flat_comb_id_10310;
        if (slt32(combine_id_9246, num_groups_9138) && 1) {
            bool unused_flag_array_scan_carry_out_elem_9247 = *(__global
                                                                bool *) &mem_10029[combine_id_9246];
            int32_t offsets_r_flat_scan_carry_out_elem_9248 = *(__global
                                                                int32_t *) &mem_10032[combine_id_9246 *
                                                                                      4];
            
            *(__local bool *) &mem_10034[combine_id_9246] =
                unused_flag_array_scan_carry_out_elem_9247;
            *(__local int32_t *) &mem_10037[combine_id_9246 * 4] =
                offsets_r_flat_scan_carry_out_elem_9248;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t my_index_9232;
    int32_t other_index_9233;
    bool x_flag_9234;
    int32_t x_9235;
    bool y_flag_9236;
    int32_t y_9237;
    int32_t my_index_10311;
    int32_t other_index_10312;
    bool x_flag_10313;
    int32_t x_10314;
    bool y_flag_10315;
    int32_t y_10316;
    
    my_index_9232 = local_tid_9244;
    if (slt32(local_tid_9244, num_groups_9138)) {
        y_flag_9236 = *(volatile __local bool *) &mem_10034[local_tid_9244 *
                                                            sizeof(bool)];
        y_9237 = *(volatile __local int32_t *) &mem_10037[local_tid_9244 *
                                                          sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        int32_t skip_threads_10320 = 1;
        
        while (slt32(skip_threads_10320, 32)) {
            if (slt32(local_tid_9244, num_groups_9138) &&
                sle32(skip_threads_10320, local_tid_9244 -
                      squot32(local_tid_9244, 32) * 32)) {
                // read operands
                {
                    x_flag_9234 = *(volatile __local
                                    bool *) &mem_10034[(local_tid_9244 -
                                                        skip_threads_10320) *
                                                       sizeof(bool)];
                    x_9235 = *(volatile __local
                               int32_t *) &mem_10037[(local_tid_9244 -
                                                      skip_threads_10320) *
                                                     sizeof(int32_t)];
                }
                // perform operation
                {
                    bool new_flag_9238;
                    int32_t seg_lhs_9239;
                    int32_t zz_9242;
                    
                    if (thread_active_10308) {
                        new_flag_9238 = x_flag_9234 || y_flag_9236;
                        if (y_flag_9236) {
                            seg_lhs_9239 = 0;
                        } else {
                            seg_lhs_9239 = x_9235;
                        }
                        zz_9242 = y_9237 + seg_lhs_9239;
                    }
                    y_flag_9236 = new_flag_9238;
                    y_9237 = zz_9242;
                }
            }
            if (sle32(wave_sizze_10306, skip_threads_10320)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (slt32(local_tid_9244, num_groups_9138) &&
                sle32(skip_threads_10320, local_tid_9244 -
                      squot32(local_tid_9244, 32) * 32)) {
                // write result
                {
                    *(volatile __local bool *) &mem_10034[local_tid_9244 *
                                                          sizeof(bool)] =
                        y_flag_9236;
                    *(volatile __local int32_t *) &mem_10037[local_tid_9244 *
                                                             sizeof(int32_t)] =
                        y_9237;
                }
            }
            if (sle32(wave_sizze_10306, skip_threads_10320)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_10320 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_9244 - squot32(local_tid_9244, 32) * 32) == 31 &&
            slt32(local_tid_9244, num_groups_9138)) {
            *(volatile __local bool *) &mem_10034[squot32(local_tid_9244, 32) *
                                                  sizeof(bool)] = y_flag_9236;
            *(volatile __local int32_t *) &mem_10037[squot32(local_tid_9244,
                                                             32) *
                                                     sizeof(int32_t)] = y_9237;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_tid_9244, 32) == 0 && slt32(local_tid_9244,
                                                      num_groups_9138)) {
            y_flag_10315 = *(volatile __local
                             bool *) &mem_10034[local_tid_9244 * sizeof(bool)];
            y_10316 = *(volatile __local int32_t *) &mem_10037[local_tid_9244 *
                                                               sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            int32_t skip_threads_10321 = 1;
            
            while (slt32(skip_threads_10321, 32)) {
                if ((squot32(local_tid_9244, 32) == 0 && slt32(local_tid_9244,
                                                               num_groups_9138)) &&
                    sle32(skip_threads_10321, local_tid_9244 -
                          squot32(local_tid_9244, 32) * 32)) {
                    // read operands
                    {
                        x_flag_10313 = *(volatile __local
                                         bool *) &mem_10034[(local_tid_9244 -
                                                             skip_threads_10321) *
                                                            sizeof(bool)];
                        x_10314 = *(volatile __local
                                    int32_t *) &mem_10037[(local_tid_9244 -
                                                           skip_threads_10321) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        bool new_flag_10317;
                        int32_t seg_lhs_10318;
                        int32_t zz_10319;
                        
                        if (thread_active_10308) {
                            new_flag_10317 = x_flag_10313 || y_flag_10315;
                            if (y_flag_10315) {
                                seg_lhs_10318 = 0;
                            } else {
                                seg_lhs_10318 = x_10314;
                            }
                            zz_10319 = y_10316 + seg_lhs_10318;
                        }
                        y_flag_10315 = new_flag_10317;
                        y_10316 = zz_10319;
                    }
                }
                if (sle32(wave_sizze_10306, skip_threads_10321)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if ((squot32(local_tid_9244, 32) == 0 && slt32(local_tid_9244,
                                                               num_groups_9138)) &&
                    sle32(skip_threads_10321, local_tid_9244 -
                          squot32(local_tid_9244, 32) * 32)) {
                    // write result
                    {
                        *(volatile __local bool *) &mem_10034[local_tid_9244 *
                                                              sizeof(bool)] =
                            y_flag_10315;
                        *(volatile __local
                          int32_t *) &mem_10037[local_tid_9244 *
                                                sizeof(int32_t)] = y_10316;
                    }
                }
                if (sle32(wave_sizze_10306, skip_threads_10321)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_10321 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_9244, 32) == 0 || !slt32(local_tid_9244,
                                                         num_groups_9138))) {
            // read operands
            {
                x_flag_9234 = *(volatile __local
                                bool *) &mem_10034[(squot32(local_tid_9244,
                                                            32) - 1) *
                                                   sizeof(bool)];
                x_9235 = *(volatile __local
                           int32_t *) &mem_10037[(squot32(local_tid_9244, 32) -
                                                  1) * sizeof(int32_t)];
            }
            // perform operation
            {
                bool new_flag_9238;
                int32_t seg_lhs_9239;
                int32_t zz_9242;
                
                if (thread_active_10308) {
                    new_flag_9238 = x_flag_9234 || y_flag_9236;
                    if (y_flag_9236) {
                        seg_lhs_9239 = 0;
                    } else {
                        seg_lhs_9239 = x_9235;
                    }
                    zz_9242 = y_9237 + seg_lhs_9239;
                }
                y_flag_9236 = new_flag_9238;
                y_9237 = zz_9242;
            }
            // write final result
            {
                *(volatile __local bool *) &mem_10034[local_tid_9244 *
                                                      sizeof(bool)] =
                    y_flag_9236;
                *(volatile __local int32_t *) &mem_10037[local_tid_9244 *
                                                         sizeof(int32_t)] =
                    y_9237;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_9244, 32) == 0) {
            *(volatile __local bool *) &mem_10034[local_tid_9244 *
                                                  sizeof(bool)] = y_flag_9236;
            *(volatile __local int32_t *) &mem_10037[local_tid_9244 *
                                                     sizeof(int32_t)] = y_9237;
        }
    }
    
    bool scanned_elem_9253;
    int32_t scanned_elem_9254;
    
    if (thread_active_10308) {
        scanned_elem_9253 = *(__local bool *) &mem_10034[local_tid_9244];
        scanned_elem_9254 = *(__local int32_t *) &mem_10037[local_tid_9244 * 4];
    }
    *(__global bool *) &mem_10039[global_tid_9243] = scanned_elem_9253;
    *(__global int32_t *) &mem_10042[global_tid_9243 * 4] = scanned_elem_9254;
}
__kernel void segmented_redomap__large_comm_many_kernel_9453(__local volatile
                                                             int64_t *mem_aligned_0,
                                                             int32_t flat_dim_8801,
                                                             int32_t x_8818,
                                                             int32_t elements_per_thread_9442,
                                                             int32_t num_groups_per_segment_9447,
                                                             int32_t threads_within_segment_9450,
                                                             __global
                                                             unsigned char *mem_10112,
                                                             __global
                                                             unsigned char *mem_10118)
{
    __local volatile char *restrict mem_10115 = mem_aligned_0;
    int32_t wave_sizze_10373;
    int32_t group_sizze_10374;
    bool thread_active_10375;
    int32_t gtid_9332;
    int32_t gtid_9451;
    int32_t gtid_9452;
    int32_t global_tid_9453;
    int32_t local_tid_9454;
    int32_t group_id_9455;
    
    global_tid_9453 = get_global_id(0);
    local_tid_9454 = get_local_id(0);
    group_sizze_10374 = get_local_size(0);
    wave_sizze_10373 = LOCKSTEP_WIDTH;
    group_id_9455 = get_group_id(0);
    gtid_9332 = squot32(global_tid_9453, num_groups_per_segment_9447 *
                        group_sizze_9363);
    gtid_9451 = squot32(global_tid_9453 - squot32(global_tid_9453,
                                                  num_groups_per_segment_9447 *
                                                  group_sizze_9363) *
                        (num_groups_per_segment_9447 * group_sizze_9363),
                        group_sizze_9363);
    gtid_9452 = global_tid_9453 - squot32(global_tid_9453,
                                          num_groups_per_segment_9447 *
                                          group_sizze_9363) *
        (num_groups_per_segment_9447 * group_sizze_9363) -
        squot32(global_tid_9453 - squot32(global_tid_9453,
                                          num_groups_per_segment_9447 *
                                          group_sizze_9363) *
                (num_groups_per_segment_9447 * group_sizze_9363),
                group_sizze_9363) * group_sizze_9363;
    thread_active_10375 = (slt32(gtid_9332, x_8818) && slt32(gtid_9451,
                                                             num_groups_per_segment_9447)) &&
        slt32(gtid_9452, group_sizze_9363);
    
    int32_t y_9469;
    int32_t y_9470;
    int32_t index_within_segment_9471;
    
    if (thread_active_10375) {
        y_9469 = srem32(group_id_9455, num_groups_per_segment_9447);
        y_9470 = group_sizze_9363 * y_9469;
        index_within_segment_9471 = gtid_9452 + y_9470;
    }
    
    int32_t chunk_sizze_9474 = smin32(elements_per_thread_9442,
                                      squot32(flat_dim_8801 -
                                              index_within_segment_9471 +
                                              threads_within_segment_9450 - 1,
                                              threads_within_segment_9450));
    float res_9479;
    
    if (thread_active_10375) {
        float acc_9482 = 0.0F;
        
        for (int32_t i_9481 = 0; i_9481 < chunk_sizze_9474; i_9481++) {
            int32_t j_t_s_9954 = threads_within_segment_9450 * i_9481;
            int32_t j_p_i_t_s_9955 = index_within_segment_9471 + j_t_s_9954;
            float x_9484 = *(__global float *) &mem_10112[(gtid_9332 *
                                                           flat_dim_8801 +
                                                           j_p_i_t_s_9955) * 4];
            float res_9487 = acc_9482 + x_9484;
            float acc_tmp_10376 = res_9487;
            
            acc_9482 = acc_tmp_10376;
        }
        res_9479 = acc_9482;
    }
    
    float final_result_9490;
    
    for (int32_t comb_iter_10377 = 0; comb_iter_10377 <
         squot32(group_sizze_9363 + group_sizze_9363 - 1, group_sizze_9363);
         comb_iter_10377++) {
        int32_t cid_9465;
        int32_t flat_comb_id_10378 = comb_iter_10377 * group_sizze_9363 +
                local_tid_9454;
        
        cid_9465 = flat_comb_id_10378;
        if (slt32(cid_9465, group_sizze_9363) && 1) {
            *(__local float *) &mem_10115[cid_9465 * 4] = res_9479;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_10380;
    int32_t skip_waves_10379;
    int32_t my_index_9491;
    int32_t other_offset_9492;
    float x_9493;
    float x_9494;
    
    my_index_9491 = local_tid_9454;
    offset_10380 = 0;
    other_offset_9492 = local_tid_9454 + offset_10380;
    if (slt32(local_tid_9454, group_sizze_9363)) {
        x_9493 = *(__local float *) &mem_10115[(local_tid_9454 + offset_10380) *
                                               4];
    }
    offset_10380 = 1;
    other_offset_9492 = local_tid_9454 + offset_10380;
    while (slt32(offset_10380, wave_sizze_10373)) {
        if (slt32(other_offset_9492, group_sizze_9363) && ((local_tid_9454 -
                                                            squot32(local_tid_9454,
                                                                    wave_sizze_10373) *
                                                            wave_sizze_10373) &
                                                           (2 * offset_10380 -
                                                            1)) == 0) {
            // read array element
            {
                x_9494 = *(volatile __local
                           float *) &mem_10115[(local_tid_9454 + offset_10380) *
                                               4];
            }
            
            float res_9495;
            
            if (thread_active_10375) {
                res_9495 = x_9493 + x_9494;
            }
            x_9493 = res_9495;
            *(volatile __local float *) &mem_10115[local_tid_9454 * 4] = x_9493;
        }
        offset_10380 *= 2;
        other_offset_9492 = local_tid_9454 + offset_10380;
    }
    skip_waves_10379 = 1;
    while (slt32(skip_waves_10379, squot32(group_sizze_9363 + wave_sizze_10373 -
                                           1, wave_sizze_10373))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_10380 = skip_waves_10379 * wave_sizze_10373;
        other_offset_9492 = local_tid_9454 + offset_10380;
        if (slt32(other_offset_9492, group_sizze_9363) && ((local_tid_9454 -
                                                            squot32(local_tid_9454,
                                                                    wave_sizze_10373) *
                                                            wave_sizze_10373) ==
                                                           0 &&
                                                           (squot32(local_tid_9454,
                                                                    wave_sizze_10373) &
                                                            (2 *
                                                             skip_waves_10379 -
                                                             1)) == 0)) {
            // read array element
            {
                x_9494 = *(__local float *) &mem_10115[(local_tid_9454 +
                                                        offset_10380) * 4];
            }
            
            float res_9495;
            
            if (thread_active_10375) {
                res_9495 = x_9493 + x_9494;
            }
            x_9493 = res_9495;
            *(__local float *) &mem_10115[local_tid_9454 * 4] = x_9493;
        }
        skip_waves_10379 *= 2;
    }
    final_result_9490 = x_9493;
    if (local_tid_9454 == 0) {
        *(__global float *) &mem_10118[group_id_9455 * 4] = final_result_9490;
    }
}
__kernel void segmented_redomap__large_comm_one_kernel_9390(__local volatile
                                                            int64_t *mem_aligned_0,
                                                            int32_t flat_dim_8801,
                                                            int32_t x_8818,
                                                            int32_t elements_per_thread_9385,
                                                            __global
                                                            unsigned char *mem_10102,
                                                            __global
                                                            unsigned char *mem_10108)
{
    __local volatile char *restrict mem_10105 = mem_aligned_0;
    int32_t wave_sizze_10364;
    int32_t group_sizze_10365;
    bool thread_active_10366;
    int32_t gtid_9332;
    int32_t gtid_9388;
    int32_t gtid_9389;
    int32_t global_tid_9390;
    int32_t local_tid_9391;
    int32_t group_id_9392;
    
    global_tid_9390 = get_global_id(0);
    local_tid_9391 = get_local_id(0);
    group_sizze_10365 = get_local_size(0);
    wave_sizze_10364 = LOCKSTEP_WIDTH;
    group_id_9392 = get_group_id(0);
    gtid_9332 = squot32(global_tid_9390, group_sizze_9363);
    gtid_9388 = squot32(global_tid_9390 - squot32(global_tid_9390,
                                                  group_sizze_9363) *
                        group_sizze_9363, group_sizze_9363);
    gtid_9389 = global_tid_9390 - squot32(global_tid_9390, group_sizze_9363) *
        group_sizze_9363 - squot32(global_tid_9390 - squot32(global_tid_9390,
                                                             group_sizze_9363) *
                                   group_sizze_9363, group_sizze_9363) *
        group_sizze_9363;
    thread_active_10366 = (slt32(gtid_9332, x_8818) && slt32(gtid_9388, 1)) &&
        slt32(gtid_9389, group_sizze_9363);
    
    int32_t chunk_sizze_9411 = smin32(elements_per_thread_9385,
                                      squot32(flat_dim_8801 - gtid_9389 +
                                              group_sizze_9363 - 1,
                                              group_sizze_9363));
    float res_9416;
    
    if (thread_active_10366) {
        float acc_9419 = 0.0F;
        
        for (int32_t i_9418 = 0; i_9418 < chunk_sizze_9411; i_9418++) {
            int32_t j_t_s_9950 = group_sizze_9363 * i_9418;
            int32_t j_p_i_t_s_9951 = gtid_9389 + j_t_s_9950;
            float x_9421 = *(__global float *) &mem_10102[(gtid_9332 *
                                                           flat_dim_8801 +
                                                           j_p_i_t_s_9951) * 4];
            float res_9424 = acc_9419 + x_9421;
            float acc_tmp_10367 = res_9424;
            
            acc_9419 = acc_tmp_10367;
        }
        res_9416 = acc_9419;
    }
    
    float final_result_9427;
    
    for (int32_t comb_iter_10368 = 0; comb_iter_10368 <
         squot32(group_sizze_9363 + group_sizze_9363 - 1, group_sizze_9363);
         comb_iter_10368++) {
        int32_t cid_9402;
        int32_t flat_comb_id_10369 = comb_iter_10368 * group_sizze_9363 +
                local_tid_9391;
        
        cid_9402 = flat_comb_id_10369;
        if (slt32(cid_9402, group_sizze_9363) && 1) {
            *(__local float *) &mem_10105[cid_9402 * 4] = res_9416;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_10371;
    int32_t skip_waves_10370;
    int32_t my_index_9428;
    int32_t other_offset_9429;
    float x_9430;
    float x_9431;
    
    my_index_9428 = local_tid_9391;
    offset_10371 = 0;
    other_offset_9429 = local_tid_9391 + offset_10371;
    if (slt32(local_tid_9391, group_sizze_9363)) {
        x_9430 = *(__local float *) &mem_10105[(local_tid_9391 + offset_10371) *
                                               4];
    }
    offset_10371 = 1;
    other_offset_9429 = local_tid_9391 + offset_10371;
    while (slt32(offset_10371, wave_sizze_10364)) {
        if (slt32(other_offset_9429, group_sizze_9363) && ((local_tid_9391 -
                                                            squot32(local_tid_9391,
                                                                    wave_sizze_10364) *
                                                            wave_sizze_10364) &
                                                           (2 * offset_10371 -
                                                            1)) == 0) {
            // read array element
            {
                x_9431 = *(volatile __local
                           float *) &mem_10105[(local_tid_9391 + offset_10371) *
                                               4];
            }
            
            float res_9432;
            
            if (thread_active_10366) {
                res_9432 = x_9430 + x_9431;
            }
            x_9430 = res_9432;
            *(volatile __local float *) &mem_10105[local_tid_9391 * 4] = x_9430;
        }
        offset_10371 *= 2;
        other_offset_9429 = local_tid_9391 + offset_10371;
    }
    skip_waves_10370 = 1;
    while (slt32(skip_waves_10370, squot32(group_sizze_9363 + wave_sizze_10364 -
                                           1, wave_sizze_10364))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_10371 = skip_waves_10370 * wave_sizze_10364;
        other_offset_9429 = local_tid_9391 + offset_10371;
        if (slt32(other_offset_9429, group_sizze_9363) && ((local_tid_9391 -
                                                            squot32(local_tid_9391,
                                                                    wave_sizze_10364) *
                                                            wave_sizze_10364) ==
                                                           0 &&
                                                           (squot32(local_tid_9391,
                                                                    wave_sizze_10364) &
                                                            (2 *
                                                             skip_waves_10370 -
                                                             1)) == 0)) {
            // read array element
            {
                x_9431 = *(__local float *) &mem_10105[(local_tid_9391 +
                                                        offset_10371) * 4];
            }
            
            float res_9432;
            
            if (thread_active_10366) {
                res_9432 = x_9430 + x_9431;
            }
            x_9430 = res_9432;
            *(__local float *) &mem_10105[local_tid_9391 * 4] = x_9430;
        }
        skip_waves_10370 *= 2;
    }
    final_result_9427 = x_9430;
    if (local_tid_9391 == 0) {
        *(__global float *) &mem_10108[group_id_9392 * 4] = final_result_9427;
    }
}
__kernel void segmented_redomap__large_comm_one_kernel_9509(__local volatile
                                                            int64_t *mem_aligned_0,
                                                            int32_t x_8818,
                                                            int32_t num_groups_per_segment_9447,
                                                            int32_t elements_per_thread_9585,
                                                            __global
                                                            unsigned char *mem_10118,
                                                            __global
                                                            unsigned char *mem_10124)
{
    __local volatile char *restrict mem_10121 = mem_aligned_0;
    int32_t wave_sizze_10382;
    int32_t group_sizze_10383;
    bool thread_active_10384;
    int32_t gtid_9332;
    int32_t gtid_9507;
    int32_t gtid_9508;
    int32_t global_tid_9509;
    int32_t local_tid_9510;
    int32_t group_id_9511;
    
    global_tid_9509 = get_global_id(0);
    local_tid_9510 = get_local_id(0);
    group_sizze_10383 = get_local_size(0);
    wave_sizze_10382 = LOCKSTEP_WIDTH;
    group_id_9511 = get_group_id(0);
    gtid_9332 = squot32(global_tid_9509, group_sizze_9497);
    gtid_9507 = squot32(global_tid_9509 - squot32(global_tid_9509,
                                                  group_sizze_9497) *
                        group_sizze_9497, group_sizze_9497);
    gtid_9508 = global_tid_9509 - squot32(global_tid_9509, group_sizze_9497) *
        group_sizze_9497 - squot32(global_tid_9509 - squot32(global_tid_9509,
                                                             group_sizze_9497) *
                                   group_sizze_9497, group_sizze_9497) *
        group_sizze_9497;
    thread_active_10384 = (slt32(gtid_9332, x_8818) && slt32(gtid_9507, 1)) &&
        slt32(gtid_9508, group_sizze_9497);
    
    int32_t chunk_sizze_9595 = smin32(elements_per_thread_9585,
                                      squot32(num_groups_per_segment_9447 -
                                              gtid_9508 + group_sizze_9497 - 1,
                                              group_sizze_9497));
    int32_t binop_x_9960;
    float res_9600;
    
    if (thread_active_10384) {
        binop_x_9960 = gtid_9332 * num_groups_per_segment_9447;
        
        float acc_9603 = 0.0F;
        
        for (int32_t i_9602 = 0; i_9602 < chunk_sizze_9595; i_9602++) {
            int32_t j_t_s_9958 = group_sizze_9497 * i_9602;
            int32_t j_p_i_t_s_9959 = gtid_9508 + j_t_s_9958;
            int32_t new_index_9961 = j_p_i_t_s_9959 + binop_x_9960;
            float x_9605 = *(__global float *) &mem_10118[new_index_9961 * 4];
            float res_9607 = acc_9603 + x_9605;
            float acc_tmp_10385 = res_9607;
            
            acc_9603 = acc_tmp_10385;
        }
        res_9600 = acc_9603;
    }
    
    float final_result_9610;
    
    for (int32_t comb_iter_10386 = 0; comb_iter_10386 <
         squot32(group_sizze_9497 + group_sizze_9497 - 1, group_sizze_9497);
         comb_iter_10386++) {
        int32_t cid_9521;
        int32_t flat_comb_id_10387 = comb_iter_10386 * group_sizze_9497 +
                local_tid_9510;
        
        cid_9521 = flat_comb_id_10387;
        if (slt32(cid_9521, group_sizze_9497) && 1) {
            *(__local float *) &mem_10121[cid_9521 * 4] = res_9600;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_10389;
    int32_t skip_waves_10388;
    int32_t my_index_9611;
    int32_t other_offset_9612;
    float x_9613;
    float x_9614;
    
    my_index_9611 = local_tid_9510;
    offset_10389 = 0;
    other_offset_9612 = local_tid_9510 + offset_10389;
    if (slt32(local_tid_9510, group_sizze_9497)) {
        x_9613 = *(__local float *) &mem_10121[(local_tid_9510 + offset_10389) *
                                               4];
    }
    offset_10389 = 1;
    other_offset_9612 = local_tid_9510 + offset_10389;
    while (slt32(offset_10389, wave_sizze_10382)) {
        if (slt32(other_offset_9612, group_sizze_9497) && ((local_tid_9510 -
                                                            squot32(local_tid_9510,
                                                                    wave_sizze_10382) *
                                                            wave_sizze_10382) &
                                                           (2 * offset_10389 -
                                                            1)) == 0) {
            // read array element
            {
                x_9614 = *(volatile __local
                           float *) &mem_10121[(local_tid_9510 + offset_10389) *
                                               4];
            }
            
            float res_9615;
            
            if (thread_active_10384) {
                res_9615 = x_9613 + x_9614;
            }
            x_9613 = res_9615;
            *(volatile __local float *) &mem_10121[local_tid_9510 * 4] = x_9613;
        }
        offset_10389 *= 2;
        other_offset_9612 = local_tid_9510 + offset_10389;
    }
    skip_waves_10388 = 1;
    while (slt32(skip_waves_10388, squot32(group_sizze_9497 + wave_sizze_10382 -
                                           1, wave_sizze_10382))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_10389 = skip_waves_10388 * wave_sizze_10382;
        other_offset_9612 = local_tid_9510 + offset_10389;
        if (slt32(other_offset_9612, group_sizze_9497) && ((local_tid_9510 -
                                                            squot32(local_tid_9510,
                                                                    wave_sizze_10382) *
                                                            wave_sizze_10382) ==
                                                           0 &&
                                                           (squot32(local_tid_9510,
                                                                    wave_sizze_10382) &
                                                            (2 *
                                                             skip_waves_10388 -
                                                             1)) == 0)) {
            // read array element
            {
                x_9614 = *(__local float *) &mem_10121[(local_tid_9510 +
                                                        offset_10389) * 4];
            }
            
            float res_9615;
            
            if (thread_active_10384) {
                res_9615 = x_9613 + x_9614;
            }
            x_9613 = res_9615;
            *(__local float *) &mem_10121[local_tid_9510 * 4] = x_9613;
        }
        skip_waves_10388 *= 2;
    }
    final_result_9610 = x_9613;
    if (local_tid_9510 == 0) {
        *(__global float *) &mem_10124[group_id_9511 * 4] = final_result_9610;
    }
}
__kernel void segmented_redomap__small_comm_kernel_9540(__local volatile
                                                        int64_t *mem_aligned_0,
                                                        __local volatile
                                                        int64_t *mem_aligned_1,
                                                        int32_t x_8818,
                                                        int32_t num_groups_per_segment_9447,
                                                        int32_t num_segments_per_group_9618,
                                                        int32_t active_threads_per_group_9623,
                                                        int32_t active_threads_last_group_9628,
                                                        int32_t y_9630, __global
                                                        unsigned char *mem_10118,
                                                        __global
                                                        unsigned char *mem_10127)
{
    __local volatile char *restrict mem_10129 = mem_aligned_0;
    __local volatile char *restrict mem_10132 = mem_aligned_1;
    int32_t wave_sizze_10391;
    int32_t group_sizze_10392;
    bool thread_active_10393;
    int32_t global_tid_9540;
    int32_t local_tid_9541;
    int32_t group_id_9542;
    
    global_tid_9540 = get_global_id(0);
    local_tid_9541 = get_local_id(0);
    group_sizze_10392 = get_local_size(0);
    wave_sizze_10391 = LOCKSTEP_WIDTH;
    group_id_9542 = get_group_id(0);
    thread_active_10393 = 1;
    
    bool islastgroup_9631;
    int32_t active_thread_this_group_9632;
    bool isactive_9633;
    float redtmp_res_9635;
    int32_t x_9649;
    bool isfirstinsegment_9650;
    
    if (thread_active_10393) {
        islastgroup_9631 = group_id_9542 == y_9630;
        if (islastgroup_9631) {
            active_thread_this_group_9632 = active_threads_last_group_9628;
        } else {
            active_thread_this_group_9632 = active_threads_per_group_9623;
        }
        isactive_9633 = slt32(local_tid_9541, active_thread_this_group_9632);
        if (isactive_9633) {
            int32_t x_9636;
            int32_t y_9637;
            int32_t segment_index_9638;
            int32_t index_within_segment_9639;
            int32_t y_9640;
            int32_t offset_9641;
            float x_9645;
            
            x_9636 = squot32(local_tid_9541, num_groups_per_segment_9447);
            y_9637 = group_id_9542 * num_segments_per_group_9618;
            segment_index_9638 = x_9636 + y_9637;
            index_within_segment_9639 = srem32(local_tid_9541,
                                               num_groups_per_segment_9447);
            y_9640 = num_groups_per_segment_9447 * segment_index_9638;
            offset_9641 = index_within_segment_9639 + y_9640;
            x_9645 = *(__global float *) &mem_10118[offset_9641 * 4];
            redtmp_res_9635 = x_9645;
        } else {
            redtmp_res_9635 = 0.0F;
        }
        x_9649 = srem32(local_tid_9541, num_groups_per_segment_9447);
        isfirstinsegment_9650 = x_9649 == 0;
    }
    for (int32_t comb_iter_10394 = 0; comb_iter_10394 <
         squot32(group_sizze_9497 + group_sizze_9497 - 1, group_sizze_9497);
         comb_iter_10394++) {
        int32_t cid_9561;
        int32_t flat_comb_id_10395 = comb_iter_10394 * group_sizze_9497 +
                local_tid_9541;
        
        cid_9561 = flat_comb_id_10395;
        if (slt32(cid_9561, group_sizze_9497) && 1) {
            *(__local bool *) &mem_10129[cid_9561] = isfirstinsegment_9650;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t comb_iter_10396 = 0; comb_iter_10396 <
         squot32(group_sizze_9497 + group_sizze_9497 - 1, group_sizze_9497);
         comb_iter_10396++) {
        int32_t cid_9562;
        int32_t flat_comb_id_10397 = comb_iter_10396 * group_sizze_9497 +
                local_tid_9541;
        
        cid_9562 = flat_comb_id_10397;
        if (slt32(cid_9562, group_sizze_9497) && 1) {
            *(__local float *) &mem_10132[cid_9562 * 4] = redtmp_res_9635;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t my_index_9655;
    int32_t other_offset_9656;
    bool x_flag_9657;
    float x_9658;
    bool y_flag_9659;
    float x_9660;
    int32_t my_index_10398;
    int32_t other_offset_10399;
    bool x_flag_10400;
    float x_10401;
    bool y_flag_10402;
    float x_10403;
    
    my_index_9655 = local_tid_9541;
    if (slt32(local_tid_9541, group_sizze_9497)) {
        y_flag_9659 = *(volatile __local bool *) &mem_10129[local_tid_9541 *
                                                            sizeof(bool)];
        x_9660 = *(volatile __local float *) &mem_10132[local_tid_9541 *
                                                        sizeof(float)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        int32_t skip_threads_10407 = 1;
        
        while (slt32(skip_threads_10407, 32)) {
            if (slt32(local_tid_9541, group_sizze_9497) &&
                sle32(skip_threads_10407, local_tid_9541 -
                      squot32(local_tid_9541, 32) * 32)) {
                // read operands
                {
                    x_flag_9657 = *(volatile __local
                                    bool *) &mem_10129[(local_tid_9541 -
                                                        skip_threads_10407) *
                                                       sizeof(bool)];
                    x_9658 = *(volatile __local
                               float *) &mem_10132[(local_tid_9541 -
                                                    skip_threads_10407) *
                                                   sizeof(float)];
                }
                // perform operation
                {
                    bool new_flag_9661;
                    float seg_lhs_9662;
                    float res_9665;
                    
                    if (thread_active_10393) {
                        new_flag_9661 = x_flag_9657 || y_flag_9659;
                        if (y_flag_9659) {
                            seg_lhs_9662 = 0.0F;
                        } else {
                            seg_lhs_9662 = x_9658;
                        }
                        res_9665 = x_9660 + seg_lhs_9662;
                    }
                    y_flag_9659 = new_flag_9661;
                    x_9660 = res_9665;
                }
            }
            if (sle32(wave_sizze_10391, skip_threads_10407)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (slt32(local_tid_9541, group_sizze_9497) &&
                sle32(skip_threads_10407, local_tid_9541 -
                      squot32(local_tid_9541, 32) * 32)) {
                // write result
                {
                    *(volatile __local bool *) &mem_10129[local_tid_9541 *
                                                          sizeof(bool)] =
                        y_flag_9659;
                    *(volatile __local float *) &mem_10132[local_tid_9541 *
                                                           sizeof(float)] =
                        x_9660;
                }
            }
            if (sle32(wave_sizze_10391, skip_threads_10407)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_10407 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_9541 - squot32(local_tid_9541, 32) * 32) == 31 &&
            slt32(local_tid_9541, group_sizze_9497)) {
            *(volatile __local bool *) &mem_10129[squot32(local_tid_9541, 32) *
                                                  sizeof(bool)] = y_flag_9659;
            *(volatile __local float *) &mem_10132[squot32(local_tid_9541, 32) *
                                                   sizeof(float)] = x_9660;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_tid_9541, 32) == 0 && slt32(local_tid_9541,
                                                      group_sizze_9497)) {
            y_flag_10402 = *(volatile __local
                             bool *) &mem_10129[local_tid_9541 * sizeof(bool)];
            x_10403 = *(volatile __local float *) &mem_10132[local_tid_9541 *
                                                             sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            int32_t skip_threads_10408 = 1;
            
            while (slt32(skip_threads_10408, 32)) {
                if ((squot32(local_tid_9541, 32) == 0 && slt32(local_tid_9541,
                                                               group_sizze_9497)) &&
                    sle32(skip_threads_10408, local_tid_9541 -
                          squot32(local_tid_9541, 32) * 32)) {
                    // read operands
                    {
                        x_flag_10400 = *(volatile __local
                                         bool *) &mem_10129[(local_tid_9541 -
                                                             skip_threads_10408) *
                                                            sizeof(bool)];
                        x_10401 = *(volatile __local
                                    float *) &mem_10132[(local_tid_9541 -
                                                         skip_threads_10408) *
                                                        sizeof(float)];
                    }
                    // perform operation
                    {
                        bool new_flag_10404;
                        float seg_lhs_10405;
                        float res_10406;
                        
                        if (thread_active_10393) {
                            new_flag_10404 = x_flag_10400 || y_flag_10402;
                            if (y_flag_10402) {
                                seg_lhs_10405 = 0.0F;
                            } else {
                                seg_lhs_10405 = x_10401;
                            }
                            res_10406 = x_10403 + seg_lhs_10405;
                        }
                        y_flag_10402 = new_flag_10404;
                        x_10403 = res_10406;
                    }
                }
                if (sle32(wave_sizze_10391, skip_threads_10408)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if ((squot32(local_tid_9541, 32) == 0 && slt32(local_tid_9541,
                                                               group_sizze_9497)) &&
                    sle32(skip_threads_10408, local_tid_9541 -
                          squot32(local_tid_9541, 32) * 32)) {
                    // write result
                    {
                        *(volatile __local bool *) &mem_10129[local_tid_9541 *
                                                              sizeof(bool)] =
                            y_flag_10402;
                        *(volatile __local float *) &mem_10132[local_tid_9541 *
                                                               sizeof(float)] =
                            x_10403;
                    }
                }
                if (sle32(wave_sizze_10391, skip_threads_10408)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_10408 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_9541, 32) == 0 || !slt32(local_tid_9541,
                                                         group_sizze_9497))) {
            // read operands
            {
                x_flag_9657 = *(volatile __local
                                bool *) &mem_10129[(squot32(local_tid_9541,
                                                            32) - 1) *
                                                   sizeof(bool)];
                x_9658 = *(volatile __local
                           float *) &mem_10132[(squot32(local_tid_9541, 32) -
                                                1) * sizeof(float)];
            }
            // perform operation
            {
                bool new_flag_9661;
                float seg_lhs_9662;
                float res_9665;
                
                if (thread_active_10393) {
                    new_flag_9661 = x_flag_9657 || y_flag_9659;
                    if (y_flag_9659) {
                        seg_lhs_9662 = 0.0F;
                    } else {
                        seg_lhs_9662 = x_9658;
                    }
                    res_9665 = x_9660 + seg_lhs_9662;
                }
                y_flag_9659 = new_flag_9661;
                x_9660 = res_9665;
            }
            // write final result
            {
                *(volatile __local bool *) &mem_10129[local_tid_9541 *
                                                      sizeof(bool)] =
                    y_flag_9659;
                *(volatile __local float *) &mem_10132[local_tid_9541 *
                                                       sizeof(float)] = x_9660;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_9541, 32) == 0) {
            *(volatile __local bool *) &mem_10129[local_tid_9541 *
                                                  sizeof(bool)] = y_flag_9659;
            *(volatile __local float *) &mem_10132[local_tid_9541 *
                                                   sizeof(float)] = x_9660;
        }
    }
    
    int32_t redoffset_9666;
    float red_res_9667;
    
    if (thread_active_10393) {
        if (isactive_9633) {
            int32_t x_9668;
            int32_t y_9669;
            int32_t segment_index_9670;
            int32_t y_9672;
            bool islastinseg_9673;
            int32_t redoffset_9674;
            float red_return_elem_9675;
            
            x_9668 = squot32(local_tid_9541, num_groups_per_segment_9447);
            y_9669 = group_id_9542 * num_segments_per_group_9618;
            segment_index_9670 = x_9668 + y_9669;
            y_9672 = num_groups_per_segment_9447 - 1;
            islastinseg_9673 = x_9649 == y_9672;
            if (islastinseg_9673) {
                redoffset_9674 = segment_index_9670;
            } else {
                redoffset_9674 = -1;
            }
            if (islastinseg_9673) {
                float x_9676 = *(__local float *) &mem_10132[local_tid_9541 *
                                                             4];
                
                red_return_elem_9675 = x_9676;
            } else {
                red_return_elem_9675 = 0.0F;
            }
            redoffset_9666 = redoffset_9674;
            red_res_9667 = red_return_elem_9675;
        } else {
            redoffset_9666 = -1;
            red_res_9667 = 0.0F;
        }
    }
    if (thread_active_10393 && (sle32(0, redoffset_9666) &&
                                slt32(redoffset_9666, x_8818))) {
        *(__global float *) &mem_10127[redoffset_9666 * 4] = red_res_9667;
    }
}
__kernel void segmented_redomap__small_comm_kernel_9694(__local volatile
                                                        int64_t *mem_aligned_0,
                                                        __local volatile
                                                        int64_t *mem_aligned_1,
                                                        int32_t num_elems_8798,
                                                        int32_t flat_dim_8801,
                                                        int32_t x_8818,
                                                        int32_t num_segments_per_group_9683,
                                                        int32_t active_threads_per_group_9688,
                                                        int32_t active_threads_last_group_9693,
                                                        int32_t y_9697, __global
                                                        unsigned char *mem_10139,
                                                        __global
                                                        unsigned char *mem_10144)
{
    __local volatile char *restrict mem_10146 = mem_aligned_0;
    __local volatile char *restrict mem_10149 = mem_aligned_1;
    int32_t wave_sizze_10409;
    int32_t group_sizze_10410;
    bool thread_active_10411;
    int32_t global_tid_9694;
    int32_t local_tid_9695;
    int32_t group_id_9696;
    
    global_tid_9694 = get_global_id(0);
    local_tid_9695 = get_local_id(0);
    group_sizze_10410 = get_local_size(0);
    wave_sizze_10409 = LOCKSTEP_WIDTH;
    group_id_9696 = get_group_id(0);
    thread_active_10411 = 1;
    
    bool islastgroup_9698;
    int32_t active_thread_this_group_9699;
    bool isactive_9700;
    float redtmp_res_9710;
    int32_t x_9711;
    bool isfirstinsegment_9712;
    
    if (thread_active_10411) {
        islastgroup_9698 = group_id_9696 == y_9697;
        if (islastgroup_9698) {
            active_thread_this_group_9699 = active_threads_last_group_9693;
        } else {
            active_thread_this_group_9699 = active_threads_per_group_9688;
        }
        isactive_9700 = slt32(local_tid_9695, active_thread_this_group_9699);
        if (isactive_9700) {
            int32_t x_9701;
            int32_t y_9702;
            int32_t segment_index_9703;
            int32_t index_within_segment_9704;
            int32_t y_9705;
            int32_t offset_9706;
            int32_t binop_y_9913;
            int32_t new_index_9914;
            int32_t binop_y_9918;
            int32_t binop_x_9919;
            int32_t new_index_9920;
            int32_t binop_y_9932;
            int32_t new_index_9933;
            float x_9680;
            
            x_9701 = squot32(local_tid_9695, flat_dim_8801);
            y_9702 = num_segments_per_group_9683 * group_id_9696;
            segment_index_9703 = x_9701 + y_9702;
            index_within_segment_9704 = srem32(local_tid_9695, flat_dim_8801);
            y_9705 = flat_dim_8801 * segment_index_9703;
            offset_9706 = index_within_segment_9704 + y_9705;
            binop_y_9913 = 2 * num_elems_8798;
            new_index_9914 = squot32(offset_9706, binop_y_9913);
            binop_y_9918 = binop_y_9913 * new_index_9914;
            binop_x_9919 = offset_9706 - binop_y_9918;
            new_index_9920 = squot32(binop_x_9919, 2);
            binop_y_9932 = 2 * new_index_9920;
            new_index_9933 = binop_x_9919 - binop_y_9932;
            x_9680 = *(__global float *) &mem_10144[(new_index_9914 *
                                                     (num_elems_8798 * 2) +
                                                     new_index_9920 * 2 +
                                                     new_index_9933) * 4];
            redtmp_res_9710 = x_9680;
        } else {
            redtmp_res_9710 = 0.0F;
        }
        x_9711 = srem32(local_tid_9695, flat_dim_8801);
        isfirstinsegment_9712 = x_9711 == 0;
    }
    for (int32_t comb_iter_10412 = 0; comb_iter_10412 <
         squot32(group_sizze_9363 + group_sizze_9363 - 1, group_sizze_9363);
         comb_iter_10412++) {
        int32_t cid_9715;
        int32_t flat_comb_id_10413 = comb_iter_10412 * group_sizze_9363 +
                local_tid_9695;
        
        cid_9715 = flat_comb_id_10413;
        if (slt32(cid_9715, group_sizze_9363) && 1) {
            *(__local bool *) &mem_10146[cid_9715] = isfirstinsegment_9712;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t comb_iter_10414 = 0; comb_iter_10414 <
         squot32(group_sizze_9363 + group_sizze_9363 - 1, group_sizze_9363);
         comb_iter_10414++) {
        int32_t cid_9716;
        int32_t flat_comb_id_10415 = comb_iter_10414 * group_sizze_9363 +
                local_tid_9695;
        
        cid_9716 = flat_comb_id_10415;
        if (slt32(cid_9716, group_sizze_9363) && 1) {
            *(__local float *) &mem_10149[cid_9716 * 4] = redtmp_res_9710;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float x_8938;
    float x_8939;
    int32_t my_index_9353;
    int32_t other_offset_9354;
    bool x_flag_9355;
    bool y_flag_9356;
    int32_t my_index_10416;
    int32_t other_offset_10417;
    bool x_flag_10418;
    float x_10419;
    bool y_flag_10420;
    float x_10421;
    
    my_index_9353 = local_tid_9695;
    if (slt32(local_tid_9695, group_sizze_9363)) {
        y_flag_9356 = *(volatile __local bool *) &mem_10146[local_tid_9695 *
                                                            sizeof(bool)];
        x_8939 = *(volatile __local float *) &mem_10149[local_tid_9695 *
                                                        sizeof(float)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        int32_t skip_threads_10425 = 1;
        
        while (slt32(skip_threads_10425, 32)) {
            if (slt32(local_tid_9695, group_sizze_9363) &&
                sle32(skip_threads_10425, local_tid_9695 -
                      squot32(local_tid_9695, 32) * 32)) {
                // read operands
                {
                    x_flag_9355 = *(volatile __local
                                    bool *) &mem_10146[(local_tid_9695 -
                                                        skip_threads_10425) *
                                                       sizeof(bool)];
                    x_8938 = *(volatile __local
                               float *) &mem_10149[(local_tid_9695 -
                                                    skip_threads_10425) *
                                                   sizeof(float)];
                }
                // perform operation
                {
                    bool new_flag_9357;
                    float seg_lhs_9358;
                    float res_9361;
                    
                    if (thread_active_10411) {
                        new_flag_9357 = x_flag_9355 || y_flag_9356;
                        if (y_flag_9356) {
                            seg_lhs_9358 = 0.0F;
                        } else {
                            seg_lhs_9358 = x_8938;
                        }
                        res_9361 = x_8939 + seg_lhs_9358;
                    }
                    y_flag_9356 = new_flag_9357;
                    x_8939 = res_9361;
                }
            }
            if (sle32(wave_sizze_10409, skip_threads_10425)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (slt32(local_tid_9695, group_sizze_9363) &&
                sle32(skip_threads_10425, local_tid_9695 -
                      squot32(local_tid_9695, 32) * 32)) {
                // write result
                {
                    *(volatile __local bool *) &mem_10146[local_tid_9695 *
                                                          sizeof(bool)] =
                        y_flag_9356;
                    *(volatile __local float *) &mem_10149[local_tid_9695 *
                                                           sizeof(float)] =
                        x_8939;
                }
            }
            if (sle32(wave_sizze_10409, skip_threads_10425)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_10425 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_9695 - squot32(local_tid_9695, 32) * 32) == 31 &&
            slt32(local_tid_9695, group_sizze_9363)) {
            *(volatile __local bool *) &mem_10146[squot32(local_tid_9695, 32) *
                                                  sizeof(bool)] = y_flag_9356;
            *(volatile __local float *) &mem_10149[squot32(local_tid_9695, 32) *
                                                   sizeof(float)] = x_8939;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_tid_9695, 32) == 0 && slt32(local_tid_9695,
                                                      group_sizze_9363)) {
            y_flag_10420 = *(volatile __local
                             bool *) &mem_10146[local_tid_9695 * sizeof(bool)];
            x_10421 = *(volatile __local float *) &mem_10149[local_tid_9695 *
                                                             sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            int32_t skip_threads_10426 = 1;
            
            while (slt32(skip_threads_10426, 32)) {
                if ((squot32(local_tid_9695, 32) == 0 && slt32(local_tid_9695,
                                                               group_sizze_9363)) &&
                    sle32(skip_threads_10426, local_tid_9695 -
                          squot32(local_tid_9695, 32) * 32)) {
                    // read operands
                    {
                        x_flag_10418 = *(volatile __local
                                         bool *) &mem_10146[(local_tid_9695 -
                                                             skip_threads_10426) *
                                                            sizeof(bool)];
                        x_10419 = *(volatile __local
                                    float *) &mem_10149[(local_tid_9695 -
                                                         skip_threads_10426) *
                                                        sizeof(float)];
                    }
                    // perform operation
                    {
                        bool new_flag_10422;
                        float seg_lhs_10423;
                        float res_10424;
                        
                        if (thread_active_10411) {
                            new_flag_10422 = x_flag_10418 || y_flag_10420;
                            if (y_flag_10420) {
                                seg_lhs_10423 = 0.0F;
                            } else {
                                seg_lhs_10423 = x_10419;
                            }
                            res_10424 = x_10421 + seg_lhs_10423;
                        }
                        y_flag_10420 = new_flag_10422;
                        x_10421 = res_10424;
                    }
                }
                if (sle32(wave_sizze_10409, skip_threads_10426)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if ((squot32(local_tid_9695, 32) == 0 && slt32(local_tid_9695,
                                                               group_sizze_9363)) &&
                    sle32(skip_threads_10426, local_tid_9695 -
                          squot32(local_tid_9695, 32) * 32)) {
                    // write result
                    {
                        *(volatile __local bool *) &mem_10146[local_tid_9695 *
                                                              sizeof(bool)] =
                            y_flag_10420;
                        *(volatile __local float *) &mem_10149[local_tid_9695 *
                                                               sizeof(float)] =
                            x_10421;
                    }
                }
                if (sle32(wave_sizze_10409, skip_threads_10426)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_10426 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_9695, 32) == 0 || !slt32(local_tid_9695,
                                                         group_sizze_9363))) {
            // read operands
            {
                x_flag_9355 = *(volatile __local
                                bool *) &mem_10146[(squot32(local_tid_9695,
                                                            32) - 1) *
                                                   sizeof(bool)];
                x_8938 = *(volatile __local
                           float *) &mem_10149[(squot32(local_tid_9695, 32) -
                                                1) * sizeof(float)];
            }
            // perform operation
            {
                bool new_flag_9357;
                float seg_lhs_9358;
                float res_9361;
                
                if (thread_active_10411) {
                    new_flag_9357 = x_flag_9355 || y_flag_9356;
                    if (y_flag_9356) {
                        seg_lhs_9358 = 0.0F;
                    } else {
                        seg_lhs_9358 = x_8938;
                    }
                    res_9361 = x_8939 + seg_lhs_9358;
                }
                y_flag_9356 = new_flag_9357;
                x_8939 = res_9361;
            }
            // write final result
            {
                *(volatile __local bool *) &mem_10146[local_tid_9695 *
                                                      sizeof(bool)] =
                    y_flag_9356;
                *(volatile __local float *) &mem_10149[local_tid_9695 *
                                                       sizeof(float)] = x_8939;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_9695, 32) == 0) {
            *(volatile __local bool *) &mem_10146[local_tid_9695 *
                                                  sizeof(bool)] = y_flag_9356;
            *(volatile __local float *) &mem_10149[local_tid_9695 *
                                                   sizeof(float)] = x_8939;
        }
    }
    
    int32_t redoffset_9728;
    float red_res_9729;
    
    if (thread_active_10411) {
        if (isactive_9700) {
            int32_t x_9719;
            int32_t y_9720;
            int32_t segment_index_9721;
            int32_t y_9723;
            bool islastinseg_9724;
            int32_t redoffset_9725;
            float red_return_elem_9727;
            
            x_9719 = squot32(local_tid_9695, flat_dim_8801);
            y_9720 = num_segments_per_group_9683 * group_id_9696;
            segment_index_9721 = x_9719 + y_9720;
            y_9723 = flat_dim_8801 - 1;
            islastinseg_9724 = x_9711 == y_9723;
            if (islastinseg_9724) {
                redoffset_9725 = segment_index_9721;
            } else {
                redoffset_9725 = -1;
            }
            if (islastinseg_9724) {
                float x_9726 = *(__local float *) &mem_10149[local_tid_9695 *
                                                             4];
                
                red_return_elem_9727 = x_9726;
            } else {
                red_return_elem_9727 = 0.0F;
            }
            redoffset_9728 = redoffset_9725;
            red_res_9729 = red_return_elem_9727;
        } else {
            redoffset_9728 = -1;
            red_res_9729 = 0.0F;
        }
    }
    if (thread_active_10411 && (sle32(0, redoffset_9728) &&
                                slt32(redoffset_9728, x_8818))) {
        *(__global float *) &mem_10139[redoffset_9728 * 4] = red_res_9729;
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
class forwardprojection_dpintegrated:
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
                                       all_sizes={"group_size_8963": {"class": "group_size", "value": None},
                                        "group_size_8979": {"class": "group_size", "value": None},
                                        "group_size_8991": {"class": "group_size", "value": None},
                                        "group_size_9033": {"class": "group_size", "value": None},
                                        "group_size_9130": {"class": "group_size", "value": None},
                                        "max_num_groups_9132": {"class": "num_groups", "value": None},
                                        "group_size_9280": {"class": "group_size", "value": None},
                                        "group_size_9290": {"class": "group_size", "value": None},
                                        "group_size_9362": {"class": "group_size", "value": None},
                                        "num_groups_hint_9364": {"class": "num_groups", "value": None},
                                        "num_groups_hint_9433": {"class": "num_groups", "value": None},
                                        "group_size_9496": {"class": "group_size", "value": None},
                                        "group_size_9745": {"class": "group_size", "value": None},
                                        "group_size_9836": {"class": "group_size", "value": None},
                                        "group_size_10184": {"class": "group_size", "value": None},
                                        "max_num_groups_10186": {"class": "num_groups", "value": None},
                                        "group_size_10349": {"class": "group_size", "value": None}})
    self.chunked_reduce_kernel_10201_var = program.chunked_reduce_kernel_10201
    self.fut_kernel_map_transpose_f32_var = program.fut_kernel_map_transpose_f32
    self.fut_kernel_map_transpose_lowheight_f32_var = program.fut_kernel_map_transpose_lowheight_f32
    self.fut_kernel_map_transpose_lowwidth_f32_var = program.fut_kernel_map_transpose_lowwidth_f32
    self.fut_kernel_map_transpose_small_f32_var = program.fut_kernel_map_transpose_small_f32
    self.kernel_replicate_8791_var = program.kernel_replicate_8791
    self.map_kernel_8969_var = program.map_kernel_8969
    self.map_kernel_8985_var = program.map_kernel_8985
    self.map_kernel_8997_var = program.map_kernel_8997
    self.map_kernel_9039_var = program.map_kernel_9039
    self.map_kernel_9286_var = program.map_kernel_9286
    self.map_kernel_9296_var = program.map_kernel_9296
    self.map_kernel_9751_var = program.map_kernel_9751
    self.map_kernel_9842_var = program.map_kernel_9842
    self.reduce_kernel_10231_var = program.reduce_kernel_10231
    self.scan1_kernel_9185_var = program.scan1_kernel_9185
    self.scan2_kernel_9243_var = program.scan2_kernel_9243
    self.segmented_redomap__large_comm_many_kernel_9453_var = program.segmented_redomap__large_comm_many_kernel_9453
    self.segmented_redomap__large_comm_one_kernel_9390_var = program.segmented_redomap__large_comm_one_kernel_9390
    self.segmented_redomap__large_comm_one_kernel_9509_var = program.segmented_redomap__large_comm_one_kernel_9509
    self.segmented_redomap__small_comm_kernel_9540_var = program.segmented_redomap__small_comm_kernel_9540
    self.segmented_redomap__small_comm_kernel_9694_var = program.segmented_redomap__small_comm_kernel_9694
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
  def futhark_main(self, angles_mem_sizze_9962, angles_mem_9963,
                   rays_mem_sizze_9964, rays_mem_9965, voxels_mem_sizze_9966,
                   voxels_mem_9967, sizze_8688, sizze_8689, sizze_8690,
                   stepsizze_8694):
    res_8695 = sitofp_i32_f32(sizze_8690)
    res_8696 = futhark_sqrt32(res_8695)
    res_8697 = fptosi_f32_i32(res_8696)
    res_8698 = sdiv32(res_8697, np.int32(2))
    res_8699 = sitofp_i32_f32(res_8698)
    group_sizze_8992 = self.sizes["group_size_8991"]
    y_8993 = (group_sizze_8992 - np.int32(1))
    x_8994 = (sizze_8688 + y_8993)
    num_groups_8995 = squot32(x_8994, group_sizze_8992)
    num_threads_8996 = (group_sizze_8992 * num_groups_8995)
    binop_x_9969 = sext_i32_i64(sizze_8688)
    bytes_9968 = (np.int64(4) * binop_x_9969)
    mem_9970 = opencl_alloc(self, bytes_9968, "mem_9970")
    mem_9973 = opencl_alloc(self, bytes_9968, "mem_9973")
    if ((1 * (num_groups_8995 * group_sizze_8992)) != 0):
      self.map_kernel_8997_var.set_args(np.int32(sizze_8688), angles_mem_9963,
                                        mem_9970, mem_9973)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8997_var,
                                 (np.long((num_groups_8995 * group_sizze_8992)),),
                                 (np.long(group_sizze_8992),))
      if synchronous:
        self.queue.finish()
    nesting_sizze_8978 = (sizze_8688 * sizze_8689)
    group_sizze_8980 = self.sizes["group_size_8979"]
    y_8981 = (group_sizze_8980 - np.int32(1))
    x_8982 = (nesting_sizze_8978 + y_8981)
    num_groups_8983 = squot32(x_8982, group_sizze_8980)
    num_threads_8984 = (group_sizze_8980 * num_groups_8983)
    binop_x_9976 = sext_i32_i64(nesting_sizze_8978)
    bytes_9974 = (np.int64(4) * binop_x_9976)
    mem_9977 = opencl_alloc(self, bytes_9974, "mem_9977")
    if ((1 * (num_groups_8983 * group_sizze_8980)) != 0):
      self.map_kernel_8985_var.set_args(np.int32(sizze_8688),
                                        np.int32(sizze_8689), mem_9970,
                                        mem_9977)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8985_var,
                                 (np.long((num_groups_8983 * group_sizze_8980)),),
                                 (np.long(group_sizze_8980),))
      if synchronous:
        self.queue.finish()
    mem_9970 = None
    group_sizze_8964 = self.sizes["group_size_8963"]
    y_8965 = (group_sizze_8964 - np.int32(1))
    x_8966 = (y_8965 + nesting_sizze_8978)
    num_groups_8967 = squot32(x_8966, group_sizze_8964)
    num_threads_8968 = (group_sizze_8964 * num_groups_8967)
    mem_9981 = opencl_alloc(self, bytes_9974, "mem_9981")
    if ((1 * (num_groups_8967 * group_sizze_8964)) != 0):
      self.map_kernel_8969_var.set_args(np.int32(sizze_8688),
                                        np.int32(sizze_8689), mem_9973,
                                        mem_9981)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8969_var,
                                 (np.long((num_groups_8967 * group_sizze_8964)),),
                                 (np.long(group_sizze_8964),))
      if synchronous:
        self.queue.finish()
    mem_9973 = None
    res_8712 = (np.float32(0.0) - res_8699)
    group_sizze_9291 = self.sizes["group_size_9290"]
    y_9292 = (group_sizze_9291 - np.int32(1))
    x_9293 = (nesting_sizze_8978 + y_9292)
    num_groups_9294 = squot32(x_9293, group_sizze_9291)
    num_threads_9295 = (group_sizze_9291 * num_groups_9294)
    mem_9991 = opencl_alloc(self, binop_x_9976, "mem_9991")
    convop_x_9993 = (np.int32(4) * nesting_sizze_8978)
    binop_x_9994 = sext_i32_i64(convop_x_9993)
    bytes_9992 = (np.int64(4) * binop_x_9994)
    mem_9995 = opencl_alloc(self, bytes_9992, "mem_9995")
    mem_9999 = opencl_alloc(self, bytes_9992, "mem_9999")
    mem_10002 = opencl_alloc(self, binop_x_9994, "mem_10002")
    num_threads64_10164 = sext_i32_i64(num_threads_9295)
    total_sizze_10165 = (np.int64(16) * num_threads64_10164)
    mem_9984 = opencl_alloc(self, total_sizze_10165, "mem_9984")
    total_sizze_10166 = (np.int64(16) * num_threads64_10164)
    mem_9987 = opencl_alloc(self, total_sizze_10166, "mem_9987")
    total_sizze_10167 = (np.int64(4) * num_threads64_10164)
    mem_9989 = opencl_alloc(self, total_sizze_10167, "mem_9989")
    if ((1 * (num_groups_9294 * group_sizze_9291)) != 0):
      self.map_kernel_9296_var.set_args(np.int32(sizze_8689),
                                        np.float32(res_8699),
                                        np.float32(res_8712),
                                        np.int32(nesting_sizze_8978),
                                        rays_mem_9965, mem_9977, mem_9981,
                                        mem_9984, mem_9987, mem_9989, mem_9991,
                                        mem_9995, mem_9999, mem_10002)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9296_var,
                                 (np.long((num_groups_9294 * group_sizze_9291)),),
                                 (np.long(group_sizze_9291),))
      if synchronous:
        self.queue.finish()
    mem_9981 = None
    mem_9984 = None
    mem_9987 = None
    mem_9989 = None
    group_sizze_9131 = self.sizes["group_size_9130"]
    max_num_groups_9133 = self.sizes["max_num_groups_9132"]
    y_9134 = (group_sizze_9131 - np.int32(1))
    x_9135 = (y_9134 + convop_x_9993)
    w_div_group_sizze_9136 = squot32(x_9135, group_sizze_9131)
    num_groups_maybe_zzero_9137 = smin32(max_num_groups_9133,
                                         w_div_group_sizze_9136)
    num_groups_9138 = smax32(np.int32(1), num_groups_maybe_zzero_9137)
    num_threads_9139 = (group_sizze_9131 * num_groups_9138)
    mem_10004 = opencl_alloc(self, binop_x_9994, "mem_10004")
    mem_10007 = opencl_alloc(self, bytes_9992, "mem_10007")
    mem_10010 = opencl_alloc(self, bytes_9992, "mem_10010")
    y_9188 = (num_threads_9139 - np.int32(1))
    x_9189 = (y_9188 + convop_x_9993)
    num_iterations_9190 = squot32(x_9189, num_threads_9139)
    y_9194 = (group_sizze_9131 * num_iterations_9190)
    mem_10013 = opencl_alloc(self, binop_x_9994, "mem_10013")
    if (((nesting_sizze_8978 * np.int32(4)) * np.int32(1)) != 0):
      cl.enqueue_copy(self.queue, mem_10013, mem_10002,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(0)),
                      byte_count=np.long(((nesting_sizze_8978 * np.int32(4)) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    mem_10002 = None
    bytes_10028 = sext_i32_i64(num_groups_9138)
    mem_10029 = opencl_alloc(self, bytes_10028, "mem_10029")
    bytes_10030 = (np.int64(4) * bytes_10028)
    mem_10032 = opencl_alloc(self, bytes_10030, "mem_10032")
    bytes_10016 = sext_i32_i64(group_sizze_9131)
    bytes_10018 = (np.int64(4) * bytes_10016)
    if ((1 * (num_groups_9138 * group_sizze_9131)) != 0):
      self.scan1_kernel_9185_var.set_args(cl.LocalMemory(np.long(bytes_10016)),
                                          cl.LocalMemory(np.long(bytes_10018)),
                                          np.int32(num_iterations_9190),
                                          np.int32(y_9194),
                                          np.int32(convop_x_9993), mem_10004,
                                          mem_10007, mem_10010, mem_10013,
                                          mem_10029, mem_10032)
      cl.enqueue_nd_range_kernel(self.queue, self.scan1_kernel_9185_var,
                                 (np.long((num_groups_9138 * group_sizze_9131)),),
                                 (np.long(group_sizze_9131),))
      if synchronous:
        self.queue.finish()
    mem_10013 = None
    mem_10017 = None
    mem_10020 = None
    mem_10039 = opencl_alloc(self, bytes_10028, "mem_10039")
    mem_10042 = opencl_alloc(self, bytes_10030, "mem_10042")
    if ((1 * num_groups_9138) != 0):
      self.scan2_kernel_9243_var.set_args(cl.LocalMemory(np.long(bytes_10028)),
                                          cl.LocalMemory(np.long(bytes_10030)),
                                          np.int32(num_groups_9138), mem_10029,
                                          mem_10032, mem_10039, mem_10042)
      cl.enqueue_nd_range_kernel(self.queue, self.scan2_kernel_9243_var,
                                 (np.long(num_groups_9138),),
                                 (np.long(num_groups_9138),))
      if synchronous:
        self.queue.finish()
    mem_10029 = None
    mem_10032 = None
    mem_10034 = None
    mem_10037 = None
    group_sizze_9281 = self.sizes["group_size_9280"]
    y_9282 = (group_sizze_9281 - np.int32(1))
    x_9283 = (y_9282 + convop_x_9993)
    num_groups_9284 = squot32(x_9283, group_sizze_9281)
    num_threads_9285 = (group_sizze_9281 * num_groups_9284)
    mem_10044 = opencl_alloc(self, binop_x_9994, "mem_10044")
    mem_10047 = opencl_alloc(self, bytes_9992, "mem_10047")
    if ((1 * (num_groups_9284 * group_sizze_9281)) != 0):
      self.map_kernel_9286_var.set_args(np.int32(y_9194),
                                        np.int32(convop_x_9993), mem_10004,
                                        mem_10007, mem_10039, mem_10042,
                                        mem_10044, mem_10047)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9286_var,
                                 (np.long((num_groups_9284 * group_sizze_9281)),),
                                 (np.long(group_sizze_9281),))
      if synchronous:
        self.queue.finish()
    mem_10004 = None
    mem_10007 = None
    mem_10039 = None
    mem_10042 = None
    mem_10044 = None
    group_sizze_9034 = self.sizes["group_size_9033"]
    y_9035 = (group_sizze_9034 - np.int32(1))
    x_9036 = (nesting_sizze_8978 + y_9035)
    num_groups_9037 = squot32(x_9036, group_sizze_9034)
    num_threads_9038 = (group_sizze_9034 * num_groups_9037)
    mem_10062 = opencl_alloc(self, bytes_9974, "mem_10062")
    mem_10065 = opencl_alloc(self, bytes_9974, "mem_10065")
    mem_10068 = opencl_alloc(self, bytes_9974, "mem_10068")
    mem_10071 = opencl_alloc(self, bytes_9974, "mem_10071")
    space_sizze_10176 = nesting_sizze_8978
    num_threads_10177 = sext_i32_i64(num_threads_9038)
    group_sizze_10185 = self.sizes["group_size_10184"]
    max_num_groups_10187 = self.sizes["max_num_groups_10186"]
    y_10188 = (group_sizze_10185 - np.int32(1))
    x_10189 = (nesting_sizze_8978 + y_10188)
    w_div_group_sizze_10190 = squot32(x_10189, group_sizze_10185)
    num_groups_maybe_zzero_10191 = smin32(max_num_groups_10187,
                                          w_div_group_sizze_10190)
    num_groups_10192 = smax32(np.int32(1), num_groups_maybe_zzero_10191)
    num_threads_10193 = (group_sizze_10185 * num_groups_10192)
    y_10194 = (num_threads_10193 - np.int32(1))
    x_10195 = (nesting_sizze_8978 + y_10194)
    per_thread_elements_10196 = squot32(x_10195, num_threads_10193)
    binop_x_10245 = sext_i32_i64(num_groups_10192)
    bytes_10244 = (np.int64(8) * binop_x_10245)
    mem_10246 = opencl_alloc(self, bytes_10244, "mem_10246")
    binop_x_10242 = sext_i32_i64(group_sizze_10185)
    bytes_10241 = (np.int64(8) * binop_x_10242)
    if ((1 * (num_groups_10192 * group_sizze_10185)) != 0):
      self.chunked_reduce_kernel_10201_var.set_args(cl.LocalMemory(np.long(bytes_10241)),
                                                    np.int32(nesting_sizze_8978),
                                                    mem_10047,
                                                    np.int32(num_threads_10193),
                                                    np.int32(per_thread_elements_10196),
                                                    mem_10246)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.chunked_reduce_kernel_10201_var,
                                 (np.long((num_groups_10192 * group_sizze_10185)),),
                                 (np.long(group_sizze_10185),))
      if synchronous:
        self.queue.finish()
    mem_10243 = None
    mem_10252 = opencl_alloc(self, np.int64(8), "mem_10252")
    binop_x_10248 = sext_i32_i64(max_num_groups_10187)
    bytes_10247 = (np.int64(8) * binop_x_10248)
    if ((1 * max_num_groups_10187) != 0):
      self.reduce_kernel_10231_var.set_args(cl.LocalMemory(np.long(bytes_10247)),
                                            np.int32(num_groups_10192),
                                            mem_10246, mem_10252)
      cl.enqueue_nd_range_kernel(self.queue, self.reduce_kernel_10231_var,
                                 (np.long(max_num_groups_10187),),
                                 (np.long(max_num_groups_10187),))
      if synchronous:
        self.queue.finish()
    mem_10246 = None
    mem_10249 = None
    read_res_10428 = np.empty(1, dtype=ct.c_int64)
    cl.enqueue_copy(self.queue, read_res_10428, mem_10252,
                    device_offset=np.long(np.int32(0)), is_blocking=True)
    max_per_thread_10178 = read_res_10428[0]
    mem_10252 = None
    sizze_sum_10240 = (num_threads_10177 * max_per_thread_10178)
    mem_10053 = opencl_alloc(self, sizze_sum_10240, "mem_10053")
    mem_10050 = opencl_alloc(self, sizze_sum_10240, "mem_10050")
    if ((1 * (num_groups_9037 * group_sizze_9034)) != 0):
      self.map_kernel_9039_var.set_args(np.int32(sizze_8689),
                                        np.float32(res_8699),
                                        np.float32(res_8712),
                                        np.int32(nesting_sizze_8978),
                                        np.int32(num_threads_9038),
                                        rays_mem_9965, mem_9977, mem_9991,
                                        mem_9995, mem_9999, mem_10010,
                                        mem_10047, mem_10050, mem_10053,
                                        mem_10062, mem_10065, mem_10068,
                                        mem_10071)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9039_var,
                                 (np.long((num_groups_9037 * group_sizze_9034)),),
                                 (np.long(group_sizze_9034),))
      if synchronous:
        self.queue.finish()
    mem_9977 = None
    mem_9991 = None
    mem_9995 = None
    mem_9999 = None
    mem_10010 = None
    mem_10047 = None
    mem_10050 = None
    mem_10053 = None
    res_8790 = sdiv32(nesting_sizze_8978, stepsizze_8694)
    mem_10074 = opencl_alloc(self, np.int64(4), "mem_10074")
    group_sizze_10349 = self.sizes["group_size_10349"]
    num_groups_10350 = squot32(((np.int32(1) + sext_i32_i32(group_sizze_10349)) - np.int32(1)),
                               sext_i32_i32(group_sizze_10349))
    if ((1 * (num_groups_10350 * group_sizze_10349)) != 0):
      self.kernel_replicate_8791_var.set_args(mem_10074)
      cl.enqueue_nd_range_kernel(self.queue, self.kernel_replicate_8791_var,
                                 (np.long((num_groups_10350 * group_sizze_10349)),),
                                 (np.long(group_sizze_10349),))
      if synchronous:
        self.queue.finish()
    loop_cond_8792 = slt32(np.int32(0), res_8790)
    range_start_8793 = (np.int32(0) - res_8698)
    range_end_8794 = (res_8698 - np.int32(1))
    bounds_invalid_upwards_8795 = slt32(range_end_8794, range_start_8793)
    distance_upwards_exclusive_8796 = (range_end_8794 - range_start_8793)
    distance_8797 = (np.int32(1) + distance_upwards_exclusive_8796)
    if bounds_invalid_upwards_8795:
      num_elems_8798 = np.int32(0)
    else:
      num_elems_8798 = distance_8797
    y_8800 = pow32(res_8697, np.int32(2))
    flat_dim_8801 = (np.int32(2) * num_elems_8798)
    group_sizze_9837 = self.sizes["group_size_9836"]
    y_9838 = (group_sizze_9837 - np.int32(1))
    group_sizze_9746 = self.sizes["group_size_9745"]
    y_9747 = (group_sizze_9746 - np.int32(1))
    group_sizze_9363 = self.sizes["group_size_9362"]
    num_groups_hint_9365 = self.sizes["num_groups_hint_9364"]
    x_9731 = squot32(group_sizze_9363, np.int32(2))
    cond_9732 = slt32(x_9731, flat_dim_8801)
    cond_neg_9940 = not(cond_9732)
    protect_cond_conj_9943 = (loop_cond_8792 and cond_neg_9940)
    if protect_cond_conj_9943:
      x_9941 = squot32(group_sizze_9363, flat_dim_8801)
      num_segments_per_group_9683 = x_9941
    else:
      num_segments_per_group_9683 = np.int32(0)
    y_9684 = (num_segments_per_group_9683 - np.int32(1))
    active_threads_per_group_9688 = (flat_dim_8801 * num_segments_per_group_9683)
    num_groups_hint_9434 = self.sizes["num_groups_hint_9433"]
    binop_x_10114 = sext_i32_i64(group_sizze_9363)
    bytes_10113 = (np.int64(4) * binop_x_10114)
    group_sizze_9497 = self.sizes["group_size_9496"]
    x_9577 = squot32(group_sizze_9497, np.int32(2))
    binop_x_10120 = sext_i32_i64(group_sizze_9497)
    bytes_10119 = (np.int64(4) * binop_x_10120)
    bytes_10128 = sext_i32_i64(group_sizze_9497)
    bytes_10130 = (np.int64(4) * bytes_10128)
    sizze_8806 = np.int32(1)
    output_mem_sizze_10075 = np.int64(4)
    output_mem_10076 = mem_10074
    loop_while_8807 = loop_cond_8792
    run_8809 = np.int32(0)
    while loop_while_8807:
      x_8810 = (np.int32(1) + run_8809)
      x_8811 = (stepsizze_8694 * x_8810)
      cond_8812 = sle32(nesting_sizze_8978, x_8811)
      if cond_8812:
        y_8814 = (stepsizze_8694 * run_8809)
        res_8815 = (nesting_sizze_8978 - y_8814)
        res_8813 = res_8815
      else:
        res_8813 = stepsizze_8694
      i_8816 = (stepsizze_8694 * run_8809)
      j_8817 = (res_8813 + i_8816)
      x_8818 = abs(res_8813)
      empty_slice_8819 = (x_8818 == np.int32(0))
      m_8820 = (x_8818 - np.int32(1))
      i_p_m_t_s_8821 = (i_8816 + m_8820)
      zzero_leq_i_p_m_t_s_8822 = sle32(np.int32(0), i_p_m_t_s_8821)
      i_p_m_t_s_leq_w_8823 = slt32(i_p_m_t_s_8821, nesting_sizze_8978)
      zzero_lte_i_8824 = sle32(np.int32(0), i_8816)
      i_lte_j_8825 = sle32(i_8816, j_8817)
      y_8826 = (i_p_m_t_s_leq_w_8823 and zzero_lte_i_8824)
      y_8827 = (zzero_leq_i_p_m_t_s_8822 and y_8826)
      y_8828 = (i_lte_j_8825 and y_8827)
      forwards_ok_8829 = (zzero_lte_i_8824 and y_8828)
      ok_or_empty_8830 = (empty_slice_8819 or forwards_ok_8829)
      index_certs_8831 = True
      assert ok_or_empty_8830, ("Error at forwardprojection_dpintegrated.fut:33:1-37:66 -> forwardprojection_dpintegrated.fut:37:11-37:66 -> projection_lib.fut:332:63-332:109: %s%d%s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                   i_8816,
                                                                                                                                                                                                   ":",
                                                                                                                                                                                                   j_8817,
                                                                                                                                                                                                   "] out of bounds for array of shape [",
                                                                                                                                                                                                   nesting_sizze_8978,
                                                                                                                                                                                                   "]."))
      x_9839 = (x_8818 + y_9838)
      num_groups_9840 = squot32(x_9839, group_sizze_9837)
      num_threads_9841 = (group_sizze_9837 * num_groups_9840)
      bytes_10077 = sext_i32_i64(x_8818)
      mem_10078 = opencl_alloc(self, bytes_10077, "mem_10078")
      bytes_10079 = (np.int64(4) * bytes_10077)
      mem_10081 = opencl_alloc(self, bytes_10079, "mem_10081")
      mem_10084 = opencl_alloc(self, bytes_10079, "mem_10084")
      mem_10087 = opencl_alloc(self, bytes_10079, "mem_10087")
      mem_10090 = opencl_alloc(self, bytes_10079, "mem_10090")
      if ((1 * (num_groups_9840 * group_sizze_9837)) != 0):
        self.map_kernel_9842_var.set_args(np.int32(i_8816), np.int32(x_8818),
                                          mem_10062, mem_10065, mem_10068,
                                          mem_10071, mem_10078, mem_10081,
                                          mem_10084, mem_10087, mem_10090)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9842_var,
                                   (np.long((num_groups_9840 * group_sizze_9837)),),
                                   (np.long(group_sizze_9837),))
        if synchronous:
          self.queue.finish()
      nesting_sizze_9744 = (num_elems_8798 * x_8818)
      x_9748 = (nesting_sizze_9744 + y_9747)
      num_groups_9749 = squot32(x_9748, group_sizze_9746)
      num_threads_9750 = (group_sizze_9746 * num_groups_9749)
      binop_x_10095 = (np.int32(2) * x_8818)
      convop_x_10096 = (num_elems_8798 * binop_x_10095)
      binop_x_10097 = sext_i32_i64(convop_x_10096)
      bytes_10094 = (np.int64(4) * binop_x_10097)
      mem_10098 = opencl_alloc(self, bytes_10094, "mem_10098")
      num_threads64_10258 = sext_i32_i64(num_threads_9750)
      total_sizze_10259 = (np.int64(8) * num_threads64_10258)
      mem_10093 = opencl_alloc(self, total_sizze_10259, "mem_10093")
      if ((1 * (num_groups_9749 * group_sizze_9746)) != 0):
        self.map_kernel_9751_var.set_args(np.int32(res_8697),
                                          np.int32(res_8698),
                                          np.float32(res_8699),
                                          np.int32(range_start_8793),
                                          np.int32(num_elems_8798),
                                          np.int32(y_8800), np.int32(x_8818),
                                          voxels_mem_9967, mem_10078, mem_10081,
                                          mem_10084, mem_10087, mem_10090,
                                          mem_10093, mem_10098)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9751_var,
                                   (np.long((num_groups_9749 * group_sizze_9746)),),
                                   (np.long(group_sizze_9746),))
        if synchronous:
          self.queue.finish()
      mem_10078 = None
      mem_10081 = None
      mem_10084 = None
      mem_10087 = None
      mem_10090 = None
      mem_10093 = None
      x_9367 = (m_8820 + num_groups_hint_9365)
      num_groups_per_segment_hint_9368 = squot32(x_9367, x_8818)
      x_9369 = (group_sizze_9363 * num_groups_per_segment_hint_9368)
      y_9370 = (x_9369 - np.int32(1))
      x_9371 = (flat_dim_8801 + y_9370)
      elements_per_thread_9373 = squot32(x_9371, x_9369)
      cond_9374 = (elements_per_thread_9373 == np.int32(1))
      if cond_9374:
        y_9375 = (group_sizze_9363 - np.int32(1))
        x_9376 = (flat_dim_8801 + y_9375)
        x_9377 = squot32(x_9376, group_sizze_9363)
        num_groups_per_segment_9378 = x_9377
      else:
        num_groups_per_segment_9378 = num_groups_per_segment_hint_9368
      cond_9733 = (num_groups_per_segment_9378 == np.int32(1))
      convop_x_10100 = (flat_dim_8801 * x_8818)
      binop_x_10101 = sext_i32_i64(convop_x_10100)
      bytes_10099 = (np.int64(4) * binop_x_10101)
      x_9436 = (m_8820 + num_groups_hint_9434)
      convop_x_10142 = (np.int32(2) * nesting_sizze_9744)
      binop_x_10143 = sext_i32_i64(convop_x_10142)
      bytes_10140 = (np.int64(4) * binop_x_10143)
      if cond_9732:
        if cond_9733:
          y_9382 = (group_sizze_9363 - np.int32(1))
          x_9383 = (flat_dim_8801 + y_9382)
          elements_per_thread_9385 = squot32(x_9383, group_sizze_9363)
          num_threads_9386 = (x_8818 * group_sizze_9363)
          mem_10102 = opencl_alloc(self, bytes_10099, "mem_10102")
          self.futhark_map_transpose_opencl_f32(mem_10102, np.int32(0),
                                                mem_10098, np.int32(0),
                                                np.int32(1),
                                                (x_8818 * num_elems_8798),
                                                np.int32(2),
                                                ((x_8818 * num_elems_8798) * np.int32(2)),
                                                (x_8818 * flat_dim_8801))
          mem_10108 = opencl_alloc(self, bytes_10079, "mem_10108")
          if ((1 * (x_8818 * group_sizze_9363)) != 0):
            self.segmented_redomap__large_comm_one_kernel_9390_var.set_args(cl.LocalMemory(np.long(bytes_10113)),
                                                                            np.int32(flat_dim_8801),
                                                                            np.int32(x_8818),
                                                                            np.int32(elements_per_thread_9385),
                                                                            mem_10102,
                                                                            mem_10108)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.segmented_redomap__large_comm_one_kernel_9390_var,
                                       (np.long((x_8818 * group_sizze_9363)),),
                                       (np.long(group_sizze_9363),))
            if synchronous:
              self.queue.finish()
          mem_10102 = None
          mem_10105 = None
          x_mem_10136 = mem_10108
        else:
          num_groups_per_segment_hint_9437 = squot32(x_9436, x_8818)
          x_9438 = (group_sizze_9363 * num_groups_per_segment_hint_9437)
          y_9439 = (x_9438 - np.int32(1))
          x_9440 = (flat_dim_8801 + y_9439)
          elements_per_thread_9442 = squot32(x_9440, x_9438)
          cond_9443 = (elements_per_thread_9442 == np.int32(1))
          if cond_9443:
            y_9444 = (group_sizze_9363 - np.int32(1))
            x_9445 = (flat_dim_8801 + y_9444)
            x_9446 = squot32(x_9445, group_sizze_9363)
            num_groups_per_segment_9447 = x_9446
          else:
            num_groups_per_segment_9447 = num_groups_per_segment_hint_9437
          num_groups_9448 = (x_8818 * num_groups_per_segment_9447)
          num_threads_9449 = (group_sizze_9363 * num_groups_9448)
          threads_within_segment_9450 = (group_sizze_9363 * num_groups_per_segment_9447)
          mem_10112 = opencl_alloc(self, bytes_10099, "mem_10112")
          self.futhark_map_transpose_opencl_f32(mem_10112, np.int32(0),
                                                mem_10098, np.int32(0),
                                                np.int32(1),
                                                (x_8818 * num_elems_8798),
                                                np.int32(2),
                                                ((x_8818 * num_elems_8798) * np.int32(2)),
                                                (x_8818 * flat_dim_8801))
          binop_x_10117 = sext_i32_i64(num_groups_9448)
          bytes_10116 = (np.int64(4) * binop_x_10117)
          mem_10118 = opencl_alloc(self, bytes_10116, "mem_10118")
          if ((1 * (num_groups_9448 * group_sizze_9363)) != 0):
            self.segmented_redomap__large_comm_many_kernel_9453_var.set_args(cl.LocalMemory(np.long(bytes_10113)),
                                                                             np.int32(flat_dim_8801),
                                                                             np.int32(x_8818),
                                                                             np.int32(elements_per_thread_9442),
                                                                             np.int32(num_groups_per_segment_9447),
                                                                             np.int32(threads_within_segment_9450),
                                                                             mem_10112,
                                                                             mem_10118)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.segmented_redomap__large_comm_many_kernel_9453_var,
                                       (np.long((num_groups_9448 * group_sizze_9363)),),
                                       (np.long(group_sizze_9363),))
            if synchronous:
              self.queue.finish()
          mem_10112 = None
          mem_10115 = None
          cond_9578 = slt32(x_9577, num_groups_per_segment_9447)
          if cond_9578:
            y_9582 = (group_sizze_9497 - np.int32(1))
            x_9583 = (num_groups_per_segment_9447 + y_9582)
            elements_per_thread_9585 = squot32(x_9583, group_sizze_9497)
            num_threads_9586 = (x_8818 * group_sizze_9497)
            mem_10124 = opencl_alloc(self, bytes_10079, "mem_10124")
            if ((1 * (x_8818 * group_sizze_9497)) != 0):
              self.segmented_redomap__large_comm_one_kernel_9509_var.set_args(cl.LocalMemory(np.long(bytes_10119)),
                                                                              np.int32(x_8818),
                                                                              np.int32(num_groups_per_segment_9447),
                                                                              np.int32(elements_per_thread_9585),
                                                                              mem_10118,
                                                                              mem_10124)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.segmented_redomap__large_comm_one_kernel_9509_var,
                                         (np.long((x_8818 * group_sizze_9497)),),
                                         (np.long(group_sizze_9497),))
              if synchronous:
                self.queue.finish()
            mem_10121 = None
            step_two_kernel_result_mem_10134 = mem_10124
          else:
            mem_10127 = opencl_alloc(self, bytes_10079, "mem_10127")
            num_segments_per_group_9618 = squot32(group_sizze_9497,
                                                  num_groups_per_segment_9447)
            y_9619 = (num_segments_per_group_9618 - np.int32(1))
            x_9620 = (x_8818 + y_9619)
            num_groups_9621 = squot32(x_9620, num_segments_per_group_9618)
            num_threads_9622 = (group_sizze_9497 * num_groups_9621)
            active_threads_per_group_9623 = (num_groups_per_segment_9447 * num_segments_per_group_9618)
            x_9624 = srem32(x_8818, num_segments_per_group_9618)
            cond_9625 = (x_9624 == np.int32(0))
            if cond_9625:
              seg_in_last_group_9626 = num_segments_per_group_9618
            else:
              seg_in_last_group_9626 = x_9624
            active_threads_last_group_9628 = (num_groups_per_segment_9447 * seg_in_last_group_9626)
            y_9630 = (num_groups_9621 - np.int32(1))
            if ((1 * (num_groups_9621 * group_sizze_9497)) != 0):
              self.segmented_redomap__small_comm_kernel_9540_var.set_args(cl.LocalMemory(np.long(bytes_10128)),
                                                                          cl.LocalMemory(np.long(bytes_10130)),
                                                                          np.int32(x_8818),
                                                                          np.int32(num_groups_per_segment_9447),
                                                                          np.int32(num_segments_per_group_9618),
                                                                          np.int32(active_threads_per_group_9623),
                                                                          np.int32(active_threads_last_group_9628),
                                                                          np.int32(y_9630),
                                                                          mem_10118,
                                                                          mem_10127)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.segmented_redomap__small_comm_kernel_9540_var,
                                         (np.long((num_groups_9621 * group_sizze_9497)),),
                                         (np.long(group_sizze_9497),))
              if synchronous:
                self.queue.finish()
            mem_10129 = None
            mem_10132 = None
            step_two_kernel_result_mem_10134 = mem_10127
          mem_10118 = None
          x_mem_10136 = step_two_kernel_result_mem_10134
        res_mem_10151 = x_mem_10136
      else:
        mem_10139 = opencl_alloc(self, bytes_10079, "mem_10139")
        x_9685 = (x_8818 + y_9684)
        num_groups_9686 = squot32(x_9685, num_segments_per_group_9683)
        num_threads_9687 = (group_sizze_9363 * num_groups_9686)
        x_9689 = srem32(x_8818, num_segments_per_group_9683)
        cond_9690 = (x_9689 == np.int32(0))
        if cond_9690:
          seg_in_last_group_9692 = num_segments_per_group_9683
        else:
          seg_in_last_group_9692 = x_9689
        active_threads_last_group_9693 = (flat_dim_8801 * seg_in_last_group_9692)
        y_9697 = (num_groups_9686 - np.int32(1))
        mem_10144 = opencl_alloc(self, bytes_10140, "mem_10144")
        self.futhark_map_transpose_opencl_f32(mem_10144, np.int32(0), mem_10098,
                                              np.int32(0), np.int32(1),
                                              (x_8818 * num_elems_8798),
                                              np.int32(2),
                                              ((x_8818 * num_elems_8798) * np.int32(2)),
                                              ((x_8818 * num_elems_8798) * np.int32(2)))
        if ((1 * (num_groups_9686 * group_sizze_9363)) != 0):
          self.segmented_redomap__small_comm_kernel_9694_var.set_args(cl.LocalMemory(np.long(binop_x_10114)),
                                                                      cl.LocalMemory(np.long(bytes_10113)),
                                                                      np.int32(num_elems_8798),
                                                                      np.int32(flat_dim_8801),
                                                                      np.int32(x_8818),
                                                                      np.int32(num_segments_per_group_9683),
                                                                      np.int32(active_threads_per_group_9688),
                                                                      np.int32(active_threads_last_group_9693),
                                                                      np.int32(y_9697),
                                                                      mem_10139,
                                                                      mem_10144)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.segmented_redomap__small_comm_kernel_9694_var,
                                     (np.long((num_groups_9686 * group_sizze_9363)),),
                                     (np.long(group_sizze_9363),))
          if synchronous:
            self.queue.finish()
        mem_10144 = None
        mem_10146 = None
        mem_10149 = None
        res_mem_10151 = mem_10139
      mem_10098 = None
      conc_tmp_8942 = (sizze_8806 + x_8818)
      binop_x_10153 = sext_i32_i64(conc_tmp_8942)
      bytes_10152 = (np.int64(4) * binop_x_10153)
      mem_10154 = opencl_alloc(self, bytes_10152, "mem_10154")
      tmp_offs_10427 = np.int32(0)
      if ((sizze_8806 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10154, output_mem_10076,
                        dest_offset=np.long((tmp_offs_10427 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sizze_8806 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10427 = (tmp_offs_10427 + sizze_8806)
      if ((x_8818 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10154, res_mem_10151,
                        dest_offset=np.long((tmp_offs_10427 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((x_8818 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10427 = (tmp_offs_10427 + x_8818)
      res_mem_10151 = None
      loop_cond_8944 = slt32(x_8810, res_8790)
      sizze_tmp_10351 = conc_tmp_8942
      output_mem_sizze_tmp_10352 = bytes_10152
      output_mem_tmp_10353 = mem_10154
      loop_while_tmp_10354 = loop_cond_8944
      run_tmp_10356 = x_8810
      sizze_8806 = sizze_tmp_10351
      output_mem_sizze_10075 = output_mem_sizze_tmp_10352
      output_mem_10076 = output_mem_tmp_10353
      loop_while_8807 = loop_while_tmp_10354
      run_8809 = run_tmp_10356
    sizze_8802 = sizze_8806
    res_mem_sizze_10155 = output_mem_sizze_10075
    res_mem_10156 = output_mem_10076
    res_8803 = loop_while_8807
    res_8805 = run_8809
    mem_10062 = None
    mem_10065 = None
    mem_10068 = None
    mem_10071 = None
    mem_10074 = None
    j_m_i_8945 = (sizze_8802 - np.int32(1))
    x_8946 = abs(j_m_i_8945)
    empty_slice_8947 = (x_8946 == np.int32(0))
    m_8948 = (x_8946 - np.int32(1))
    i_p_m_t_s_8949 = (np.int32(1) + m_8948)
    zzero_leq_i_p_m_t_s_8950 = sle32(np.int32(0), i_p_m_t_s_8949)
    i_p_m_t_s_leq_w_8951 = slt32(i_p_m_t_s_8949, sizze_8802)
    i_lte_j_8952 = sle32(np.int32(1), sizze_8802)
    y_8953 = (zzero_leq_i_p_m_t_s_8950 and i_p_m_t_s_leq_w_8951)
    y_8954 = (i_lte_j_8952 and y_8953)
    ok_or_empty_8955 = (empty_slice_8947 or y_8954)
    index_certs_8956 = True
    assert ok_or_empty_8955, ("Error at forwardprojection_dpintegrated.fut:33:1-37:66 -> forwardprojection_dpintegrated.fut:37:11-37:66 -> projection_lib.fut:334:20-334:31 -> /futlib/array.fut:21:29-21:33: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                                             np.int32(1),
                                                                                                                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                                                                                                                             sizze_8802,
                                                                                                                                                                                                                             "]."))
    binop_x_10158 = sext_i32_i64(x_8946)
    bytes_10157 = (np.int64(4) * binop_x_10158)
    mem_10159 = opencl_alloc(self, bytes_10157, "mem_10159")
    if ((x_8946 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_10159, res_mem_10156,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(4)),
                      byte_count=np.long((x_8946 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    res_mem_10156 = None
    out_arrsizze_10267 = x_8946
    out_memsizze_10266 = bytes_10157
    out_mem_10265 = mem_10159
    return (out_memsizze_10266, out_mem_10265, out_arrsizze_10267)
  def main(self, angles_mem_9963_ext, rays_mem_9965_ext, voxels_mem_9967_ext,
           stepsizze_8694_ext):
    try:
      assert ((type(angles_mem_9963_ext) in [np.ndarray,
                                             cl.array.Array]) and (angles_mem_9963_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_8688 = np.int32(angles_mem_9963_ext.shape[0])
      angles_mem_sizze_9962 = np.int64(angles_mem_9963_ext.nbytes)
      if (type(angles_mem_9963_ext) == cl.array.Array):
        angles_mem_9963 = angles_mem_9963_ext.data
      else:
        angles_mem_9963 = opencl_alloc(self, angles_mem_sizze_9962,
                                       "angles_mem_9963")
        if (angles_mem_sizze_9962 != 0):
          cl.enqueue_copy(self.queue, angles_mem_9963,
                          normaliseArray(angles_mem_9963_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(angles_mem_9963_ext),
                                                                                                                            angles_mem_9963_ext))
    try:
      assert ((type(rays_mem_9965_ext) in [np.ndarray,
                                           cl.array.Array]) and (rays_mem_9965_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_8689 = np.int32(rays_mem_9965_ext.shape[0])
      rays_mem_sizze_9964 = np.int64(rays_mem_9965_ext.nbytes)
      if (type(rays_mem_9965_ext) == cl.array.Array):
        rays_mem_9965 = rays_mem_9965_ext.data
      else:
        rays_mem_9965 = opencl_alloc(self, rays_mem_sizze_9964, "rays_mem_9965")
        if (rays_mem_sizze_9964 != 0):
          cl.enqueue_copy(self.queue, rays_mem_9965,
                          normaliseArray(rays_mem_9965_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(rays_mem_9965_ext),
                                                                                                                            rays_mem_9965_ext))
    try:
      assert ((type(voxels_mem_9967_ext) in [np.ndarray,
                                             cl.array.Array]) and (voxels_mem_9967_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_8690 = np.int32(voxels_mem_9967_ext.shape[0])
      voxels_mem_sizze_9966 = np.int64(voxels_mem_9967_ext.nbytes)
      if (type(voxels_mem_9967_ext) == cl.array.Array):
        voxels_mem_9967 = voxels_mem_9967_ext.data
      else:
        voxels_mem_9967 = opencl_alloc(self, voxels_mem_sizze_9966,
                                       "voxels_mem_9967")
        if (voxels_mem_sizze_9966 != 0):
          cl.enqueue_copy(self.queue, voxels_mem_9967,
                          normaliseArray(voxels_mem_9967_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(voxels_mem_9967_ext),
                                                                                                                            voxels_mem_9967_ext))
    try:
      stepsizze_8694 = np.int32(ct.c_int32(stepsizze_8694_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(stepsizze_8694_ext),
                                                                                                                            stepsizze_8694_ext))
    (out_memsizze_10266, out_mem_10265,
     out_arrsizze_10267) = self.futhark_main(angles_mem_sizze_9962,
                                             angles_mem_9963,
                                             rays_mem_sizze_9964, rays_mem_9965,
                                             voxels_mem_sizze_9966,
                                             voxels_mem_9967, sizze_8688,
                                             sizze_8689, sizze_8690,
                                             stepsizze_8694)
    return cl.array.Array(self.queue, (out_arrsizze_10267,), ct.c_float,
                          data=out_mem_10265)