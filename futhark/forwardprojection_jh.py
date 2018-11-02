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
#define group_sizze_10326 (group_size_10325)
__kernel void kernel_replicate_9955(__global unsigned char *mem_10661)
{
    const uint replicate_gtid_9955 = get_global_id(0);
    
    if (replicate_gtid_9955 >= 1)
        return;
    *(__global float *) &mem_10661[replicate_gtid_9955 * 4] = 0.0F;
}
__kernel void map_kernel_10235(int32_t sizze_9901, int32_t sizze_9902, __global
                               unsigned char *mem_10644, __global
                               unsigned char *mem_10652)
{
    int32_t wave_sizze_10730;
    int32_t group_sizze_10731;
    bool thread_active_10732;
    int32_t gtid_10226;
    int32_t gtid_10227;
    int32_t global_tid_10235;
    int32_t local_tid_10236;
    int32_t group_id_10237;
    
    global_tid_10235 = get_global_id(0);
    local_tid_10236 = get_local_id(0);
    group_sizze_10731 = get_local_size(0);
    wave_sizze_10730 = LOCKSTEP_WIDTH;
    group_id_10237 = get_group_id(0);
    gtid_10226 = squot32(global_tid_10235, sizze_9902);
    gtid_10227 = global_tid_10235 - squot32(global_tid_10235, sizze_9902) *
        sizze_9902;
    thread_active_10732 = slt32(gtid_10226, sizze_9901) && slt32(gtid_10227,
                                                                 sizze_9902);
    
    float res_10238;
    
    if (thread_active_10732) {
        res_10238 = *(__global float *) &mem_10644[gtid_10226 * 4];
    }
    if (thread_active_10732) {
        *(__global float *) &mem_10652[(gtid_10226 * sizze_9902 + gtid_10227) *
                                       4] = res_10238;
    }
}
__kernel void map_kernel_10251(int32_t sizze_9901, int32_t sizze_9902, __global
                               unsigned char *mem_10641, __global
                               unsigned char *mem_10648)
{
    int32_t wave_sizze_10727;
    int32_t group_sizze_10728;
    bool thread_active_10729;
    int32_t gtid_10242;
    int32_t gtid_10243;
    int32_t global_tid_10251;
    int32_t local_tid_10252;
    int32_t group_id_10253;
    
    global_tid_10251 = get_global_id(0);
    local_tid_10252 = get_local_id(0);
    group_sizze_10728 = get_local_size(0);
    wave_sizze_10727 = LOCKSTEP_WIDTH;
    group_id_10253 = get_group_id(0);
    gtid_10242 = squot32(global_tid_10251, sizze_9902);
    gtid_10243 = global_tid_10251 - squot32(global_tid_10251, sizze_9902) *
        sizze_9902;
    thread_active_10729 = slt32(gtid_10242, sizze_9901) && slt32(gtid_10243,
                                                                 sizze_9902);
    
    float res_10254;
    
    if (thread_active_10729) {
        res_10254 = *(__global float *) &mem_10641[gtid_10242 * 4];
    }
    if (thread_active_10729) {
        *(__global float *) &mem_10648[(gtid_10242 * sizze_9902 + gtid_10243) *
                                       4] = res_10254;
    }
}
__kernel void map_kernel_10263(int32_t sizze_9901, __global
                               unsigned char *angles_mem_10634, __global
                               unsigned char *mem_10641, __global
                               unsigned char *mem_10644)
{
    int32_t wave_sizze_10724;
    int32_t group_sizze_10725;
    bool thread_active_10726;
    int32_t gtid_10256;
    int32_t global_tid_10263;
    int32_t local_tid_10264;
    int32_t group_id_10265;
    
    global_tid_10263 = get_global_id(0);
    local_tid_10264 = get_local_id(0);
    group_sizze_10725 = get_local_size(0);
    wave_sizze_10724 = LOCKSTEP_WIDTH;
    group_id_10265 = get_group_id(0);
    gtid_10256 = global_tid_10263;
    thread_active_10726 = slt32(gtid_10256, sizze_9901);
    
    float x_10266;
    float res_10267;
    float res_10268;
    
    if (thread_active_10726) {
        x_10266 = *(__global float *) &angles_mem_10634[gtid_10256 * 4];
        res_10267 = futrts_sin32(x_10266);
        res_10268 = futrts_cos32(x_10266);
    }
    if (thread_active_10726) {
        *(__global float *) &mem_10641[gtid_10256 * 4] = res_10267;
    }
    if (thread_active_10726) {
        *(__global float *) &mem_10644[gtid_10256 * 4] = res_10268;
    }
}
__kernel void map_kernel_10276(int32_t sizze_9902, float res_9912,
                               float res_9925, int32_t nesting_sizze_10244,
                               __global unsigned char *rays_mem_10636, __global
                               unsigned char *mem_10648, __global
                               unsigned char *mem_10652, __global
                               unsigned char *mem_10655, __global
                               unsigned char *mem_10658)
{
    int32_t wave_sizze_10733;
    int32_t group_sizze_10734;
    bool thread_active_10735;
    int32_t gtid_10269;
    int32_t global_tid_10276;
    int32_t local_tid_10277;
    int32_t group_id_10278;
    
    global_tid_10276 = get_global_id(0);
    local_tid_10277 = get_local_id(0);
    group_sizze_10734 = get_local_size(0);
    wave_sizze_10733 = LOCKSTEP_WIDTH;
    group_id_10278 = get_group_id(0);
    gtid_10269 = global_tid_10276;
    thread_active_10735 = slt32(gtid_10269, nesting_sizze_10244);
    
    int32_t new_index_10608;
    int32_t binop_y_10610;
    int32_t new_index_10611;
    float x_10279;
    float x_10280;
    float x_10281;
    bool cond_10282;
    float res_10283;
    bool cond_10287;
    float res_10288;
    float res_10292;
    float res_10296;
    bool cond_10297;
    bool res_10298;
    bool x_10299;
    float res_10300;
    float res_10301;
    
    if (thread_active_10735) {
        new_index_10608 = squot32(gtid_10269, sizze_9902);
        binop_y_10610 = sizze_9902 * new_index_10608;
        new_index_10611 = gtid_10269 - binop_y_10610;
        x_10279 = *(__global float *) &mem_10648[(new_index_10608 * sizze_9902 +
                                                  new_index_10611) * 4];
        x_10280 = *(__global float *) &mem_10652[(new_index_10608 * sizze_9902 +
                                                  new_index_10611) * 4];
        x_10281 = *(__global float *) &rays_mem_10636[new_index_10611 * 4];
        cond_10282 = x_10279 == 0.0F;
        if (cond_10282) {
            res_10283 = x_10281;
        } else {
            float y_10284;
            float x_10285;
            float res_10286;
            
            y_10284 = res_9925 * x_10280;
            x_10285 = x_10281 - y_10284;
            res_10286 = x_10285 / x_10279;
            res_10283 = res_10286;
        }
        cond_10287 = x_10280 == 0.0F;
        if (cond_10287) {
            res_10288 = x_10281;
        } else {
            float y_10289;
            float x_10290;
            float res_10291;
            
            y_10289 = res_9925 * x_10279;
            x_10290 = x_10281 - y_10289;
            res_10291 = x_10290 / x_10280;
            res_10288 = res_10291;
        }
        if (cond_10287) {
            res_10292 = x_10281;
        } else {
            float y_10293;
            float x_10294;
            float res_10295;
            
            y_10293 = res_9912 * x_10279;
            x_10294 = x_10281 - y_10293;
            res_10295 = x_10294 / x_10280;
            res_10292 = res_10295;
        }
        res_10296 = (float) fabs(res_10283);
        cond_10297 = res_10296 <= res_9912;
        res_10298 = !cond_10282;
        x_10299 = cond_10297 && res_10298;
        if (x_10299) {
            res_10300 = res_9925;
            res_10301 = res_10283;
        } else {
            bool cond_10302;
            float res_10303;
            float res_10304;
            
            cond_10302 = res_10288 <= res_10292;
            if (cond_10302) {
                res_10303 = res_10288;
            } else {
                res_10303 = res_10292;
            }
            if (cond_10302) {
                res_10304 = res_9925;
            } else {
                res_10304 = res_9912;
            }
            res_10300 = res_10303;
            res_10301 = res_10304;
        }
    }
    if (thread_active_10735) {
        *(__global float *) &mem_10655[gtid_10269 * 4] = res_10300;
    }
    if (thread_active_10735) {
        *(__global float *) &mem_10658[gtid_10269 * 4] = res_10301;
    }
}
__kernel void map_kernel_10331(float res_9911, float res_9912, float res_9925,
                               int32_t res_9959, int32_t res_9971, __global
                               unsigned char *voxels_mem_10638, __global
                               unsigned char *mem_10666, __global
                               unsigned char *mem_10669, __global
                               unsigned char *mem_10671, __global
                               unsigned char *mem_10673, __global
                               unsigned char *mem_10676, __global
                               unsigned char *mem_10679, __global
                               unsigned char *mem_10682, __global
                               unsigned char *mem_10703)
{
    int32_t wave_sizze_10749;
    int32_t group_sizze_10750;
    bool thread_active_10751;
    int32_t gtid_10324;
    int32_t global_tid_10331;
    int32_t local_tid_10332;
    int32_t group_id_10333;
    
    global_tid_10331 = get_global_id(0);
    local_tid_10332 = get_local_id(0);
    group_sizze_10750 = get_local_size(0);
    wave_sizze_10749 = LOCKSTEP_WIDTH;
    group_id_10333 = get_group_id(0);
    gtid_10324 = global_tid_10331;
    thread_active_10751 = slt32(gtid_10324, res_9971);
    
    float arg_10334;
    float arg_10335;
    bool res_10336;
    bool res_10337;
    float res_10338;
    bool cond_10341;
    float res_10342;
    int32_t res_10343;
    float res_10344;
    bool res_10345;
    float res_10346;
    float res_10353;
    float res_10354;
    bool cond_10377;
    bool res_10378;
    bool x_10379;
    bool cond_10380;
    bool res_10381;
    bool x_10382;
    bool cond_10383;
    bool res_10384;
    bool x_10385;
    bool x_10386;
    bool x_10387;
    bool y_10388;
    bool res_10389;
    bool x_10390;
    float y_10391;
    bool res_10392;
    float res_10395;
    float res_10396;
    float res_10397;
    float res_10398;
    int32_t res_10399;
    float res_10536;
    
    if (thread_active_10751) {
        arg_10334 = *(__global float *) &mem_10666[gtid_10324 * 4];
        arg_10335 = *(__global float *) &mem_10669[gtid_10324 * 4];
        res_10336 = *(__global bool *) &mem_10671[gtid_10324];
        res_10337 = *(__global bool *) &mem_10673[gtid_10324];
        res_10338 = *(__global float *) &mem_10676[gtid_10324 * 4];
        for (int32_t i_10752 = 0; i_10752 < res_9959; i_10752++) {
            *(__global float *) &mem_10679[(group_id_10333 * (res_9959 *
                                                              group_sizze_10326) +
                                            i_10752 * group_sizze_10326 +
                                            local_tid_10332) * 4] = -1.0F;
        }
        for (int32_t i_10753 = 0; i_10753 < res_9959; i_10753++) {
            *(__global int32_t *) &mem_10682[(group_id_10333 * (res_9959 *
                                                                group_sizze_10326) +
                                              i_10753 * group_sizze_10326 +
                                              local_tid_10332) * 4] = -1;
        }
        cond_10341 = res_10338 < 0.0F;
        if (cond_10341) {
            res_10342 = -1.0F;
        } else {
            res_10342 = 1.0F;
        }
        res_10343 = fptosi_f32_i32(arg_10334);
        res_10344 = sitofp_i32_f32(res_10343);
        res_10345 = 0.0F <= arg_10334;
        if (res_10345) {
            bool res_10347;
            float res_10348;
            
            res_10347 = res_10344 < arg_10334;
            if (res_10347) {
                res_10348 = res_10344;
            } else {
                res_10348 = arg_10334;
            }
            res_10346 = res_10348;
        } else {
            bool res_10349;
            float res_10350;
            
            res_10349 = arg_10334 < res_10344;
            if (res_10349) {
                int32_t res_10351;
                float res_10352;
                
                res_10351 = res_10343 - 1;
                res_10352 = sitofp_i32_f32(res_10351);
                res_10350 = res_10352;
            } else {
                res_10350 = arg_10334;
            }
            res_10346 = res_10350;
        }
        res_10353 = 1.0F + res_10346;
        if (cond_10341) {
            int32_t res_10355;
            float res_10356;
            bool res_10357;
            float res_10358;
            float res_10365;
            
            res_10355 = fptosi_f32_i32(arg_10335);
            res_10356 = sitofp_i32_f32(res_10355);
            res_10357 = 0.0F <= arg_10335;
            if (res_10357) {
                bool res_10359;
                float res_10360;
                
                res_10359 = res_10356 < arg_10335;
                if (res_10359) {
                    int32_t res_10361;
                    float res_10362;
                    
                    res_10361 = 1 + res_10355;
                    res_10362 = sitofp_i32_f32(res_10361);
                    res_10360 = res_10362;
                } else {
                    res_10360 = arg_10335;
                }
                res_10358 = res_10360;
            } else {
                bool res_10363;
                float res_10364;
                
                res_10363 = arg_10335 < res_10356;
                if (res_10363) {
                    res_10364 = res_10356;
                } else {
                    res_10364 = arg_10335;
                }
                res_10358 = res_10364;
            }
            res_10365 = res_10358 - 1.0F;
            res_10354 = res_10365;
        } else {
            int32_t res_10366;
            float res_10367;
            bool res_10368;
            float res_10369;
            float res_10376;
            
            res_10366 = fptosi_f32_i32(arg_10335);
            res_10367 = sitofp_i32_f32(res_10366);
            res_10368 = 0.0F <= arg_10335;
            if (res_10368) {
                bool res_10370;
                float res_10371;
                
                res_10370 = res_10367 < arg_10335;
                if (res_10370) {
                    res_10371 = res_10367;
                } else {
                    res_10371 = arg_10335;
                }
                res_10369 = res_10371;
            } else {
                bool res_10372;
                float res_10373;
                
                res_10372 = arg_10335 < res_10367;
                if (res_10372) {
                    int32_t res_10374;
                    float res_10375;
                    
                    res_10374 = res_10366 - 1;
                    res_10375 = sitofp_i32_f32(res_10374);
                    res_10373 = res_10375;
                } else {
                    res_10373 = arg_10335;
                }
                res_10369 = res_10373;
            }
            res_10376 = 1.0F + res_10369;
            res_10354 = res_10376;
        }
        cond_10377 = res_9925 <= arg_10334;
        res_10378 = arg_10334 < res_9912;
        x_10379 = cond_10377 && res_10378;
        cond_10380 = res_9925 < arg_10335;
        res_10381 = arg_10335 <= res_9912;
        x_10382 = cond_10380 && res_10381;
        cond_10383 = res_9925 <= arg_10335;
        res_10384 = arg_10335 < res_9912;
        x_10385 = cond_10383 && res_10384;
        x_10386 = cond_10341 && x_10382;
        x_10387 = !cond_10341;
        y_10388 = x_10385 && x_10387;
        res_10389 = x_10386 || y_10388;
        x_10390 = x_10379 && res_10389;
        y_10391 = 1.0F / res_10338;
        
        bool loop_while_10400;
        float focusPoint_10403;
        float focusPoint_10404;
        float anchorX_10405;
        float anchorY_10406;
        int32_t write_index_10407;
        
        loop_while_10400 = x_10390;
        focusPoint_10403 = arg_10334;
        focusPoint_10404 = arg_10335;
        anchorX_10405 = res_10353;
        anchorY_10406 = res_10354;
        write_index_10407 = 0;
        while (loop_while_10400) {
            float arg_10408 = res_9912 + focusPoint_10404;
            int32_t res_10409 = fptosi_f32_i32(arg_10408);
            float res_10410 = sitofp_i32_f32(res_10409);
            bool res_10411 = 0.0F <= arg_10408;
            float res_10412;
            
            if (res_10411) {
                bool res_10413;
                float res_10414;
                
                res_10413 = res_10410 < arg_10408;
                if (res_10413) {
                    res_10414 = res_10410;
                } else {
                    res_10414 = arg_10408;
                }
                res_10412 = res_10414;
            } else {
                bool res_10415;
                float res_10416;
                
                res_10415 = arg_10408 < res_10410;
                if (res_10415) {
                    int32_t res_10417;
                    float res_10418;
                    
                    res_10417 = res_10409 - 1;
                    res_10418 = sitofp_i32_f32(res_10417);
                    res_10416 = res_10418;
                } else {
                    res_10416 = arg_10408;
                }
                res_10412 = res_10416;
            }
            
            int32_t res_10419 = fptosi_f32_i32(focusPoint_10404);
            float res_10420 = sitofp_i32_f32(res_10419);
            bool res_10421 = 0.0F <= focusPoint_10404;
            float res_10422;
            
            if (res_10421) {
                bool res_10423;
                float res_10424;
                
                res_10423 = res_10420 < focusPoint_10404;
                if (res_10423) {
                    res_10424 = res_10420;
                } else {
                    res_10424 = focusPoint_10404;
                }
                res_10422 = res_10424;
            } else {
                bool res_10425;
                float res_10426;
                
                res_10425 = focusPoint_10404 < res_10420;
                if (res_10425) {
                    int32_t res_10427;
                    float res_10428;
                    
                    res_10427 = res_10419 - 1;
                    res_10428 = sitofp_i32_f32(res_10427);
                    res_10426 = res_10428;
                } else {
                    res_10426 = focusPoint_10404;
                }
                res_10422 = res_10426;
            }
            
            float x_10429 = focusPoint_10404 - res_10422;
            bool res_10430 = x_10429 == 0.0F;
            bool x_10431 = cond_10341 && res_10430;
            float res_10432;
            
            if (x_10431) {
                float res_10433 = res_10412 - 1.0F;
                
                res_10432 = res_10433;
            } else {
                res_10432 = res_10412;
            }
            
            float arg_10434 = res_9912 + focusPoint_10403;
            int32_t res_10435 = fptosi_f32_i32(arg_10434);
            float res_10436 = sitofp_i32_f32(res_10435);
            bool res_10437 = 0.0F <= arg_10434;
            float res_10438;
            
            if (res_10437) {
                bool res_10439;
                float res_10440;
                
                res_10439 = res_10436 < arg_10434;
                if (res_10439) {
                    res_10440 = res_10436;
                } else {
                    res_10440 = arg_10434;
                }
                res_10438 = res_10440;
            } else {
                bool res_10441;
                float res_10442;
                
                res_10441 = arg_10434 < res_10436;
                if (res_10441) {
                    int32_t res_10443;
                    float res_10444;
                    
                    res_10443 = res_10435 - 1;
                    res_10444 = sitofp_i32_f32(res_10443);
                    res_10442 = res_10444;
                } else {
                    res_10442 = arg_10434;
                }
                res_10438 = res_10442;
            }
            
            float y_10445 = res_9911 * res_10432;
            float arg_10446 = res_10438 + y_10445;
            int32_t res_10447 = fptosi_f32_i32(arg_10446);
            float res_10448;
            
            if (res_10337) {
                res_10448 = 1.0F;
            } else {
                float res_10449;
                
                if (res_10336) {
                    res_10449 = 0.0F;
                } else {
                    float x_10450;
                    float res_10451;
                    
                    x_10450 = anchorX_10405 - focusPoint_10403;
                    res_10451 = res_10338 * x_10450;
                    res_10449 = res_10451;
                }
                res_10448 = res_10449;
            }
            
            float res_10452;
            
            if (res_10337) {
                res_10452 = 0.0F;
            } else {
                float res_10453;
                
                if (res_10336) {
                    res_10453 = 1.0F;
                } else {
                    float x_10454;
                    float res_10455;
                    
                    x_10454 = anchorY_10406 - focusPoint_10404;
                    res_10455 = y_10391 * x_10454;
                    res_10453 = res_10455;
                }
                res_10452 = res_10453;
            }
            
            float res_10456 = focusPoint_10404 + res_10448;
            float res_10457 = focusPoint_10403 + res_10452;
            float x_10458 = anchorX_10405 - focusPoint_10403;
            float x_10459 = fpow32(x_10458, 2.0F);
            float x_10460 = res_10456 - focusPoint_10404;
            float y_10461 = fpow32(x_10460, 2.0F);
            float arg_10462 = x_10459 + y_10461;
            float res_10463;
            
            res_10463 = futrts_sqrt32(arg_10462);
            
            float x_10464 = res_10457 - focusPoint_10403;
            float x_10465 = fpow32(x_10464, 2.0F);
            float x_10466 = anchorY_10406 - focusPoint_10404;
            float y_10467 = fpow32(x_10466, 2.0F);
            float arg_10468 = x_10465 + y_10467;
            float res_10469;
            
            res_10469 = futrts_sqrt32(arg_10468);
            
            float res_10472;
            float res_10473;
            float res_10474;
            float res_10475;
            int32_t res_10476;
            
            if (res_10336) {
                float res_10479;
                int32_t res_10480;
                
                *(__global float *) &mem_10679[(group_id_10333 * (res_9959 *
                                                                  group_sizze_10326) +
                                                write_index_10407 *
                                                group_sizze_10326 +
                                                local_tid_10332) * 4] =
                    res_10463;
                *(__global int32_t *) &mem_10682[(group_id_10333 * (res_9959 *
                                                                    group_sizze_10326) +
                                                  write_index_10407 *
                                                  group_sizze_10326 +
                                                  local_tid_10332) * 4] =
                    res_10447;
                res_10479 = 1.0F + anchorX_10405;
                res_10480 = 1 + write_index_10407;
                res_10472 = anchorX_10405;
                res_10473 = res_10456;
                res_10474 = res_10479;
                res_10475 = anchorY_10406;
                res_10476 = res_10480;
            } else {
                float res_10483;
                float res_10484;
                float res_10485;
                float res_10486;
                int32_t res_10487;
                
                if (res_10337) {
                    float res_10490;
                    int32_t res_10491;
                    
                    *(__global float *) &mem_10679[(group_id_10333 * (res_9959 *
                                                                      group_sizze_10326) +
                                                    write_index_10407 *
                                                    group_sizze_10326 +
                                                    local_tid_10332) * 4] =
                        res_10469;
                    *(__global int32_t *) &mem_10682[(group_id_10333 *
                                                      (res_9959 *
                                                       group_sizze_10326) +
                                                      write_index_10407 *
                                                      group_sizze_10326 +
                                                      local_tid_10332) * 4] =
                        res_10447;
                    res_10490 = res_10342 + anchorY_10406;
                    res_10491 = 1 + write_index_10407;
                    res_10483 = res_10457;
                    res_10484 = anchorY_10406;
                    res_10485 = anchorX_10405;
                    res_10486 = res_10490;
                    res_10487 = res_10491;
                } else {
                    float arg_10492;
                    float res_10493;
                    bool cond_10494;
                    float res_10497;
                    float res_10498;
                    float res_10499;
                    float res_10500;
                    int32_t res_10501;
                    
                    arg_10492 = res_10463 - res_10469;
                    res_10493 = (float) fabs(arg_10492);
                    cond_10494 = 1.0e-9F < res_10493;
                    if (cond_10494) {
                        bool cond_10502;
                        float res_10503;
                        float res_10504;
                        float res_10507;
                        float res_10508;
                        int32_t res_10509;
                        
                        cond_10502 = res_10463 < res_10469;
                        if (cond_10502) {
                            res_10503 = anchorX_10405;
                        } else {
                            res_10503 = res_10457;
                        }
                        if (cond_10502) {
                            res_10504 = res_10456;
                        } else {
                            res_10504 = anchorY_10406;
                        }
                        if (cond_10502) {
                            float res_10512;
                            int32_t res_10513;
                            
                            *(__global float *) &mem_10679[(group_id_10333 *
                                                            (res_9959 *
                                                             group_sizze_10326) +
                                                            write_index_10407 *
                                                            group_sizze_10326 +
                                                            local_tid_10332) *
                                                           4] = res_10463;
                            *(__global int32_t *) &mem_10682[(group_id_10333 *
                                                              (res_9959 *
                                                               group_sizze_10326) +
                                                              write_index_10407 *
                                                              group_sizze_10326 +
                                                              local_tid_10332) *
                                                             4] = res_10447;
                            res_10512 = 1.0F + anchorX_10405;
                            res_10513 = 1 + write_index_10407;
                            res_10507 = res_10512;
                            res_10508 = anchorY_10406;
                            res_10509 = res_10513;
                        } else {
                            float res_10516;
                            int32_t res_10517;
                            
                            *(__global float *) &mem_10679[(group_id_10333 *
                                                            (res_9959 *
                                                             group_sizze_10326) +
                                                            write_index_10407 *
                                                            group_sizze_10326 +
                                                            local_tid_10332) *
                                                           4] = res_10469;
                            *(__global int32_t *) &mem_10682[(group_id_10333 *
                                                              (res_9959 *
                                                               group_sizze_10326) +
                                                              write_index_10407 *
                                                              group_sizze_10326 +
                                                              local_tid_10332) *
                                                             4] = res_10447;
                            res_10516 = res_10342 + anchorY_10406;
                            res_10517 = 1 + write_index_10407;
                            res_10507 = anchorX_10405;
                            res_10508 = res_10516;
                            res_10509 = res_10517;
                        }
                        res_10497 = res_10503;
                        res_10498 = res_10504;
                        res_10499 = res_10507;
                        res_10500 = res_10508;
                        res_10501 = res_10509;
                    } else {
                        float res_10520;
                        float res_10521;
                        int32_t res_10522;
                        
                        *(__global float *) &mem_10679[(group_id_10333 *
                                                        (res_9959 *
                                                         group_sizze_10326) +
                                                        write_index_10407 *
                                                        group_sizze_10326 +
                                                        local_tid_10332) * 4] =
                            res_10463;
                        *(__global int32_t *) &mem_10682[(group_id_10333 *
                                                          (res_9959 *
                                                           group_sizze_10326) +
                                                          write_index_10407 *
                                                          group_sizze_10326 +
                                                          local_tid_10332) *
                                                         4] = res_10447;
                        res_10520 = 1.0F + anchorX_10405;
                        res_10521 = res_10342 + anchorY_10406;
                        res_10522 = 1 + write_index_10407;
                        res_10497 = anchorX_10405;
                        res_10498 = res_10456;
                        res_10499 = res_10520;
                        res_10500 = res_10521;
                        res_10501 = res_10522;
                    }
                    res_10483 = res_10497;
                    res_10484 = res_10498;
                    res_10485 = res_10499;
                    res_10486 = res_10500;
                    res_10487 = res_10501;
                }
                res_10472 = res_10483;
                res_10473 = res_10484;
                res_10474 = res_10485;
                res_10475 = res_10486;
                res_10476 = res_10487;
            }
            
            bool cond_10523 = res_9925 <= res_10472;
            bool res_10524 = res_10472 < res_9912;
            bool x_10525 = cond_10523 && res_10524;
            bool cond_10526 = res_9925 < res_10473;
            bool res_10527 = res_10473 <= res_9912;
            bool x_10528 = cond_10526 && res_10527;
            bool cond_10529 = res_9925 <= res_10473;
            bool res_10530 = res_10473 < res_9912;
            bool x_10531 = cond_10529 && res_10530;
            bool x_10532 = cond_10341 && x_10528;
            bool y_10533 = x_10387 && x_10531;
            bool res_10534 = x_10532 || y_10533;
            bool x_10535 = x_10525 && res_10534;
            bool loop_while_tmp_10754 = x_10535;
            float focusPoint_tmp_10757 = res_10472;
            float focusPoint_tmp_10758 = res_10473;
            float anchorX_tmp_10759 = res_10474;
            float anchorY_tmp_10760 = res_10475;
            int32_t write_index_tmp_10761;
            
            write_index_tmp_10761 = res_10476;
            loop_while_10400 = loop_while_tmp_10754;
            focusPoint_10403 = focusPoint_tmp_10757;
            focusPoint_10404 = focusPoint_tmp_10758;
            anchorX_10405 = anchorX_tmp_10759;
            anchorY_10406 = anchorY_tmp_10760;
            write_index_10407 = write_index_tmp_10761;
        }
        res_10392 = loop_while_10400;
        res_10395 = focusPoint_10403;
        res_10396 = focusPoint_10404;
        res_10397 = anchorX_10405;
        res_10398 = anchorY_10406;
        res_10399 = write_index_10407;
        
        float x_10539 = 0.0F;
        
        for (int32_t chunk_offset_10538 = 0; chunk_offset_10538 < res_9959;
             chunk_offset_10538++) {
            float x_10548 = *(__global float *) &mem_10679[(group_id_10333 *
                                                            (res_9959 *
                                                             group_sizze_10326) +
                                                            chunk_offset_10538 *
                                                            group_sizze_10326 +
                                                            local_tid_10332) *
                                                           4];
            int32_t x_10549 = *(__global int32_t *) &mem_10682[(group_id_10333 *
                                                                (res_9959 *
                                                                 group_sizze_10326) +
                                                                chunk_offset_10538 *
                                                                group_sizze_10326 +
                                                                local_tid_10332) *
                                                               4];
            bool cond_10551 = x_10549 == -1;
            float res_10552;
            
            if (cond_10551) {
                res_10552 = 0.0F;
            } else {
                float y_10553;
                float res_10554;
                
                y_10553 = *(__global float *) &voxels_mem_10638[x_10549 * 4];
                res_10554 = x_10548 * y_10553;
                res_10552 = res_10554;
            }
            
            float res_10556 = x_10539 + res_10552;
            float x_tmp_10762 = res_10556;
            
            x_10539 = x_tmp_10762;
        }
        res_10536 = x_10539;
    }
    if (thread_active_10751) {
        *(__global float *) &mem_10703[gtid_10324 * 4] = res_10536;
    }
}
__kernel void map_kernel_10591(int32_t sizze_9902, int32_t res_9971,
                               int32_t x_9984, __global
                               unsigned char *mem_10648, __global
                               unsigned char *mem_10652, __global
                               unsigned char *mem_10655, __global
                               unsigned char *mem_10658, __global
                               unsigned char *mem_10666, __global
                               unsigned char *mem_10669, __global
                               unsigned char *mem_10671, __global
                               unsigned char *mem_10673, __global
                               unsigned char *mem_10676)
{
    int32_t wave_sizze_10746;
    int32_t group_sizze_10747;
    bool thread_active_10748;
    int32_t gtid_10584;
    int32_t global_tid_10591;
    int32_t local_tid_10592;
    int32_t group_id_10593;
    
    global_tid_10591 = get_global_id(0);
    local_tid_10592 = get_local_id(0);
    group_sizze_10747 = get_local_size(0);
    wave_sizze_10746 = LOCKSTEP_WIDTH;
    group_id_10593 = get_group_id(0);
    gtid_10584 = global_tid_10591;
    thread_active_10748 = slt32(gtid_10584, res_9971);
    
    int32_t i_10595;
    int32_t new_index_10596;
    int32_t binop_y_10597;
    int32_t new_index_10598;
    float arg_10599;
    float arg_10600;
    float arg_10601;
    float arg_10602;
    bool res_10603;
    float res_10604;
    bool res_10605;
    float y_10606;
    float res_10607;
    
    if (thread_active_10748) {
        i_10595 = x_9984 + gtid_10584;
        new_index_10596 = squot32(i_10595, sizze_9902);
        binop_y_10597 = sizze_9902 * new_index_10596;
        new_index_10598 = i_10595 - binop_y_10597;
        arg_10599 = *(__global float *) &mem_10648[(new_index_10596 *
                                                    sizze_9902 +
                                                    new_index_10598) * 4];
        arg_10600 = *(__global float *) &mem_10652[(new_index_10596 *
                                                    sizze_9902 +
                                                    new_index_10598) * 4];
        arg_10601 = *(__global float *) &mem_10655[i_10595 * 4];
        arg_10602 = *(__global float *) &mem_10658[i_10595 * 4];
        res_10603 = arg_10600 == 0.0F;
        res_10604 = (float) fabs(arg_10600);
        res_10605 = res_10604 == 1.0F;
        y_10606 = 0.0F - arg_10599;
        res_10607 = arg_10600 / y_10606;
    }
    if (thread_active_10748) {
        *(__global float *) &mem_10666[gtid_10584 * 4] = arg_10601;
    }
    if (thread_active_10748) {
        *(__global float *) &mem_10669[gtid_10584 * 4] = arg_10602;
    }
    if (thread_active_10748) {
        *(__global bool *) &mem_10671[gtid_10584] = res_10603;
    }
    if (thread_active_10748) {
        *(__global bool *) &mem_10673[gtid_10584] = res_10605;
    }
    if (thread_active_10748) {
        *(__global float *) &mem_10676[gtid_10584 * 4] = res_10607;
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
class forwardprojection_jh:
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
                                       all_sizes={"group_size_10229": {"class": "group_size", "value": None},
                                        "group_size_10245": {"class": "group_size", "value": None},
                                        "group_size_10257": {"class": "group_size", "value": None},
                                        "group_size_10270": {"class": "group_size", "value": None},
                                        "group_size_10325": {"class": "group_size", "value": None},
                                        "group_size_10585": {"class": "group_size", "value": None},
                                        "group_size_10738": {"class": "group_size", "value": None}})
    self.kernel_replicate_9955_var = program.kernel_replicate_9955
    self.map_kernel_10235_var = program.map_kernel_10235
    self.map_kernel_10251_var = program.map_kernel_10251
    self.map_kernel_10263_var = program.map_kernel_10263
    self.map_kernel_10276_var = program.map_kernel_10276
    self.map_kernel_10331_var = program.map_kernel_10331
    self.map_kernel_10591_var = program.map_kernel_10591
  def futhark_main(self, angles_mem_sizze_10633, angles_mem_10634,
                   rays_mem_sizze_10635, rays_mem_10636, voxels_mem_sizze_10637,
                   voxels_mem_10638, sizze_9901, sizze_9902, sizze_9903,
                   stepsizze_9907):
    res_9908 = sitofp_i32_f32(sizze_9903)
    res_9909 = futhark_sqrt32(res_9908)
    res_9910 = fptosi_f32_i32(res_9909)
    res_9911 = sitofp_i32_f32(res_9910)
    res_9912 = (res_9911 / np.float32(2.0))
    group_sizze_10258 = self.sizes["group_size_10257"]
    y_10259 = (group_sizze_10258 - np.int32(1))
    x_10260 = (sizze_9901 + y_10259)
    num_groups_10261 = squot32(x_10260, group_sizze_10258)
    num_threads_10262 = (group_sizze_10258 * num_groups_10261)
    binop_x_10640 = sext_i32_i64(sizze_9901)
    bytes_10639 = (np.int64(4) * binop_x_10640)
    mem_10641 = opencl_alloc(self, bytes_10639, "mem_10641")
    mem_10644 = opencl_alloc(self, bytes_10639, "mem_10644")
    if ((1 * (num_groups_10261 * group_sizze_10258)) != 0):
      self.map_kernel_10263_var.set_args(np.int32(sizze_9901), angles_mem_10634,
                                         mem_10641, mem_10644)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10263_var,
                                 (np.long((num_groups_10261 * group_sizze_10258)),),
                                 (np.long(group_sizze_10258),))
      if synchronous:
        self.queue.finish()
    nesting_sizze_10244 = (sizze_9901 * sizze_9902)
    group_sizze_10246 = self.sizes["group_size_10245"]
    y_10247 = (group_sizze_10246 - np.int32(1))
    x_10248 = (nesting_sizze_10244 + y_10247)
    num_groups_10249 = squot32(x_10248, group_sizze_10246)
    num_threads_10250 = (group_sizze_10246 * num_groups_10249)
    binop_x_10647 = sext_i32_i64(nesting_sizze_10244)
    bytes_10645 = (np.int64(4) * binop_x_10647)
    mem_10648 = opencl_alloc(self, bytes_10645, "mem_10648")
    if ((1 * (num_groups_10249 * group_sizze_10246)) != 0):
      self.map_kernel_10251_var.set_args(np.int32(sizze_9901),
                                         np.int32(sizze_9902), mem_10641,
                                         mem_10648)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10251_var,
                                 (np.long((num_groups_10249 * group_sizze_10246)),),
                                 (np.long(group_sizze_10246),))
      if synchronous:
        self.queue.finish()
    mem_10641 = None
    group_sizze_10230 = self.sizes["group_size_10229"]
    y_10231 = (group_sizze_10230 - np.int32(1))
    x_10232 = (y_10231 + nesting_sizze_10244)
    num_groups_10233 = squot32(x_10232, group_sizze_10230)
    num_threads_10234 = (group_sizze_10230 * num_groups_10233)
    mem_10652 = opencl_alloc(self, bytes_10645, "mem_10652")
    if ((1 * (num_groups_10233 * group_sizze_10230)) != 0):
      self.map_kernel_10235_var.set_args(np.int32(sizze_9901),
                                         np.int32(sizze_9902), mem_10644,
                                         mem_10652)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10235_var,
                                 (np.long((num_groups_10233 * group_sizze_10230)),),
                                 (np.long(group_sizze_10230),))
      if synchronous:
        self.queue.finish()
    mem_10644 = None
    res_9925 = (np.float32(0.0) - res_9912)
    group_sizze_10271 = self.sizes["group_size_10270"]
    y_10272 = (group_sizze_10271 - np.int32(1))
    x_10273 = (nesting_sizze_10244 + y_10272)
    num_groups_10274 = squot32(x_10273, group_sizze_10271)
    num_threads_10275 = (group_sizze_10271 * num_groups_10274)
    mem_10655 = opencl_alloc(self, bytes_10645, "mem_10655")
    mem_10658 = opencl_alloc(self, bytes_10645, "mem_10658")
    if ((1 * (num_groups_10274 * group_sizze_10271)) != 0):
      self.map_kernel_10276_var.set_args(np.int32(sizze_9902),
                                         np.float32(res_9912),
                                         np.float32(res_9925),
                                         np.int32(nesting_sizze_10244),
                                         rays_mem_10636, mem_10648, mem_10652,
                                         mem_10655, mem_10658)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10276_var,
                                 (np.long((num_groups_10274 * group_sizze_10271)),),
                                 (np.long(group_sizze_10271),))
      if synchronous:
        self.queue.finish()
    res_9954 = sdiv32(nesting_sizze_10244, stepsizze_9907)
    mem_10661 = opencl_alloc(self, np.int64(4), "mem_10661")
    group_sizze_10738 = self.sizes["group_size_10738"]
    num_groups_10739 = squot32(((np.int32(1) + sext_i32_i32(group_sizze_10738)) - np.int32(1)),
                               sext_i32_i32(group_sizze_10738))
    if ((1 * (num_groups_10739 * group_sizze_10738)) != 0):
      self.kernel_replicate_9955_var.set_args(mem_10661)
      cl.enqueue_nd_range_kernel(self.queue, self.kernel_replicate_9955_var,
                                 (np.long((num_groups_10739 * group_sizze_10738)),),
                                 (np.long(group_sizze_10738),))
      if synchronous:
        self.queue.finish()
    loop_cond_9956 = slt32(np.int32(0), res_9954)
    x_9957 = (np.float32(2.0) * res_9911)
    arg_9958 = (x_9957 - np.float32(1.0))
    res_9959 = fptosi_f32_i32(arg_9958)
    group_sizze_10586 = self.sizes["group_size_10585"]
    y_10587 = (group_sizze_10586 - np.int32(1))
    group_sizze_10326 = self.sizes["group_size_10325"]
    y_10327 = (group_sizze_10326 - np.int32(1))
    binop_x_10678 = sext_i32_i64(res_9959)
    bytes_10677 = (np.int64(4) * binop_x_10678)
    sizze_9964 = np.int32(1)
    output_mem_sizze_10662 = np.int64(4)
    output_mem_10663 = mem_10661
    loop_while_9965 = loop_cond_9956
    run_9967 = np.int32(0)
    while loop_while_9965:
      x_9968 = (np.int32(1) + run_9967)
      x_9969 = (stepsizze_9907 * x_9968)
      cond_9970 = sle32(nesting_sizze_10244, x_9969)
      if cond_9970:
        y_9972 = (stepsizze_9907 * run_9967)
        res_9973 = (nesting_sizze_10244 - y_9972)
        res_9971 = res_9973
      else:
        res_9971 = stepsizze_9907
      bounds_invalid_upwards_9974 = slt32(res_9971, np.int32(0))
      eq_x_zz_9977 = (np.int32(0) == res_9971)
      not_p_9978 = not(bounds_invalid_upwards_9974)
      p_and_eq_x_y_9979 = (eq_x_zz_9977 and not_p_9978)
      dim_zzero_9980 = (bounds_invalid_upwards_9974 or p_and_eq_x_y_9979)
      both_empty_9981 = (eq_x_zz_9977 and dim_zzero_9980)
      empty_or_match_9982 = (not_p_9978 or both_empty_9981)
      empty_or_match_cert_9983 = True
      assert empty_or_match_9982, ("Error at forwardprojection_jh.fut:26:1-30:58 -> forwardprojection_jh.fut:30:11-30:58 -> projection_lib.fut:264:185-264:193 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                                                 "*",
                                                                                                                                                                                                                 "[",
                                                                                                                                                                                                                 res_9971,
                                                                                                                                                                                                                 "]",
                                                                                                                                                                                                                 "intrinsics.i32"))
      x_9984 = (stepsizze_9907 * run_9967)
      x_10588 = (res_9971 + y_10587)
      num_groups_10589 = squot32(x_10588, group_sizze_10586)
      num_threads_10590 = (group_sizze_10586 * num_groups_10589)
      binop_x_10665 = sext_i32_i64(res_9971)
      bytes_10664 = (np.int64(4) * binop_x_10665)
      mem_10666 = opencl_alloc(self, bytes_10664, "mem_10666")
      mem_10669 = opencl_alloc(self, bytes_10664, "mem_10669")
      mem_10671 = opencl_alloc(self, binop_x_10665, "mem_10671")
      mem_10673 = opencl_alloc(self, binop_x_10665, "mem_10673")
      mem_10676 = opencl_alloc(self, bytes_10664, "mem_10676")
      if ((1 * (num_groups_10589 * group_sizze_10586)) != 0):
        self.map_kernel_10591_var.set_args(np.int32(sizze_9902),
                                           np.int32(res_9971), np.int32(x_9984),
                                           mem_10648, mem_10652, mem_10655,
                                           mem_10658, mem_10666, mem_10669,
                                           mem_10671, mem_10673, mem_10676)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10591_var,
                                   (np.long((num_groups_10589 * group_sizze_10586)),),
                                   (np.long(group_sizze_10586),))
        if synchronous:
          self.queue.finish()
      x_10328 = (res_9971 + y_10327)
      num_groups_10329 = squot32(x_10328, group_sizze_10326)
      num_threads_10330 = (group_sizze_10326 * num_groups_10329)
      mem_10703 = opencl_alloc(self, bytes_10664, "mem_10703")
      num_threads64_10718 = sext_i32_i64(num_threads_10330)
      total_sizze_10719 = (bytes_10677 * num_threads64_10718)
      mem_10679 = opencl_alloc(self, total_sizze_10719, "mem_10679")
      total_sizze_10720 = (bytes_10677 * num_threads64_10718)
      mem_10682 = opencl_alloc(self, total_sizze_10720, "mem_10682")
      if ((1 * (num_groups_10329 * group_sizze_10326)) != 0):
        self.map_kernel_10331_var.set_args(np.float32(res_9911),
                                           np.float32(res_9912),
                                           np.float32(res_9925),
                                           np.int32(res_9959),
                                           np.int32(res_9971), voxels_mem_10638,
                                           mem_10666, mem_10669, mem_10671,
                                           mem_10673, mem_10676, mem_10679,
                                           mem_10682, mem_10703)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10331_var,
                                   (np.long((num_groups_10329 * group_sizze_10326)),),
                                   (np.long(group_sizze_10326),))
        if synchronous:
          self.queue.finish()
      mem_10666 = None
      mem_10669 = None
      mem_10671 = None
      mem_10673 = None
      mem_10676 = None
      mem_10679 = None
      mem_10682 = None
      conc_tmp_10208 = (sizze_9964 + res_9971)
      binop_x_10705 = sext_i32_i64(conc_tmp_10208)
      bytes_10704 = (np.int64(4) * binop_x_10705)
      mem_10706 = opencl_alloc(self, bytes_10704, "mem_10706")
      tmp_offs_10763 = np.int32(0)
      if ((sizze_9964 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10706, output_mem_10663,
                        dest_offset=np.long((tmp_offs_10763 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sizze_9964 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10763 = (tmp_offs_10763 + sizze_9964)
      if ((res_9971 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10706, mem_10703,
                        dest_offset=np.long((tmp_offs_10763 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((res_9971 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10763 = (tmp_offs_10763 + res_9971)
      mem_10703 = None
      loop_cond_10210 = slt32(x_9968, res_9954)
      sizze_tmp_10740 = conc_tmp_10208
      output_mem_sizze_tmp_10741 = bytes_10704
      output_mem_tmp_10742 = mem_10706
      loop_while_tmp_10743 = loop_cond_10210
      run_tmp_10745 = x_9968
      sizze_9964 = sizze_tmp_10740
      output_mem_sizze_10662 = output_mem_sizze_tmp_10741
      output_mem_10663 = output_mem_tmp_10742
      loop_while_9965 = loop_while_tmp_10743
      run_9967 = run_tmp_10745
    sizze_9960 = sizze_9964
    res_mem_sizze_10707 = output_mem_sizze_10662
    res_mem_10708 = output_mem_10663
    res_9961 = loop_while_9965
    res_9963 = run_9967
    mem_10648 = None
    mem_10652 = None
    mem_10655 = None
    mem_10658 = None
    mem_10661 = None
    j_m_i_10211 = (sizze_9960 - np.int32(1))
    x_10212 = abs(j_m_i_10211)
    empty_slice_10213 = (x_10212 == np.int32(0))
    m_10214 = (x_10212 - np.int32(1))
    i_p_m_t_s_10215 = (np.int32(1) + m_10214)
    zzero_leq_i_p_m_t_s_10216 = sle32(np.int32(0), i_p_m_t_s_10215)
    i_p_m_t_s_leq_w_10217 = slt32(i_p_m_t_s_10215, sizze_9960)
    i_lte_j_10218 = sle32(np.int32(1), sizze_9960)
    y_10219 = (zzero_leq_i_p_m_t_s_10216 and i_p_m_t_s_leq_w_10217)
    y_10220 = (i_lte_j_10218 and y_10219)
    ok_or_empty_10221 = (empty_slice_10213 or y_10220)
    index_certs_10222 = True
    assert ok_or_empty_10221, ("Error at forwardprojection_jh.fut:26:1-30:58 -> forwardprojection_jh.fut:30:11-30:58 -> projection_lib.fut:267:20-267:31 -> /futlib/array.fut:21:29-21:33: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                          np.int32(1),
                                                                                                                                                                                                          "] out of bounds for array of shape [",
                                                                                                                                                                                                          sizze_9960,
                                                                                                                                                                                                          "]."))
    binop_x_10710 = sext_i32_i64(x_10212)
    bytes_10709 = (np.int64(4) * binop_x_10710)
    mem_10711 = opencl_alloc(self, bytes_10709, "mem_10711")
    if ((x_10212 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_10711, res_mem_10708,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(4)),
                      byte_count=np.long((x_10212 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    res_mem_10708 = None
    out_arrsizze_10723 = x_10212
    out_memsizze_10722 = bytes_10709
    out_mem_10721 = mem_10711
    return (out_memsizze_10722, out_mem_10721, out_arrsizze_10723)
  def main(self, angles_mem_10634_ext, rays_mem_10636_ext, voxels_mem_10638_ext,
           stepsizze_9907_ext):
    try:
      assert ((type(angles_mem_10634_ext) in [np.ndarray,
                                              cl.array.Array]) and (angles_mem_10634_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9901 = np.int32(angles_mem_10634_ext.shape[0])
      angles_mem_sizze_10633 = np.int64(angles_mem_10634_ext.nbytes)
      if (type(angles_mem_10634_ext) == cl.array.Array):
        angles_mem_10634 = angles_mem_10634_ext.data
      else:
        angles_mem_10634 = opencl_alloc(self, angles_mem_sizze_10633,
                                        "angles_mem_10634")
        if (angles_mem_sizze_10633 != 0):
          cl.enqueue_copy(self.queue, angles_mem_10634,
                          normaliseArray(angles_mem_10634_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(angles_mem_10634_ext),
                                                                                                                            angles_mem_10634_ext))
    try:
      assert ((type(rays_mem_10636_ext) in [np.ndarray,
                                            cl.array.Array]) and (rays_mem_10636_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9902 = np.int32(rays_mem_10636_ext.shape[0])
      rays_mem_sizze_10635 = np.int64(rays_mem_10636_ext.nbytes)
      if (type(rays_mem_10636_ext) == cl.array.Array):
        rays_mem_10636 = rays_mem_10636_ext.data
      else:
        rays_mem_10636 = opencl_alloc(self, rays_mem_sizze_10635,
                                      "rays_mem_10636")
        if (rays_mem_sizze_10635 != 0):
          cl.enqueue_copy(self.queue, rays_mem_10636,
                          normaliseArray(rays_mem_10636_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(rays_mem_10636_ext),
                                                                                                                            rays_mem_10636_ext))
    try:
      assert ((type(voxels_mem_10638_ext) in [np.ndarray,
                                              cl.array.Array]) and (voxels_mem_10638_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9903 = np.int32(voxels_mem_10638_ext.shape[0])
      voxels_mem_sizze_10637 = np.int64(voxels_mem_10638_ext.nbytes)
      if (type(voxels_mem_10638_ext) == cl.array.Array):
        voxels_mem_10638 = voxels_mem_10638_ext.data
      else:
        voxels_mem_10638 = opencl_alloc(self, voxels_mem_sizze_10637,
                                        "voxels_mem_10638")
        if (voxels_mem_sizze_10637 != 0):
          cl.enqueue_copy(self.queue, voxels_mem_10638,
                          normaliseArray(voxels_mem_10638_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(voxels_mem_10638_ext),
                                                                                                                            voxels_mem_10638_ext))
    try:
      stepsizze_9907 = np.int32(ct.c_int32(stepsizze_9907_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(stepsizze_9907_ext),
                                                                                                                            stepsizze_9907_ext))
    (out_memsizze_10722, out_mem_10721,
     out_arrsizze_10723) = self.futhark_main(angles_mem_sizze_10633,
                                             angles_mem_10634,
                                             rays_mem_sizze_10635,
                                             rays_mem_10636,
                                             voxels_mem_sizze_10637,
                                             voxels_mem_10638, sizze_9901,
                                             sizze_9902, sizze_9903,
                                             stepsizze_9907)
    return cl.array.Array(self.queue, (out_arrsizze_10723,), ct.c_float,
                          data=out_mem_10721)