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
#define group_sizze_9748 (group_size_9747)
__kernel void kernel_replicate_9377(__global unsigned char *mem_10083)
{
    const uint replicate_gtid_9377 = get_global_id(0);
    
    if (replicate_gtid_9377 >= 1)
        return;
    *(__global float *) &mem_10083[replicate_gtid_9377 * 4] = 0.0F;
}
__kernel void map_kernel_10013(int32_t sizze_9324, int32_t res_9393,
                               int32_t x_9406, __global
                               unsigned char *mem_10070, __global
                               unsigned char *mem_10074, __global
                               unsigned char *mem_10077, __global
                               unsigned char *mem_10080, __global
                               unsigned char *mem_10088, __global
                               unsigned char *mem_10091, __global
                               unsigned char *mem_10093, __global
                               unsigned char *mem_10095, __global
                               unsigned char *mem_10098)
{
    int32_t wave_sizze_10168;
    int32_t group_sizze_10169;
    bool thread_active_10170;
    int32_t gtid_10006;
    int32_t global_tid_10013;
    int32_t local_tid_10014;
    int32_t group_id_10015;
    
    global_tid_10013 = get_global_id(0);
    local_tid_10014 = get_local_id(0);
    group_sizze_10169 = get_local_size(0);
    wave_sizze_10168 = LOCKSTEP_WIDTH;
    group_id_10015 = get_group_id(0);
    gtid_10006 = global_tid_10013;
    thread_active_10170 = slt32(gtid_10006, res_9393);
    
    int32_t i_10017;
    int32_t new_index_10018;
    int32_t binop_y_10019;
    int32_t new_index_10020;
    float arg_10021;
    float arg_10022;
    float arg_10023;
    float arg_10024;
    bool res_10025;
    float res_10026;
    bool res_10027;
    float y_10028;
    float res_10029;
    
    if (thread_active_10170) {
        i_10017 = x_9406 + gtid_10006;
        new_index_10018 = squot32(i_10017, sizze_9324);
        binop_y_10019 = sizze_9324 * new_index_10018;
        new_index_10020 = i_10017 - binop_y_10019;
        arg_10021 = *(__global float *) &mem_10070[(new_index_10018 *
                                                    sizze_9324 +
                                                    new_index_10020) * 4];
        arg_10022 = *(__global float *) &mem_10074[(new_index_10018 *
                                                    sizze_9324 +
                                                    new_index_10020) * 4];
        arg_10023 = *(__global float *) &mem_10077[i_10017 * 4];
        arg_10024 = *(__global float *) &mem_10080[i_10017 * 4];
        res_10025 = arg_10022 == 0.0F;
        res_10026 = (float) fabs(arg_10022);
        res_10027 = res_10026 == 1.0F;
        y_10028 = 0.0F - arg_10021;
        res_10029 = arg_10022 / y_10028;
    }
    if (thread_active_10170) {
        *(__global float *) &mem_10088[gtid_10006 * 4] = arg_10023;
    }
    if (thread_active_10170) {
        *(__global float *) &mem_10091[gtid_10006 * 4] = arg_10024;
    }
    if (thread_active_10170) {
        *(__global bool *) &mem_10093[gtid_10006] = res_10025;
    }
    if (thread_active_10170) {
        *(__global bool *) &mem_10095[gtid_10006] = res_10027;
    }
    if (thread_active_10170) {
        *(__global float *) &mem_10098[gtid_10006 * 4] = res_10029;
    }
}
__kernel void map_kernel_9657(int32_t sizze_9323, int32_t sizze_9324, __global
                              unsigned char *mem_10066, __global
                              unsigned char *mem_10074)
{
    int32_t wave_sizze_10152;
    int32_t group_sizze_10153;
    bool thread_active_10154;
    int32_t gtid_9648;
    int32_t gtid_9649;
    int32_t global_tid_9657;
    int32_t local_tid_9658;
    int32_t group_id_9659;
    
    global_tid_9657 = get_global_id(0);
    local_tid_9658 = get_local_id(0);
    group_sizze_10153 = get_local_size(0);
    wave_sizze_10152 = LOCKSTEP_WIDTH;
    group_id_9659 = get_group_id(0);
    gtid_9648 = squot32(global_tid_9657, sizze_9324);
    gtid_9649 = global_tid_9657 - squot32(global_tid_9657, sizze_9324) *
        sizze_9324;
    thread_active_10154 = slt32(gtid_9648, sizze_9323) && slt32(gtid_9649,
                                                                sizze_9324);
    
    float res_9660;
    
    if (thread_active_10154) {
        res_9660 = *(__global float *) &mem_10066[gtid_9648 * 4];
    }
    if (thread_active_10154) {
        *(__global float *) &mem_10074[(gtid_9648 * sizze_9324 + gtid_9649) *
                                       4] = res_9660;
    }
}
__kernel void map_kernel_9673(int32_t sizze_9323, int32_t sizze_9324, __global
                              unsigned char *mem_10063, __global
                              unsigned char *mem_10070)
{
    int32_t wave_sizze_10149;
    int32_t group_sizze_10150;
    bool thread_active_10151;
    int32_t gtid_9664;
    int32_t gtid_9665;
    int32_t global_tid_9673;
    int32_t local_tid_9674;
    int32_t group_id_9675;
    
    global_tid_9673 = get_global_id(0);
    local_tid_9674 = get_local_id(0);
    group_sizze_10150 = get_local_size(0);
    wave_sizze_10149 = LOCKSTEP_WIDTH;
    group_id_9675 = get_group_id(0);
    gtid_9664 = squot32(global_tid_9673, sizze_9324);
    gtid_9665 = global_tid_9673 - squot32(global_tid_9673, sizze_9324) *
        sizze_9324;
    thread_active_10151 = slt32(gtid_9664, sizze_9323) && slt32(gtid_9665,
                                                                sizze_9324);
    
    float res_9676;
    
    if (thread_active_10151) {
        res_9676 = *(__global float *) &mem_10063[gtid_9664 * 4];
    }
    if (thread_active_10151) {
        *(__global float *) &mem_10070[(gtid_9664 * sizze_9324 + gtid_9665) *
                                       4] = res_9676;
    }
}
__kernel void map_kernel_9685(int32_t sizze_9323, __global
                              unsigned char *angles_mem_10056, __global
                              unsigned char *mem_10063, __global
                              unsigned char *mem_10066)
{
    int32_t wave_sizze_10146;
    int32_t group_sizze_10147;
    bool thread_active_10148;
    int32_t gtid_9678;
    int32_t global_tid_9685;
    int32_t local_tid_9686;
    int32_t group_id_9687;
    
    global_tid_9685 = get_global_id(0);
    local_tid_9686 = get_local_id(0);
    group_sizze_10147 = get_local_size(0);
    wave_sizze_10146 = LOCKSTEP_WIDTH;
    group_id_9687 = get_group_id(0);
    gtid_9678 = global_tid_9685;
    thread_active_10148 = slt32(gtid_9678, sizze_9323);
    
    float x_9688;
    float res_9689;
    float res_9690;
    
    if (thread_active_10148) {
        x_9688 = *(__global float *) &angles_mem_10056[gtid_9678 * 4];
        res_9689 = futrts_sin32(x_9688);
        res_9690 = futrts_cos32(x_9688);
    }
    if (thread_active_10148) {
        *(__global float *) &mem_10063[gtid_9678 * 4] = res_9689;
    }
    if (thread_active_10148) {
        *(__global float *) &mem_10066[gtid_9678 * 4] = res_9690;
    }
}
__kernel void map_kernel_9698(int32_t sizze_9324, float res_9334,
                              float res_9347, int32_t nesting_sizze_9666,
                              __global unsigned char *rays_mem_10058, __global
                              unsigned char *mem_10070, __global
                              unsigned char *mem_10074, __global
                              unsigned char *mem_10077, __global
                              unsigned char *mem_10080)
{
    int32_t wave_sizze_10155;
    int32_t group_sizze_10156;
    bool thread_active_10157;
    int32_t gtid_9691;
    int32_t global_tid_9698;
    int32_t local_tid_9699;
    int32_t group_id_9700;
    
    global_tid_9698 = get_global_id(0);
    local_tid_9699 = get_local_id(0);
    group_sizze_10156 = get_local_size(0);
    wave_sizze_10155 = LOCKSTEP_WIDTH;
    group_id_9700 = get_group_id(0);
    gtid_9691 = global_tid_9698;
    thread_active_10157 = slt32(gtid_9691, nesting_sizze_9666);
    
    int32_t new_index_10030;
    int32_t binop_y_10032;
    int32_t new_index_10033;
    float x_9701;
    float x_9702;
    float x_9703;
    bool cond_9704;
    float res_9705;
    bool cond_9709;
    float res_9710;
    float res_9714;
    float res_9718;
    bool cond_9719;
    bool res_9720;
    bool x_9721;
    float res_9722;
    float res_9723;
    
    if (thread_active_10157) {
        new_index_10030 = squot32(gtid_9691, sizze_9324);
        binop_y_10032 = sizze_9324 * new_index_10030;
        new_index_10033 = gtid_9691 - binop_y_10032;
        x_9701 = *(__global float *) &mem_10070[(new_index_10030 * sizze_9324 +
                                                 new_index_10033) * 4];
        x_9702 = *(__global float *) &mem_10074[(new_index_10030 * sizze_9324 +
                                                 new_index_10033) * 4];
        x_9703 = *(__global float *) &rays_mem_10058[new_index_10033 * 4];
        cond_9704 = x_9701 == 0.0F;
        if (cond_9704) {
            res_9705 = x_9703;
        } else {
            float y_9706;
            float x_9707;
            float res_9708;
            
            y_9706 = res_9347 * x_9702;
            x_9707 = x_9703 - y_9706;
            res_9708 = x_9707 / x_9701;
            res_9705 = res_9708;
        }
        cond_9709 = x_9702 == 0.0F;
        if (cond_9709) {
            res_9710 = x_9703;
        } else {
            float y_9711;
            float x_9712;
            float res_9713;
            
            y_9711 = res_9347 * x_9701;
            x_9712 = x_9703 - y_9711;
            res_9713 = x_9712 / x_9702;
            res_9710 = res_9713;
        }
        if (cond_9709) {
            res_9714 = x_9703;
        } else {
            float y_9715;
            float x_9716;
            float res_9717;
            
            y_9715 = res_9334 * x_9701;
            x_9716 = x_9703 - y_9715;
            res_9717 = x_9716 / x_9702;
            res_9714 = res_9717;
        }
        res_9718 = (float) fabs(res_9705);
        cond_9719 = res_9718 <= res_9334;
        res_9720 = !cond_9704;
        x_9721 = cond_9719 && res_9720;
        if (x_9721) {
            res_9722 = res_9347;
            res_9723 = res_9705;
        } else {
            bool cond_9724;
            float res_9725;
            float res_9726;
            
            cond_9724 = res_9710 <= res_9714;
            if (cond_9724) {
                res_9725 = res_9710;
            } else {
                res_9725 = res_9714;
            }
            if (cond_9724) {
                res_9726 = res_9347;
            } else {
                res_9726 = res_9334;
            }
            res_9722 = res_9725;
            res_9723 = res_9726;
        }
    }
    if (thread_active_10157) {
        *(__global float *) &mem_10077[gtid_9691 * 4] = res_9722;
    }
    if (thread_active_10157) {
        *(__global float *) &mem_10080[gtid_9691 * 4] = res_9723;
    }
}
__kernel void map_kernel_9753(float res_9333, float res_9334, float res_9347,
                              int32_t res_9381, int32_t res_9393, __global
                              unsigned char *voxels_mem_10060, __global
                              unsigned char *mem_10088, __global
                              unsigned char *mem_10091, __global
                              unsigned char *mem_10093, __global
                              unsigned char *mem_10095, __global
                              unsigned char *mem_10098, __global
                              unsigned char *mem_10101, __global
                              unsigned char *mem_10104, __global
                              unsigned char *mem_10125)
{
    int32_t wave_sizze_10171;
    int32_t group_sizze_10172;
    bool thread_active_10173;
    int32_t gtid_9746;
    int32_t global_tid_9753;
    int32_t local_tid_9754;
    int32_t group_id_9755;
    
    global_tid_9753 = get_global_id(0);
    local_tid_9754 = get_local_id(0);
    group_sizze_10172 = get_local_size(0);
    wave_sizze_10171 = LOCKSTEP_WIDTH;
    group_id_9755 = get_group_id(0);
    gtid_9746 = global_tid_9753;
    thread_active_10173 = slt32(gtid_9746, res_9393);
    
    float arg_9756;
    float arg_9757;
    bool res_9758;
    bool res_9759;
    float res_9760;
    bool cond_9763;
    float res_9764;
    int32_t res_9765;
    float res_9766;
    bool res_9767;
    float res_9768;
    float res_9775;
    float res_9776;
    bool cond_9799;
    bool res_9800;
    bool x_9801;
    bool cond_9802;
    bool res_9803;
    bool x_9804;
    bool cond_9805;
    bool res_9806;
    bool x_9807;
    bool x_9808;
    bool x_9809;
    bool y_9810;
    bool res_9811;
    bool x_9812;
    float y_9813;
    bool res_9814;
    float res_9817;
    float res_9818;
    float res_9819;
    float res_9820;
    int32_t res_9821;
    float res_9958;
    
    if (thread_active_10173) {
        arg_9756 = *(__global float *) &mem_10088[gtid_9746 * 4];
        arg_9757 = *(__global float *) &mem_10091[gtid_9746 * 4];
        res_9758 = *(__global bool *) &mem_10093[gtid_9746];
        res_9759 = *(__global bool *) &mem_10095[gtid_9746];
        res_9760 = *(__global float *) &mem_10098[gtid_9746 * 4];
        for (int32_t i_10174 = 0; i_10174 < res_9381; i_10174++) {
            *(__global float *) &mem_10101[(group_id_9755 * (res_9381 *
                                                             group_sizze_9748) +
                                            i_10174 * group_sizze_9748 +
                                            local_tid_9754) * 4] = -1.0F;
        }
        for (int32_t i_10175 = 0; i_10175 < res_9381; i_10175++) {
            *(__global int32_t *) &mem_10104[(group_id_9755 * (res_9381 *
                                                               group_sizze_9748) +
                                              i_10175 * group_sizze_9748 +
                                              local_tid_9754) * 4] = -1;
        }
        cond_9763 = res_9760 < 0.0F;
        if (cond_9763) {
            res_9764 = -1.0F;
        } else {
            res_9764 = 1.0F;
        }
        res_9765 = fptosi_f32_i32(arg_9756);
        res_9766 = sitofp_i32_f32(res_9765);
        res_9767 = 0.0F <= arg_9756;
        if (res_9767) {
            bool res_9769;
            float res_9770;
            
            res_9769 = res_9766 < arg_9756;
            if (res_9769) {
                res_9770 = res_9766;
            } else {
                res_9770 = arg_9756;
            }
            res_9768 = res_9770;
        } else {
            bool res_9771;
            float res_9772;
            
            res_9771 = arg_9756 < res_9766;
            if (res_9771) {
                int32_t res_9773;
                float res_9774;
                
                res_9773 = res_9765 - 1;
                res_9774 = sitofp_i32_f32(res_9773);
                res_9772 = res_9774;
            } else {
                res_9772 = arg_9756;
            }
            res_9768 = res_9772;
        }
        res_9775 = 1.0F + res_9768;
        if (cond_9763) {
            int32_t res_9777;
            float res_9778;
            bool res_9779;
            float res_9780;
            float res_9787;
            
            res_9777 = fptosi_f32_i32(arg_9757);
            res_9778 = sitofp_i32_f32(res_9777);
            res_9779 = 0.0F <= arg_9757;
            if (res_9779) {
                bool res_9781;
                float res_9782;
                
                res_9781 = res_9778 < arg_9757;
                if (res_9781) {
                    int32_t res_9783;
                    float res_9784;
                    
                    res_9783 = 1 + res_9777;
                    res_9784 = sitofp_i32_f32(res_9783);
                    res_9782 = res_9784;
                } else {
                    res_9782 = arg_9757;
                }
                res_9780 = res_9782;
            } else {
                bool res_9785;
                float res_9786;
                
                res_9785 = arg_9757 < res_9778;
                if (res_9785) {
                    res_9786 = res_9778;
                } else {
                    res_9786 = arg_9757;
                }
                res_9780 = res_9786;
            }
            res_9787 = res_9780 - 1.0F;
            res_9776 = res_9787;
        } else {
            int32_t res_9788;
            float res_9789;
            bool res_9790;
            float res_9791;
            float res_9798;
            
            res_9788 = fptosi_f32_i32(arg_9757);
            res_9789 = sitofp_i32_f32(res_9788);
            res_9790 = 0.0F <= arg_9757;
            if (res_9790) {
                bool res_9792;
                float res_9793;
                
                res_9792 = res_9789 < arg_9757;
                if (res_9792) {
                    res_9793 = res_9789;
                } else {
                    res_9793 = arg_9757;
                }
                res_9791 = res_9793;
            } else {
                bool res_9794;
                float res_9795;
                
                res_9794 = arg_9757 < res_9789;
                if (res_9794) {
                    int32_t res_9796;
                    float res_9797;
                    
                    res_9796 = res_9788 - 1;
                    res_9797 = sitofp_i32_f32(res_9796);
                    res_9795 = res_9797;
                } else {
                    res_9795 = arg_9757;
                }
                res_9791 = res_9795;
            }
            res_9798 = 1.0F + res_9791;
            res_9776 = res_9798;
        }
        cond_9799 = res_9347 <= arg_9756;
        res_9800 = arg_9756 < res_9334;
        x_9801 = cond_9799 && res_9800;
        cond_9802 = res_9347 < arg_9757;
        res_9803 = arg_9757 <= res_9334;
        x_9804 = cond_9802 && res_9803;
        cond_9805 = res_9347 <= arg_9757;
        res_9806 = arg_9757 < res_9334;
        x_9807 = cond_9805 && res_9806;
        x_9808 = cond_9763 && x_9804;
        x_9809 = !cond_9763;
        y_9810 = x_9807 && x_9809;
        res_9811 = x_9808 || y_9810;
        x_9812 = x_9801 && res_9811;
        y_9813 = 1.0F / res_9760;
        
        bool loop_while_9822;
        float focusPoint_9825;
        float focusPoint_9826;
        float anchorX_9827;
        float anchorY_9828;
        int32_t write_index_9829;
        
        loop_while_9822 = x_9812;
        focusPoint_9825 = arg_9756;
        focusPoint_9826 = arg_9757;
        anchorX_9827 = res_9775;
        anchorY_9828 = res_9776;
        write_index_9829 = 0;
        while (loop_while_9822) {
            float arg_9830 = res_9334 + focusPoint_9826;
            int32_t res_9831 = fptosi_f32_i32(arg_9830);
            float res_9832 = sitofp_i32_f32(res_9831);
            bool res_9833 = 0.0F <= arg_9830;
            float res_9834;
            
            if (res_9833) {
                bool res_9835;
                float res_9836;
                
                res_9835 = res_9832 < arg_9830;
                if (res_9835) {
                    res_9836 = res_9832;
                } else {
                    res_9836 = arg_9830;
                }
                res_9834 = res_9836;
            } else {
                bool res_9837;
                float res_9838;
                
                res_9837 = arg_9830 < res_9832;
                if (res_9837) {
                    int32_t res_9839;
                    float res_9840;
                    
                    res_9839 = res_9831 - 1;
                    res_9840 = sitofp_i32_f32(res_9839);
                    res_9838 = res_9840;
                } else {
                    res_9838 = arg_9830;
                }
                res_9834 = res_9838;
            }
            
            int32_t res_9841 = fptosi_f32_i32(focusPoint_9826);
            float res_9842 = sitofp_i32_f32(res_9841);
            bool res_9843 = 0.0F <= focusPoint_9826;
            float res_9844;
            
            if (res_9843) {
                bool res_9845;
                float res_9846;
                
                res_9845 = res_9842 < focusPoint_9826;
                if (res_9845) {
                    res_9846 = res_9842;
                } else {
                    res_9846 = focusPoint_9826;
                }
                res_9844 = res_9846;
            } else {
                bool res_9847;
                float res_9848;
                
                res_9847 = focusPoint_9826 < res_9842;
                if (res_9847) {
                    int32_t res_9849;
                    float res_9850;
                    
                    res_9849 = res_9841 - 1;
                    res_9850 = sitofp_i32_f32(res_9849);
                    res_9848 = res_9850;
                } else {
                    res_9848 = focusPoint_9826;
                }
                res_9844 = res_9848;
            }
            
            float x_9851 = focusPoint_9826 - res_9844;
            bool res_9852 = x_9851 == 0.0F;
            bool x_9853 = cond_9763 && res_9852;
            float res_9854;
            
            if (x_9853) {
                float res_9855 = res_9834 - 1.0F;
                
                res_9854 = res_9855;
            } else {
                res_9854 = res_9834;
            }
            
            float arg_9856 = res_9334 + focusPoint_9825;
            int32_t res_9857 = fptosi_f32_i32(arg_9856);
            float res_9858 = sitofp_i32_f32(res_9857);
            bool res_9859 = 0.0F <= arg_9856;
            float res_9860;
            
            if (res_9859) {
                bool res_9861;
                float res_9862;
                
                res_9861 = res_9858 < arg_9856;
                if (res_9861) {
                    res_9862 = res_9858;
                } else {
                    res_9862 = arg_9856;
                }
                res_9860 = res_9862;
            } else {
                bool res_9863;
                float res_9864;
                
                res_9863 = arg_9856 < res_9858;
                if (res_9863) {
                    int32_t res_9865;
                    float res_9866;
                    
                    res_9865 = res_9857 - 1;
                    res_9866 = sitofp_i32_f32(res_9865);
                    res_9864 = res_9866;
                } else {
                    res_9864 = arg_9856;
                }
                res_9860 = res_9864;
            }
            
            float y_9867 = res_9333 * res_9854;
            float arg_9868 = res_9860 + y_9867;
            int32_t res_9869 = fptosi_f32_i32(arg_9868);
            float res_9870;
            
            if (res_9759) {
                res_9870 = 1.0F;
            } else {
                float res_9871;
                
                if (res_9758) {
                    res_9871 = 0.0F;
                } else {
                    float x_9872;
                    float res_9873;
                    
                    x_9872 = anchorX_9827 - focusPoint_9825;
                    res_9873 = res_9760 * x_9872;
                    res_9871 = res_9873;
                }
                res_9870 = res_9871;
            }
            
            float res_9874;
            
            if (res_9759) {
                res_9874 = 0.0F;
            } else {
                float res_9875;
                
                if (res_9758) {
                    res_9875 = 1.0F;
                } else {
                    float x_9876;
                    float res_9877;
                    
                    x_9876 = anchorY_9828 - focusPoint_9826;
                    res_9877 = y_9813 * x_9876;
                    res_9875 = res_9877;
                }
                res_9874 = res_9875;
            }
            
            float res_9878 = focusPoint_9826 + res_9870;
            float res_9879 = focusPoint_9825 + res_9874;
            float x_9880 = anchorX_9827 - focusPoint_9825;
            float x_9881 = fpow32(x_9880, 2.0F);
            float x_9882 = res_9878 - focusPoint_9826;
            float y_9883 = fpow32(x_9882, 2.0F);
            float arg_9884 = x_9881 + y_9883;
            float res_9885;
            
            res_9885 = futrts_sqrt32(arg_9884);
            
            float x_9886 = res_9879 - focusPoint_9825;
            float x_9887 = fpow32(x_9886, 2.0F);
            float x_9888 = anchorY_9828 - focusPoint_9826;
            float y_9889 = fpow32(x_9888, 2.0F);
            float arg_9890 = x_9887 + y_9889;
            float res_9891;
            
            res_9891 = futrts_sqrt32(arg_9890);
            
            float res_9894;
            float res_9895;
            float res_9896;
            float res_9897;
            int32_t res_9898;
            
            if (res_9758) {
                float res_9901;
                int32_t res_9902;
                
                *(__global float *) &mem_10101[(group_id_9755 * (res_9381 *
                                                                 group_sizze_9748) +
                                                write_index_9829 *
                                                group_sizze_9748 +
                                                local_tid_9754) * 4] = res_9885;
                *(__global int32_t *) &mem_10104[(group_id_9755 * (res_9381 *
                                                                   group_sizze_9748) +
                                                  write_index_9829 *
                                                  group_sizze_9748 +
                                                  local_tid_9754) * 4] =
                    res_9869;
                res_9901 = 1.0F + anchorX_9827;
                res_9902 = 1 + write_index_9829;
                res_9894 = anchorX_9827;
                res_9895 = res_9878;
                res_9896 = res_9901;
                res_9897 = anchorY_9828;
                res_9898 = res_9902;
            } else {
                float res_9905;
                float res_9906;
                float res_9907;
                float res_9908;
                int32_t res_9909;
                
                if (res_9759) {
                    float res_9912;
                    int32_t res_9913;
                    
                    *(__global float *) &mem_10101[(group_id_9755 * (res_9381 *
                                                                     group_sizze_9748) +
                                                    write_index_9829 *
                                                    group_sizze_9748 +
                                                    local_tid_9754) * 4] =
                        res_9891;
                    *(__global int32_t *) &mem_10104[(group_id_9755 *
                                                      (res_9381 *
                                                       group_sizze_9748) +
                                                      write_index_9829 *
                                                      group_sizze_9748 +
                                                      local_tid_9754) * 4] =
                        res_9869;
                    res_9912 = res_9764 + anchorY_9828;
                    res_9913 = 1 + write_index_9829;
                    res_9905 = res_9879;
                    res_9906 = anchorY_9828;
                    res_9907 = anchorX_9827;
                    res_9908 = res_9912;
                    res_9909 = res_9913;
                } else {
                    float arg_9914;
                    float res_9915;
                    bool cond_9916;
                    float res_9919;
                    float res_9920;
                    float res_9921;
                    float res_9922;
                    int32_t res_9923;
                    
                    arg_9914 = res_9885 - res_9891;
                    res_9915 = (float) fabs(arg_9914);
                    cond_9916 = 1.0e-9F < res_9915;
                    if (cond_9916) {
                        bool cond_9924;
                        float res_9925;
                        float res_9926;
                        float res_9929;
                        float res_9930;
                        int32_t res_9931;
                        
                        cond_9924 = res_9885 < res_9891;
                        if (cond_9924) {
                            res_9925 = anchorX_9827;
                        } else {
                            res_9925 = res_9879;
                        }
                        if (cond_9924) {
                            res_9926 = res_9878;
                        } else {
                            res_9926 = anchorY_9828;
                        }
                        if (cond_9924) {
                            float res_9934;
                            int32_t res_9935;
                            
                            *(__global float *) &mem_10101[(group_id_9755 *
                                                            (res_9381 *
                                                             group_sizze_9748) +
                                                            write_index_9829 *
                                                            group_sizze_9748 +
                                                            local_tid_9754) *
                                                           4] = res_9885;
                            *(__global int32_t *) &mem_10104[(group_id_9755 *
                                                              (res_9381 *
                                                               group_sizze_9748) +
                                                              write_index_9829 *
                                                              group_sizze_9748 +
                                                              local_tid_9754) *
                                                             4] = res_9869;
                            res_9934 = 1.0F + anchorX_9827;
                            res_9935 = 1 + write_index_9829;
                            res_9929 = res_9934;
                            res_9930 = anchorY_9828;
                            res_9931 = res_9935;
                        } else {
                            float res_9938;
                            int32_t res_9939;
                            
                            *(__global float *) &mem_10101[(group_id_9755 *
                                                            (res_9381 *
                                                             group_sizze_9748) +
                                                            write_index_9829 *
                                                            group_sizze_9748 +
                                                            local_tid_9754) *
                                                           4] = res_9891;
                            *(__global int32_t *) &mem_10104[(group_id_9755 *
                                                              (res_9381 *
                                                               group_sizze_9748) +
                                                              write_index_9829 *
                                                              group_sizze_9748 +
                                                              local_tid_9754) *
                                                             4] = res_9869;
                            res_9938 = res_9764 + anchorY_9828;
                            res_9939 = 1 + write_index_9829;
                            res_9929 = anchorX_9827;
                            res_9930 = res_9938;
                            res_9931 = res_9939;
                        }
                        res_9919 = res_9925;
                        res_9920 = res_9926;
                        res_9921 = res_9929;
                        res_9922 = res_9930;
                        res_9923 = res_9931;
                    } else {
                        float res_9942;
                        float res_9943;
                        int32_t res_9944;
                        
                        *(__global float *) &mem_10101[(group_id_9755 *
                                                        (res_9381 *
                                                         group_sizze_9748) +
                                                        write_index_9829 *
                                                        group_sizze_9748 +
                                                        local_tid_9754) * 4] =
                            res_9885;
                        *(__global int32_t *) &mem_10104[(group_id_9755 *
                                                          (res_9381 *
                                                           group_sizze_9748) +
                                                          write_index_9829 *
                                                          group_sizze_9748 +
                                                          local_tid_9754) * 4] =
                            res_9869;
                        res_9942 = 1.0F + anchorX_9827;
                        res_9943 = res_9764 + anchorY_9828;
                        res_9944 = 1 + write_index_9829;
                        res_9919 = anchorX_9827;
                        res_9920 = res_9878;
                        res_9921 = res_9942;
                        res_9922 = res_9943;
                        res_9923 = res_9944;
                    }
                    res_9905 = res_9919;
                    res_9906 = res_9920;
                    res_9907 = res_9921;
                    res_9908 = res_9922;
                    res_9909 = res_9923;
                }
                res_9894 = res_9905;
                res_9895 = res_9906;
                res_9896 = res_9907;
                res_9897 = res_9908;
                res_9898 = res_9909;
            }
            
            bool cond_9945 = res_9347 <= res_9894;
            bool res_9946 = res_9894 < res_9334;
            bool x_9947 = cond_9945 && res_9946;
            bool cond_9948 = res_9347 < res_9895;
            bool res_9949 = res_9895 <= res_9334;
            bool x_9950 = cond_9948 && res_9949;
            bool cond_9951 = res_9347 <= res_9895;
            bool res_9952 = res_9895 < res_9334;
            bool x_9953 = cond_9951 && res_9952;
            bool x_9954 = cond_9763 && x_9950;
            bool y_9955 = x_9809 && x_9953;
            bool res_9956 = x_9954 || y_9955;
            bool x_9957 = x_9947 && res_9956;
            bool loop_while_tmp_10176 = x_9957;
            float focusPoint_tmp_10179 = res_9894;
            float focusPoint_tmp_10180 = res_9895;
            float anchorX_tmp_10181 = res_9896;
            float anchorY_tmp_10182 = res_9897;
            int32_t write_index_tmp_10183;
            
            write_index_tmp_10183 = res_9898;
            loop_while_9822 = loop_while_tmp_10176;
            focusPoint_9825 = focusPoint_tmp_10179;
            focusPoint_9826 = focusPoint_tmp_10180;
            anchorX_9827 = anchorX_tmp_10181;
            anchorY_9828 = anchorY_tmp_10182;
            write_index_9829 = write_index_tmp_10183;
        }
        res_9814 = loop_while_9822;
        res_9817 = focusPoint_9825;
        res_9818 = focusPoint_9826;
        res_9819 = anchorX_9827;
        res_9820 = anchorY_9828;
        res_9821 = write_index_9829;
        
        float x_9961 = 0.0F;
        
        for (int32_t chunk_offset_9960 = 0; chunk_offset_9960 < res_9381;
             chunk_offset_9960++) {
            float x_9970 = *(__global float *) &mem_10101[(group_id_9755 *
                                                           (res_9381 *
                                                            group_sizze_9748) +
                                                           chunk_offset_9960 *
                                                           group_sizze_9748 +
                                                           local_tid_9754) * 4];
            int32_t x_9971 = *(__global int32_t *) &mem_10104[(group_id_9755 *
                                                               (res_9381 *
                                                                group_sizze_9748) +
                                                               chunk_offset_9960 *
                                                               group_sizze_9748 +
                                                               local_tid_9754) *
                                                              4];
            bool cond_9973 = x_9971 == -1;
            float res_9974;
            
            if (cond_9973) {
                res_9974 = 0.0F;
            } else {
                float y_9975;
                float res_9976;
                
                y_9975 = *(__global float *) &voxels_mem_10060[x_9971 * 4];
                res_9976 = x_9970 * y_9975;
                res_9974 = res_9976;
            }
            
            float res_9978 = x_9961 + res_9974;
            float x_tmp_10184 = res_9978;
            
            x_9961 = x_tmp_10184;
        }
        res_9958 = x_9961;
    }
    if (thread_active_10173) {
        *(__global float *) &mem_10125[gtid_9746 * 4] = res_9958;
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
                                       all_sizes={"group_size_9651": {"class": "group_size", "value": None},
                                        "group_size_9667": {"class": "group_size", "value": None},
                                        "group_size_9679": {"class": "group_size", "value": None},
                                        "group_size_9692": {"class": "group_size", "value": None},
                                        "group_size_9747": {"class": "group_size", "value": None},
                                        "group_size_10007": {"class": "group_size", "value": None},
                                        "group_size_10160": {"class": "group_size", "value": None}})
    self.kernel_replicate_9377_var = program.kernel_replicate_9377
    self.map_kernel_10013_var = program.map_kernel_10013
    self.map_kernel_9657_var = program.map_kernel_9657
    self.map_kernel_9673_var = program.map_kernel_9673
    self.map_kernel_9685_var = program.map_kernel_9685
    self.map_kernel_9698_var = program.map_kernel_9698
    self.map_kernel_9753_var = program.map_kernel_9753
  def futhark_main(self, angles_mem_sizze_10055, angles_mem_10056,
                   rays_mem_sizze_10057, rays_mem_10058, voxels_mem_sizze_10059,
                   voxels_mem_10060, sizze_9323, sizze_9324, sizze_9325,
                   stepsizze_9329):
    res_9330 = sitofp_i32_f32(sizze_9325)
    res_9331 = futhark_sqrt32(res_9330)
    res_9332 = fptosi_f32_i32(res_9331)
    res_9333 = sitofp_i32_f32(res_9332)
    res_9334 = (res_9333 / np.float32(2.0))
    group_sizze_9680 = self.sizes["group_size_9679"]
    y_9681 = (group_sizze_9680 - np.int32(1))
    x_9682 = (sizze_9323 + y_9681)
    num_groups_9683 = squot32(x_9682, group_sizze_9680)
    num_threads_9684 = (group_sizze_9680 * num_groups_9683)
    binop_x_10062 = sext_i32_i64(sizze_9323)
    bytes_10061 = (np.int64(4) * binop_x_10062)
    mem_10063 = opencl_alloc(self, bytes_10061, "mem_10063")
    mem_10066 = opencl_alloc(self, bytes_10061, "mem_10066")
    if ((1 * (num_groups_9683 * group_sizze_9680)) != 0):
      self.map_kernel_9685_var.set_args(np.int32(sizze_9323), angles_mem_10056,
                                        mem_10063, mem_10066)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9685_var,
                                 (np.long((num_groups_9683 * group_sizze_9680)),),
                                 (np.long(group_sizze_9680),))
      if synchronous:
        self.queue.finish()
    nesting_sizze_9666 = (sizze_9323 * sizze_9324)
    group_sizze_9668 = self.sizes["group_size_9667"]
    y_9669 = (group_sizze_9668 - np.int32(1))
    x_9670 = (nesting_sizze_9666 + y_9669)
    num_groups_9671 = squot32(x_9670, group_sizze_9668)
    num_threads_9672 = (group_sizze_9668 * num_groups_9671)
    binop_x_10069 = sext_i32_i64(nesting_sizze_9666)
    bytes_10067 = (np.int64(4) * binop_x_10069)
    mem_10070 = opencl_alloc(self, bytes_10067, "mem_10070")
    if ((1 * (num_groups_9671 * group_sizze_9668)) != 0):
      self.map_kernel_9673_var.set_args(np.int32(sizze_9323),
                                        np.int32(sizze_9324), mem_10063,
                                        mem_10070)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9673_var,
                                 (np.long((num_groups_9671 * group_sizze_9668)),),
                                 (np.long(group_sizze_9668),))
      if synchronous:
        self.queue.finish()
    mem_10063 = None
    group_sizze_9652 = self.sizes["group_size_9651"]
    y_9653 = (group_sizze_9652 - np.int32(1))
    x_9654 = (y_9653 + nesting_sizze_9666)
    num_groups_9655 = squot32(x_9654, group_sizze_9652)
    num_threads_9656 = (group_sizze_9652 * num_groups_9655)
    mem_10074 = opencl_alloc(self, bytes_10067, "mem_10074")
    if ((1 * (num_groups_9655 * group_sizze_9652)) != 0):
      self.map_kernel_9657_var.set_args(np.int32(sizze_9323),
                                        np.int32(sizze_9324), mem_10066,
                                        mem_10074)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9657_var,
                                 (np.long((num_groups_9655 * group_sizze_9652)),),
                                 (np.long(group_sizze_9652),))
      if synchronous:
        self.queue.finish()
    mem_10066 = None
    res_9347 = (np.float32(0.0) - res_9334)
    group_sizze_9693 = self.sizes["group_size_9692"]
    y_9694 = (group_sizze_9693 - np.int32(1))
    x_9695 = (nesting_sizze_9666 + y_9694)
    num_groups_9696 = squot32(x_9695, group_sizze_9693)
    num_threads_9697 = (group_sizze_9693 * num_groups_9696)
    mem_10077 = opencl_alloc(self, bytes_10067, "mem_10077")
    mem_10080 = opencl_alloc(self, bytes_10067, "mem_10080")
    if ((1 * (num_groups_9696 * group_sizze_9693)) != 0):
      self.map_kernel_9698_var.set_args(np.int32(sizze_9324),
                                        np.float32(res_9334),
                                        np.float32(res_9347),
                                        np.int32(nesting_sizze_9666),
                                        rays_mem_10058, mem_10070, mem_10074,
                                        mem_10077, mem_10080)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9698_var,
                                 (np.long((num_groups_9696 * group_sizze_9693)),),
                                 (np.long(group_sizze_9693),))
      if synchronous:
        self.queue.finish()
    res_9376 = sdiv32(nesting_sizze_9666, stepsizze_9329)
    mem_10083 = opencl_alloc(self, np.int64(4), "mem_10083")
    group_sizze_10160 = self.sizes["group_size_10160"]
    num_groups_10161 = squot32(((np.int32(1) + sext_i32_i32(group_sizze_10160)) - np.int32(1)),
                               sext_i32_i32(group_sizze_10160))
    if ((1 * (num_groups_10161 * group_sizze_10160)) != 0):
      self.kernel_replicate_9377_var.set_args(mem_10083)
      cl.enqueue_nd_range_kernel(self.queue, self.kernel_replicate_9377_var,
                                 (np.long((num_groups_10161 * group_sizze_10160)),),
                                 (np.long(group_sizze_10160),))
      if synchronous:
        self.queue.finish()
    loop_cond_9378 = slt32(np.int32(0), res_9376)
    x_9379 = (np.float32(2.0) * res_9333)
    arg_9380 = (x_9379 - np.float32(1.0))
    res_9381 = fptosi_f32_i32(arg_9380)
    group_sizze_10008 = self.sizes["group_size_10007"]
    y_10009 = (group_sizze_10008 - np.int32(1))
    group_sizze_9748 = self.sizes["group_size_9747"]
    y_9749 = (group_sizze_9748 - np.int32(1))
    binop_x_10100 = sext_i32_i64(res_9381)
    bytes_10099 = (np.int64(4) * binop_x_10100)
    sizze_9386 = np.int32(1)
    output_mem_sizze_10084 = np.int64(4)
    output_mem_10085 = mem_10083
    loop_while_9387 = loop_cond_9378
    run_9389 = np.int32(0)
    while loop_while_9387:
      x_9390 = (np.int32(1) + run_9389)
      x_9391 = (stepsizze_9329 * x_9390)
      cond_9392 = sle32(nesting_sizze_9666, x_9391)
      if cond_9392:
        y_9394 = (stepsizze_9329 * run_9389)
        res_9395 = (nesting_sizze_9666 - y_9394)
        res_9393 = res_9395
      else:
        res_9393 = stepsizze_9329
      bounds_invalid_upwards_9396 = slt32(res_9393, np.int32(0))
      eq_x_zz_9399 = (np.int32(0) == res_9393)
      not_p_9400 = not(bounds_invalid_upwards_9396)
      p_and_eq_x_y_9401 = (eq_x_zz_9399 and not_p_9400)
      dim_zzero_9402 = (bounds_invalid_upwards_9396 or p_and_eq_x_y_9401)
      both_empty_9403 = (eq_x_zz_9399 and dim_zzero_9402)
      empty_or_match_9404 = (not_p_9400 or both_empty_9403)
      empty_or_match_cert_9405 = True
      assert empty_or_match_9404, ("Error at forwardprojection_jh.fut:6:1-10:58 -> forwardprojection_jh.fut:10:11-10:58 -> projection_lib.fut:186:185-186:193 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                                                "*",
                                                                                                                                                                                                                "[",
                                                                                                                                                                                                                res_9393,
                                                                                                                                                                                                                "]",
                                                                                                                                                                                                                "intrinsics.i32"))
      x_9406 = (stepsizze_9329 * run_9389)
      x_10010 = (res_9393 + y_10009)
      num_groups_10011 = squot32(x_10010, group_sizze_10008)
      num_threads_10012 = (group_sizze_10008 * num_groups_10011)
      binop_x_10087 = sext_i32_i64(res_9393)
      bytes_10086 = (np.int64(4) * binop_x_10087)
      mem_10088 = opencl_alloc(self, bytes_10086, "mem_10088")
      mem_10091 = opencl_alloc(self, bytes_10086, "mem_10091")
      mem_10093 = opencl_alloc(self, binop_x_10087, "mem_10093")
      mem_10095 = opencl_alloc(self, binop_x_10087, "mem_10095")
      mem_10098 = opencl_alloc(self, bytes_10086, "mem_10098")
      if ((1 * (num_groups_10011 * group_sizze_10008)) != 0):
        self.map_kernel_10013_var.set_args(np.int32(sizze_9324),
                                           np.int32(res_9393), np.int32(x_9406),
                                           mem_10070, mem_10074, mem_10077,
                                           mem_10080, mem_10088, mem_10091,
                                           mem_10093, mem_10095, mem_10098)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10013_var,
                                   (np.long((num_groups_10011 * group_sizze_10008)),),
                                   (np.long(group_sizze_10008),))
        if synchronous:
          self.queue.finish()
      x_9750 = (res_9393 + y_9749)
      num_groups_9751 = squot32(x_9750, group_sizze_9748)
      num_threads_9752 = (group_sizze_9748 * num_groups_9751)
      mem_10125 = opencl_alloc(self, bytes_10086, "mem_10125")
      num_threads64_10140 = sext_i32_i64(num_threads_9752)
      total_sizze_10141 = (bytes_10099 * num_threads64_10140)
      mem_10101 = opencl_alloc(self, total_sizze_10141, "mem_10101")
      total_sizze_10142 = (bytes_10099 * num_threads64_10140)
      mem_10104 = opencl_alloc(self, total_sizze_10142, "mem_10104")
      if ((1 * (num_groups_9751 * group_sizze_9748)) != 0):
        self.map_kernel_9753_var.set_args(np.float32(res_9333),
                                          np.float32(res_9334),
                                          np.float32(res_9347),
                                          np.int32(res_9381),
                                          np.int32(res_9393), voxels_mem_10060,
                                          mem_10088, mem_10091, mem_10093,
                                          mem_10095, mem_10098, mem_10101,
                                          mem_10104, mem_10125)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9753_var,
                                   (np.long((num_groups_9751 * group_sizze_9748)),),
                                   (np.long(group_sizze_9748),))
        if synchronous:
          self.queue.finish()
      mem_10088 = None
      mem_10091 = None
      mem_10093 = None
      mem_10095 = None
      mem_10098 = None
      mem_10101 = None
      mem_10104 = None
      conc_tmp_9630 = (sizze_9386 + res_9393)
      binop_x_10127 = sext_i32_i64(conc_tmp_9630)
      bytes_10126 = (np.int64(4) * binop_x_10127)
      mem_10128 = opencl_alloc(self, bytes_10126, "mem_10128")
      tmp_offs_10185 = np.int32(0)
      if ((sizze_9386 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10128, output_mem_10085,
                        dest_offset=np.long((tmp_offs_10185 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sizze_9386 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10185 = (tmp_offs_10185 + sizze_9386)
      if ((res_9393 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10128, mem_10125,
                        dest_offset=np.long((tmp_offs_10185 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((res_9393 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10185 = (tmp_offs_10185 + res_9393)
      mem_10125 = None
      loop_cond_9632 = slt32(x_9390, res_9376)
      sizze_tmp_10162 = conc_tmp_9630
      output_mem_sizze_tmp_10163 = bytes_10126
      output_mem_tmp_10164 = mem_10128
      loop_while_tmp_10165 = loop_cond_9632
      run_tmp_10167 = x_9390
      sizze_9386 = sizze_tmp_10162
      output_mem_sizze_10084 = output_mem_sizze_tmp_10163
      output_mem_10085 = output_mem_tmp_10164
      loop_while_9387 = loop_while_tmp_10165
      run_9389 = run_tmp_10167
    sizze_9382 = sizze_9386
    res_mem_sizze_10129 = output_mem_sizze_10084
    res_mem_10130 = output_mem_10085
    res_9383 = loop_while_9387
    res_9385 = run_9389
    mem_10070 = None
    mem_10074 = None
    mem_10077 = None
    mem_10080 = None
    mem_10083 = None
    j_m_i_9633 = (sizze_9382 - np.int32(1))
    x_9634 = abs(j_m_i_9633)
    empty_slice_9635 = (x_9634 == np.int32(0))
    m_9636 = (x_9634 - np.int32(1))
    i_p_m_t_s_9637 = (np.int32(1) + m_9636)
    zzero_leq_i_p_m_t_s_9638 = sle32(np.int32(0), i_p_m_t_s_9637)
    i_p_m_t_s_leq_w_9639 = slt32(i_p_m_t_s_9637, sizze_9382)
    i_lte_j_9640 = sle32(np.int32(1), sizze_9382)
    y_9641 = (zzero_leq_i_p_m_t_s_9638 and i_p_m_t_s_leq_w_9639)
    y_9642 = (i_lte_j_9640 and y_9641)
    ok_or_empty_9643 = (empty_slice_9635 or y_9642)
    index_certs_9644 = True
    assert ok_or_empty_9643, ("Error at forwardprojection_jh.fut:6:1-10:58 -> forwardprojection_jh.fut:10:11-10:58 -> projection_lib.fut:189:20-189:31 -> /futlib/array.fut:21:29-21:33: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                        np.int32(1),
                                                                                                                                                                                                        "] out of bounds for array of shape [",
                                                                                                                                                                                                        sizze_9382,
                                                                                                                                                                                                        "]."))
    binop_x_10132 = sext_i32_i64(x_9634)
    bytes_10131 = (np.int64(4) * binop_x_10132)
    mem_10133 = opencl_alloc(self, bytes_10131, "mem_10133")
    if ((x_9634 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_10133, res_mem_10130,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(4)),
                      byte_count=np.long((x_9634 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    res_mem_10130 = None
    out_arrsizze_10145 = x_9634
    out_memsizze_10144 = bytes_10131
    out_mem_10143 = mem_10133
    return (out_memsizze_10144, out_mem_10143, out_arrsizze_10145)
  def main(self, angles_mem_10056_ext, rays_mem_10058_ext, voxels_mem_10060_ext,
           stepsizze_9329_ext):
    try:
      assert ((type(angles_mem_10056_ext) in [np.ndarray,
                                              cl.array.Array]) and (angles_mem_10056_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9323 = np.int32(angles_mem_10056_ext.shape[0])
      angles_mem_sizze_10055 = np.int64(angles_mem_10056_ext.nbytes)
      if (type(angles_mem_10056_ext) == cl.array.Array):
        angles_mem_10056 = angles_mem_10056_ext.data
      else:
        angles_mem_10056 = opencl_alloc(self, angles_mem_sizze_10055,
                                        "angles_mem_10056")
        if (angles_mem_sizze_10055 != 0):
          cl.enqueue_copy(self.queue, angles_mem_10056,
                          normaliseArray(angles_mem_10056_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(angles_mem_10056_ext),
                                                                                                                            angles_mem_10056_ext))
    try:
      assert ((type(rays_mem_10058_ext) in [np.ndarray,
                                            cl.array.Array]) and (rays_mem_10058_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9324 = np.int32(rays_mem_10058_ext.shape[0])
      rays_mem_sizze_10057 = np.int64(rays_mem_10058_ext.nbytes)
      if (type(rays_mem_10058_ext) == cl.array.Array):
        rays_mem_10058 = rays_mem_10058_ext.data
      else:
        rays_mem_10058 = opencl_alloc(self, rays_mem_sizze_10057,
                                      "rays_mem_10058")
        if (rays_mem_sizze_10057 != 0):
          cl.enqueue_copy(self.queue, rays_mem_10058,
                          normaliseArray(rays_mem_10058_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(rays_mem_10058_ext),
                                                                                                                            rays_mem_10058_ext))
    try:
      assert ((type(voxels_mem_10060_ext) in [np.ndarray,
                                              cl.array.Array]) and (voxels_mem_10060_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9325 = np.int32(voxels_mem_10060_ext.shape[0])
      voxels_mem_sizze_10059 = np.int64(voxels_mem_10060_ext.nbytes)
      if (type(voxels_mem_10060_ext) == cl.array.Array):
        voxels_mem_10060 = voxels_mem_10060_ext.data
      else:
        voxels_mem_10060 = opencl_alloc(self, voxels_mem_sizze_10059,
                                        "voxels_mem_10060")
        if (voxels_mem_sizze_10059 != 0):
          cl.enqueue_copy(self.queue, voxels_mem_10060,
                          normaliseArray(voxels_mem_10060_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(voxels_mem_10060_ext),
                                                                                                                            voxels_mem_10060_ext))
    try:
      stepsizze_9329 = np.int32(ct.c_int32(stepsizze_9329_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(stepsizze_9329_ext),
                                                                                                                            stepsizze_9329_ext))
    (out_memsizze_10144, out_mem_10143,
     out_arrsizze_10145) = self.futhark_main(angles_mem_sizze_10055,
                                             angles_mem_10056,
                                             rays_mem_sizze_10057,
                                             rays_mem_10058,
                                             voxels_mem_sizze_10059,
                                             voxels_mem_10060, sizze_9323,
                                             sizze_9324, sizze_9325,
                                             stepsizze_9329)
    return cl.array.Array(self.queue, (out_arrsizze_10145,), ct.c_float,
                          data=out_mem_10143)