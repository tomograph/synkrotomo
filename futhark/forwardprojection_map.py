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
#define group_sizze_10205 (group_size_10204)
__kernel void kernel_replicate_9851(__global unsigned char *mem_10511)
{
    const uint replicate_gtid_9851 = get_global_id(0);
    
    if (replicate_gtid_9851 >= 1)
        return;
    *(__global float *) &mem_10511[replicate_gtid_9851 * 4] = 0.0F;
}
__kernel void map_kernel_10116(int32_t sizze_9797, int32_t sizze_9798, __global
                               unsigned char *mem_10494, __global
                               unsigned char *mem_10502)
{
    int32_t wave_sizze_10562;
    int32_t group_sizze_10563;
    bool thread_active_10564;
    int32_t gtid_10107;
    int32_t gtid_10108;
    int32_t global_tid_10116;
    int32_t local_tid_10117;
    int32_t group_id_10118;
    
    global_tid_10116 = get_global_id(0);
    local_tid_10117 = get_local_id(0);
    group_sizze_10563 = get_local_size(0);
    wave_sizze_10562 = LOCKSTEP_WIDTH;
    group_id_10118 = get_group_id(0);
    gtid_10107 = squot32(global_tid_10116, sizze_9798);
    gtid_10108 = global_tid_10116 - squot32(global_tid_10116, sizze_9798) *
        sizze_9798;
    thread_active_10564 = slt32(gtid_10107, sizze_9797) && slt32(gtid_10108,
                                                                 sizze_9798);
    
    float res_10119;
    
    if (thread_active_10564) {
        res_10119 = *(__global float *) &mem_10494[gtid_10107 * 4];
    }
    if (thread_active_10564) {
        *(__global float *) &mem_10502[(gtid_10107 * sizze_9798 + gtid_10108) *
                                       4] = res_10119;
    }
}
__kernel void map_kernel_10132(int32_t sizze_9797, int32_t sizze_9798, __global
                               unsigned char *mem_10491, __global
                               unsigned char *mem_10498)
{
    int32_t wave_sizze_10559;
    int32_t group_sizze_10560;
    bool thread_active_10561;
    int32_t gtid_10123;
    int32_t gtid_10124;
    int32_t global_tid_10132;
    int32_t local_tid_10133;
    int32_t group_id_10134;
    
    global_tid_10132 = get_global_id(0);
    local_tid_10133 = get_local_id(0);
    group_sizze_10560 = get_local_size(0);
    wave_sizze_10559 = LOCKSTEP_WIDTH;
    group_id_10134 = get_group_id(0);
    gtid_10123 = squot32(global_tid_10132, sizze_9798);
    gtid_10124 = global_tid_10132 - squot32(global_tid_10132, sizze_9798) *
        sizze_9798;
    thread_active_10561 = slt32(gtid_10123, sizze_9797) && slt32(gtid_10124,
                                                                 sizze_9798);
    
    float res_10135;
    
    if (thread_active_10561) {
        res_10135 = *(__global float *) &mem_10491[gtid_10123 * 4];
    }
    if (thread_active_10561) {
        *(__global float *) &mem_10498[(gtid_10123 * sizze_9798 + gtid_10124) *
                                       4] = res_10135;
    }
}
__kernel void map_kernel_10144(int32_t sizze_9797, __global
                               unsigned char *angles_mem_10484, __global
                               unsigned char *mem_10491, __global
                               unsigned char *mem_10494)
{
    int32_t wave_sizze_10556;
    int32_t group_sizze_10557;
    bool thread_active_10558;
    int32_t gtid_10137;
    int32_t global_tid_10144;
    int32_t local_tid_10145;
    int32_t group_id_10146;
    
    global_tid_10144 = get_global_id(0);
    local_tid_10145 = get_local_id(0);
    group_sizze_10557 = get_local_size(0);
    wave_sizze_10556 = LOCKSTEP_WIDTH;
    group_id_10146 = get_group_id(0);
    gtid_10137 = global_tid_10144;
    thread_active_10558 = slt32(gtid_10137, sizze_9797);
    
    float x_10147;
    float res_10148;
    float res_10149;
    
    if (thread_active_10558) {
        x_10147 = *(__global float *) &angles_mem_10484[gtid_10137 * 4];
        res_10148 = futrts_sin32(x_10147);
        res_10149 = futrts_cos32(x_10147);
    }
    if (thread_active_10558) {
        *(__global float *) &mem_10491[gtid_10137 * 4] = res_10148;
    }
    if (thread_active_10558) {
        *(__global float *) &mem_10494[gtid_10137 * 4] = res_10149;
    }
}
__kernel void map_kernel_10157(int32_t sizze_9798, float res_9808,
                               float res_9821, int32_t nesting_sizze_10125,
                               __global unsigned char *rays_mem_10486, __global
                               unsigned char *mem_10498, __global
                               unsigned char *mem_10502, __global
                               unsigned char *mem_10505, __global
                               unsigned char *mem_10508)
{
    int32_t wave_sizze_10565;
    int32_t group_sizze_10566;
    bool thread_active_10567;
    int32_t gtid_10150;
    int32_t global_tid_10157;
    int32_t local_tid_10158;
    int32_t group_id_10159;
    
    global_tid_10157 = get_global_id(0);
    local_tid_10158 = get_local_id(0);
    group_sizze_10566 = get_local_size(0);
    wave_sizze_10565 = LOCKSTEP_WIDTH;
    group_id_10159 = get_group_id(0);
    gtid_10150 = global_tid_10157;
    thread_active_10567 = slt32(gtid_10150, nesting_sizze_10125);
    
    int32_t new_index_10464;
    int32_t binop_y_10466;
    int32_t new_index_10467;
    float x_10160;
    float x_10161;
    float x_10162;
    bool cond_10163;
    float res_10164;
    bool cond_10168;
    float res_10169;
    float res_10173;
    float res_10177;
    bool cond_10178;
    bool res_10179;
    bool x_10180;
    float res_10181;
    float res_10182;
    
    if (thread_active_10567) {
        new_index_10464 = squot32(gtid_10150, sizze_9798);
        binop_y_10466 = sizze_9798 * new_index_10464;
        new_index_10467 = gtid_10150 - binop_y_10466;
        x_10160 = *(__global float *) &mem_10498[(new_index_10464 * sizze_9798 +
                                                  new_index_10467) * 4];
        x_10161 = *(__global float *) &mem_10502[(new_index_10464 * sizze_9798 +
                                                  new_index_10467) * 4];
        x_10162 = *(__global float *) &rays_mem_10486[new_index_10467 * 4];
        cond_10163 = x_10160 == 0.0F;
        if (cond_10163) {
            res_10164 = x_10162;
        } else {
            float y_10165;
            float x_10166;
            float res_10167;
            
            y_10165 = res_9821 * x_10161;
            x_10166 = x_10162 - y_10165;
            res_10167 = x_10166 / x_10160;
            res_10164 = res_10167;
        }
        cond_10168 = x_10161 == 0.0F;
        if (cond_10168) {
            res_10169 = x_10162;
        } else {
            float y_10170;
            float x_10171;
            float res_10172;
            
            y_10170 = res_9821 * x_10160;
            x_10171 = x_10162 - y_10170;
            res_10172 = x_10171 / x_10161;
            res_10169 = res_10172;
        }
        if (cond_10168) {
            res_10173 = x_10162;
        } else {
            float y_10174;
            float x_10175;
            float res_10176;
            
            y_10174 = res_9808 * x_10160;
            x_10175 = x_10162 - y_10174;
            res_10176 = x_10175 / x_10161;
            res_10173 = res_10176;
        }
        res_10177 = (float) fabs(res_10164);
        cond_10178 = res_10177 <= res_9808;
        res_10179 = !cond_10163;
        x_10180 = cond_10178 && res_10179;
        if (x_10180) {
            res_10181 = res_9821;
            res_10182 = res_10164;
        } else {
            bool cond_10183;
            float res_10184;
            float res_10185;
            
            cond_10183 = res_10169 <= res_10173;
            if (cond_10183) {
                res_10184 = res_10169;
            } else {
                res_10184 = res_10173;
            }
            if (cond_10183) {
                res_10185 = res_9821;
            } else {
                res_10185 = res_9808;
            }
            res_10181 = res_10184;
            res_10182 = res_10185;
        }
    }
    if (thread_active_10567) {
        *(__global float *) &mem_10505[gtid_10150 * 4] = res_10181;
    }
    if (thread_active_10567) {
        *(__global float *) &mem_10508[gtid_10150 * 4] = res_10182;
    }
}
__kernel void map_kernel_10210(float res_9808, float res_9821, int32_t res_9856,
                               int32_t arg_9857, int32_t res_9873, __global
                               unsigned char *voxels_mem_10488, __global
                               unsigned char *mem_10516, __global
                               unsigned char *mem_10519, __global
                               unsigned char *mem_10521, __global
                               unsigned char *mem_10524, __global
                               unsigned char *mem_10526, __global
                               unsigned char *mem_10529, __global
                               unsigned char *mem_10532, __global
                               unsigned char *mem_10535)
{
    int32_t wave_sizze_10581;
    int32_t group_sizze_10582;
    bool thread_active_10583;
    int32_t gtid_10203;
    int32_t global_tid_10210;
    int32_t local_tid_10211;
    int32_t group_id_10212;
    
    global_tid_10210 = get_global_id(0);
    local_tid_10211 = get_local_id(0);
    group_sizze_10582 = get_local_size(0);
    wave_sizze_10581 = LOCKSTEP_WIDTH;
    group_id_10212 = get_group_id(0);
    gtid_10203 = global_tid_10210;
    thread_active_10583 = slt32(gtid_10203, res_9873);
    
    float arg_10213;
    float arg_10214;
    bool res_10215;
    float res_10216;
    bool cond_10217;
    bool cond_10220;
    bool res_10221;
    bool x_10222;
    bool cond_10223;
    bool res_10224;
    bool x_10225;
    bool cond_10226;
    bool res_10227;
    bool x_10228;
    bool x_10229;
    bool x_10230;
    bool y_10231;
    bool res_10232;
    bool x_10233;
    bool cond_10234;
    bool cond_10235;
    float y_10236;
    bool res_10237;
    float res_10240;
    float res_10241;
    int32_t res_10242;
    float res_10314;
    
    if (thread_active_10583) {
        arg_10213 = *(__global float *) &mem_10516[gtid_10203 * 4];
        arg_10214 = *(__global float *) &mem_10519[gtid_10203 * 4];
        res_10215 = *(__global bool *) &mem_10521[gtid_10203];
        res_10216 = *(__global float *) &mem_10524[gtid_10203 * 4];
        cond_10217 = *(__global bool *) &mem_10526[gtid_10203];
        for (int32_t i_10584 = 0; i_10584 < res_9856; i_10584++) {
            *(__global float *) &mem_10529[(group_id_10212 * (res_9856 *
                                                              group_sizze_10205) +
                                            i_10584 * group_sizze_10205 +
                                            local_tid_10211) * 4] = -INFINITY;
        }
        for (int32_t i_10585 = 0; i_10585 < res_9856; i_10585++) {
            *(__global float *) &mem_10532[(group_id_10212 * (res_9856 *
                                                              group_sizze_10205) +
                                            i_10585 * group_sizze_10205 +
                                            local_tid_10211) * 4] = -INFINITY;
        }
        cond_10220 = res_9821 <= arg_10213;
        res_10221 = arg_10213 < res_9808;
        x_10222 = cond_10220 && res_10221;
        cond_10223 = res_9821 < arg_10214;
        res_10224 = arg_10214 <= res_9808;
        x_10225 = cond_10223 && res_10224;
        cond_10226 = res_9821 <= arg_10214;
        res_10227 = arg_10214 < res_9808;
        x_10228 = cond_10226 && res_10227;
        x_10229 = cond_10217 && x_10225;
        x_10230 = !cond_10217;
        y_10231 = x_10228 && x_10230;
        res_10232 = x_10229 || y_10231;
        x_10233 = x_10222 && res_10232;
        cond_10234 = res_10216 == 0.0F;
        cond_10235 = res_10216 == 1.0F;
        y_10236 = 1.0F / res_10216;
        
        bool loop_while_10243;
        float focusPoint_10246;
        float focusPoint_10247;
        int32_t write_index_10248;
        
        loop_while_10243 = x_10233;
        focusPoint_10246 = arg_10213;
        focusPoint_10247 = arg_10214;
        write_index_10248 = 0;
        while (loop_while_10243) {
            float res_10249;
            
            if (res_10215) {
                res_10249 = focusPoint_10246;
            } else {
                int32_t res_10250;
                float res_10251;
                bool res_10252;
                float res_10253;
                float res_10260;
                
                res_10250 = fptosi_f32_i32(focusPoint_10246);
                res_10251 = sitofp_i32_f32(res_10250);
                res_10252 = 0.0F <= focusPoint_10246;
                if (res_10252) {
                    bool res_10254;
                    float res_10255;
                    
                    res_10254 = res_10251 < focusPoint_10246;
                    if (res_10254) {
                        res_10255 = res_10251;
                    } else {
                        res_10255 = focusPoint_10246;
                    }
                    res_10253 = res_10255;
                } else {
                    bool res_10256;
                    float res_10257;
                    
                    res_10256 = focusPoint_10246 < res_10251;
                    if (res_10256) {
                        int32_t res_10258;
                        float res_10259;
                        
                        res_10258 = res_10250 - 1;
                        res_10259 = sitofp_i32_f32(res_10258);
                        res_10257 = res_10259;
                    } else {
                        res_10257 = focusPoint_10246;
                    }
                    res_10253 = res_10257;
                }
                res_10260 = 1.0F + res_10253;
                res_10249 = res_10260;
            }
            
            float res_10261;
            
            if (cond_10234) {
                res_10261 = focusPoint_10247;
            } else {
                float res_10262;
                
                if (cond_10217) {
                    int32_t res_10263;
                    float res_10264;
                    bool res_10265;
                    float res_10266;
                    float res_10273;
                    
                    res_10263 = fptosi_f32_i32(focusPoint_10247);
                    res_10264 = sitofp_i32_f32(res_10263);
                    res_10265 = 0.0F <= focusPoint_10247;
                    if (res_10265) {
                        bool res_10267;
                        float res_10268;
                        
                        res_10267 = res_10264 < focusPoint_10247;
                        if (res_10267) {
                            int32_t res_10269;
                            float res_10270;
                            
                            res_10269 = 1 + res_10263;
                            res_10270 = sitofp_i32_f32(res_10269);
                            res_10268 = res_10270;
                        } else {
                            res_10268 = focusPoint_10247;
                        }
                        res_10266 = res_10268;
                    } else {
                        bool res_10271;
                        float res_10272;
                        
                        res_10271 = focusPoint_10247 < res_10264;
                        if (res_10271) {
                            res_10272 = res_10264;
                        } else {
                            res_10272 = focusPoint_10247;
                        }
                        res_10266 = res_10272;
                    }
                    res_10273 = res_10266 - 1.0F;
                    res_10262 = res_10273;
                } else {
                    int32_t res_10274;
                    float res_10275;
                    bool res_10276;
                    float res_10277;
                    float res_10284;
                    
                    res_10274 = fptosi_f32_i32(focusPoint_10247);
                    res_10275 = sitofp_i32_f32(res_10274);
                    res_10276 = 0.0F <= focusPoint_10247;
                    if (res_10276) {
                        bool res_10278;
                        float res_10279;
                        
                        res_10278 = res_10275 < focusPoint_10247;
                        if (res_10278) {
                            res_10279 = res_10275;
                        } else {
                            res_10279 = focusPoint_10247;
                        }
                        res_10277 = res_10279;
                    } else {
                        bool res_10280;
                        float res_10281;
                        
                        res_10280 = focusPoint_10247 < res_10275;
                        if (res_10280) {
                            int32_t res_10282;
                            float res_10283;
                            
                            res_10282 = res_10274 - 1;
                            res_10283 = sitofp_i32_f32(res_10282);
                            res_10281 = res_10283;
                        } else {
                            res_10281 = focusPoint_10247;
                        }
                        res_10277 = res_10281;
                    }
                    res_10284 = 1.0F + res_10277;
                    res_10262 = res_10284;
                }
                res_10261 = res_10262;
            }
            
            float res_10285;
            
            if (cond_10235) {
                res_10285 = 1.0F;
            } else {
                float res_10286;
                
                if (cond_10234) {
                    res_10286 = 0.0F;
                } else {
                    float x_10287;
                    float res_10288;
                    
                    x_10287 = res_10249 - focusPoint_10246;
                    res_10288 = res_10216 * x_10287;
                    res_10286 = res_10288;
                }
                res_10285 = res_10286;
            }
            
            float res_10289;
            
            if (cond_10235) {
                res_10289 = 0.0F;
            } else {
                float res_10290;
                
                if (cond_10234) {
                    res_10290 = 1.0F;
                } else {
                    float x_10291;
                    float res_10292;
                    
                    x_10291 = res_10261 - focusPoint_10247;
                    res_10292 = y_10236 * x_10291;
                    res_10290 = res_10292;
                }
                res_10289 = res_10290;
            }
            
            float res_10293 = focusPoint_10247 + res_10285;
            float res_10294 = focusPoint_10246 + res_10289;
            bool cond_10295 = res_10249 < res_10294;
            float res_10296;
            
            if (cond_10295) {
                res_10296 = res_10249;
            } else {
                res_10296 = res_10294;
            }
            
            float res_10297;
            
            if (cond_10295) {
                res_10297 = res_10293;
            } else {
                res_10297 = res_10261;
            }
            *(__global float *) &mem_10529[(group_id_10212 * (res_9856 *
                                                              group_sizze_10205) +
                                            write_index_10248 *
                                            group_sizze_10205 +
                                            local_tid_10211) * 4] =
                focusPoint_10246;
            *(__global float *) &mem_10532[(group_id_10212 * (res_9856 *
                                                              group_sizze_10205) +
                                            write_index_10248 *
                                            group_sizze_10205 +
                                            local_tid_10211) * 4] =
                focusPoint_10247;
            
            int32_t res_10300 = 1 + write_index_10248;
            bool cond_10301 = res_9821 <= res_10296;
            bool res_10302 = res_10296 < res_9808;
            bool x_10303 = cond_10301 && res_10302;
            bool cond_10304 = res_9821 < res_10297;
            bool res_10305 = res_10297 <= res_9808;
            bool x_10306 = cond_10304 && res_10305;
            bool cond_10307 = res_9821 <= res_10297;
            bool res_10308 = res_10297 < res_9808;
            bool x_10309 = cond_10307 && res_10308;
            bool x_10310 = cond_10217 && x_10306;
            bool y_10311 = x_10230 && x_10309;
            bool res_10312 = x_10310 || y_10311;
            bool x_10313 = x_10303 && res_10312;
            bool loop_while_tmp_10586 = x_10313;
            float focusPoint_tmp_10589 = res_10296;
            float focusPoint_tmp_10590 = res_10297;
            int32_t write_index_tmp_10591;
            
            write_index_tmp_10591 = res_10300;
            loop_while_10243 = loop_while_tmp_10586;
            focusPoint_10246 = focusPoint_tmp_10589;
            focusPoint_10247 = focusPoint_tmp_10590;
            write_index_10248 = write_index_tmp_10591;
        }
        res_10237 = loop_while_10243;
        res_10240 = focusPoint_10246;
        res_10241 = focusPoint_10247;
        res_10242 = write_index_10248;
        
        float x_10317 = 0.0F;
        
        for (int32_t chunk_offset_10316 = 0; chunk_offset_10316 < arg_9857;
             chunk_offset_10316++) {
            float arg_10326 = *(__global float *) &mem_10529[(group_id_10212 *
                                                              (res_9856 *
                                                               group_sizze_10205) +
                                                              chunk_offset_10316 *
                                                              group_sizze_10205 +
                                                              local_tid_10211) *
                                                             4];
            float arg_10327 = *(__global float *) &mem_10532[(group_id_10212 *
                                                              (res_9856 *
                                                               group_sizze_10205) +
                                                              chunk_offset_10316 *
                                                              group_sizze_10205 +
                                                              local_tid_10211) *
                                                             4];
            bool cond_10328 = res_9821 <= arg_10326;
            bool res_10329 = arg_10326 < res_9808;
            bool x_10330 = cond_10328 && res_10329;
            bool cond_10331 = res_9821 < arg_10327;
            bool res_10332 = arg_10327 <= res_9808;
            bool x_10333 = cond_10331 && res_10332;
            bool cond_10334 = res_9821 <= arg_10327;
            bool res_10335 = arg_10327 < res_9808;
            bool x_10336 = cond_10334 && res_10335;
            bool x_10337 = cond_10217 && x_10333;
            bool y_10338 = x_10230 && x_10336;
            bool res_10339 = x_10337 || y_10338;
            bool x_10340 = x_10330 && res_10339;
            bool cond_10341 = !x_10340;
            int32_t res_10342;
            
            if (cond_10341) {
                res_10342 = -1;
            } else {
                float arg_10343;
                int32_t res_10344;
                float res_10345;
                bool res_10346;
                float res_10347;
                int32_t res_10354;
                float res_10355;
                bool res_10356;
                float res_10357;
                float x_10364;
                bool res_10365;
                bool x_10366;
                float res_10367;
                float arg_10369;
                int32_t res_10370;
                float res_10371;
                bool res_10372;
                float res_10373;
                float x_10380;
                float y_10381;
                float arg_10382;
                int32_t res_10383;
                
                arg_10343 = res_9808 + arg_10327;
                res_10344 = fptosi_f32_i32(arg_10343);
                res_10345 = sitofp_i32_f32(res_10344);
                res_10346 = 0.0F <= arg_10343;
                if (res_10346) {
                    bool res_10348;
                    float res_10349;
                    
                    res_10348 = res_10345 < arg_10343;
                    if (res_10348) {
                        res_10349 = res_10345;
                    } else {
                        res_10349 = arg_10343;
                    }
                    res_10347 = res_10349;
                } else {
                    bool res_10350;
                    float res_10351;
                    
                    res_10350 = arg_10343 < res_10345;
                    if (res_10350) {
                        int32_t res_10352;
                        float res_10353;
                        
                        res_10352 = res_10344 - 1;
                        res_10353 = sitofp_i32_f32(res_10352);
                        res_10351 = res_10353;
                    } else {
                        res_10351 = arg_10343;
                    }
                    res_10347 = res_10351;
                }
                res_10354 = fptosi_f32_i32(arg_10327);
                res_10355 = sitofp_i32_f32(res_10354);
                res_10356 = 0.0F <= arg_10327;
                if (res_10356) {
                    bool res_10358;
                    float res_10359;
                    
                    res_10358 = res_10355 < arg_10327;
                    if (res_10358) {
                        res_10359 = res_10355;
                    } else {
                        res_10359 = arg_10327;
                    }
                    res_10357 = res_10359;
                } else {
                    bool res_10360;
                    float res_10361;
                    
                    res_10360 = arg_10327 < res_10355;
                    if (res_10360) {
                        int32_t res_10362;
                        float res_10363;
                        
                        res_10362 = res_10354 - 1;
                        res_10363 = sitofp_i32_f32(res_10362);
                        res_10361 = res_10363;
                    } else {
                        res_10361 = arg_10327;
                    }
                    res_10357 = res_10361;
                }
                x_10364 = arg_10327 - res_10357;
                res_10365 = x_10364 == 0.0F;
                x_10366 = cond_10217 && res_10365;
                if (x_10366) {
                    float res_10368 = res_10347 - 1.0F;
                    
                    res_10367 = res_10368;
                } else {
                    res_10367 = res_10347;
                }
                arg_10369 = res_9808 + arg_10326;
                res_10370 = fptosi_f32_i32(arg_10369);
                res_10371 = sitofp_i32_f32(res_10370);
                res_10372 = 0.0F <= arg_10369;
                if (res_10372) {
                    bool res_10374;
                    float res_10375;
                    
                    res_10374 = res_10371 < arg_10369;
                    if (res_10374) {
                        res_10375 = res_10371;
                    } else {
                        res_10375 = arg_10369;
                    }
                    res_10373 = res_10375;
                } else {
                    bool res_10376;
                    float res_10377;
                    
                    res_10376 = arg_10369 < res_10371;
                    if (res_10376) {
                        int32_t res_10378;
                        float res_10379;
                        
                        res_10378 = res_10370 - 1;
                        res_10379 = sitofp_i32_f32(res_10378);
                        res_10377 = res_10379;
                    } else {
                        res_10377 = arg_10369;
                    }
                    res_10373 = res_10377;
                }
                x_10380 = 2.0F * res_9808;
                y_10381 = res_10367 * x_10380;
                arg_10382 = res_10373 + y_10381;
                res_10383 = fptosi_f32_i32(arg_10382);
                res_10342 = res_10383;
            }
            
            int32_t i_10384 = 1 + chunk_offset_10316;
            float arg_10385 = *(__global float *) &mem_10529[(group_id_10212 *
                                                              (res_9856 *
                                                               group_sizze_10205) +
                                                              i_10384 *
                                                              group_sizze_10205 +
                                                              local_tid_10211) *
                                                             4];
            float arg_10386 = *(__global float *) &mem_10532[(group_id_10212 *
                                                              (res_9856 *
                                                               group_sizze_10205) +
                                                              i_10384 *
                                                              group_sizze_10205 +
                                                              local_tid_10211) *
                                                             4];
            bool cond_10387 = res_9821 <= arg_10385;
            bool res_10388 = arg_10385 < res_9808;
            bool x_10389 = cond_10387 && res_10388;
            bool cond_10390 = res_9821 < arg_10386;
            bool res_10391 = arg_10386 <= res_9808;
            bool x_10392 = cond_10390 && res_10391;
            bool cond_10393 = res_9821 <= arg_10386;
            bool res_10394 = arg_10386 < res_9808;
            bool x_10395 = cond_10393 && res_10394;
            bool x_10396 = cond_10217 && x_10392;
            bool y_10397 = x_10230 && x_10395;
            bool res_10398 = x_10396 || y_10397;
            bool x_10399 = x_10389 && res_10398;
            float res_10400;
            
            if (x_10399) {
                float x_10401;
                float x_10402;
                float x_10403;
                float y_10404;
                float arg_10405;
                float res_10406;
                
                x_10401 = arg_10385 - arg_10326;
                x_10402 = fpow32(x_10401, 2.0F);
                x_10403 = arg_10386 - arg_10327;
                y_10404 = fpow32(x_10403, 2.0F);
                arg_10405 = x_10402 + y_10404;
                res_10406 = futrts_sqrt32(arg_10405);
                res_10400 = res_10406;
            } else {
                res_10400 = 0.0F;
            }
            
            bool cond_10407 = res_10342 == -1;
            float res_10408;
            
            if (cond_10407) {
                res_10408 = 0.0F;
            } else {
                float y_10409;
                float res_10410;
                
                y_10409 = *(__global float *) &voxels_mem_10488[res_10342 * 4];
                res_10410 = res_10400 * y_10409;
                res_10408 = res_10410;
            }
            
            float res_10412 = x_10317 + res_10408;
            float x_tmp_10592 = res_10412;
            
            x_10317 = x_tmp_10592;
        }
        res_10314 = x_10317;
    }
    if (thread_active_10583) {
        *(__global float *) &mem_10535[gtid_10203 * 4] = res_10314;
    }
}
__kernel void map_kernel_10447(int32_t sizze_9798, int32_t res_9873,
                               int32_t x_9886, __global
                               unsigned char *mem_10498, __global
                               unsigned char *mem_10502, __global
                               unsigned char *mem_10505, __global
                               unsigned char *mem_10508, __global
                               unsigned char *mem_10516, __global
                               unsigned char *mem_10519, __global
                               unsigned char *mem_10521, __global
                               unsigned char *mem_10524, __global
                               unsigned char *mem_10526)
{
    int32_t wave_sizze_10578;
    int32_t group_sizze_10579;
    bool thread_active_10580;
    int32_t gtid_10440;
    int32_t global_tid_10447;
    int32_t local_tid_10448;
    int32_t group_id_10449;
    
    global_tid_10447 = get_global_id(0);
    local_tid_10448 = get_local_id(0);
    group_sizze_10579 = get_local_size(0);
    wave_sizze_10578 = LOCKSTEP_WIDTH;
    group_id_10449 = get_group_id(0);
    gtid_10440 = global_tid_10447;
    thread_active_10580 = slt32(gtid_10440, res_9873);
    
    int32_t i_10451;
    int32_t new_index_10452;
    int32_t binop_y_10453;
    int32_t new_index_10454;
    float arg_10455;
    float arg_10456;
    float arg_10457;
    float arg_10458;
    float res_10459;
    bool res_10460;
    float y_10461;
    float res_10462;
    bool cond_10463;
    
    if (thread_active_10580) {
        i_10451 = x_9886 + gtid_10440;
        new_index_10452 = squot32(i_10451, sizze_9798);
        binop_y_10453 = sizze_9798 * new_index_10452;
        new_index_10454 = i_10451 - binop_y_10453;
        arg_10455 = *(__global float *) &mem_10498[(new_index_10452 *
                                                    sizze_9798 +
                                                    new_index_10454) * 4];
        arg_10456 = *(__global float *) &mem_10502[(new_index_10452 *
                                                    sizze_9798 +
                                                    new_index_10454) * 4];
        arg_10457 = *(__global float *) &mem_10505[i_10451 * 4];
        arg_10458 = *(__global float *) &mem_10508[i_10451 * 4];
        res_10459 = (float) fabs(arg_10456);
        res_10460 = res_10459 == 1.0F;
        y_10461 = 0.0F - arg_10455;
        res_10462 = arg_10456 / y_10461;
        cond_10463 = res_10462 < 0.0F;
    }
    if (thread_active_10580) {
        *(__global float *) &mem_10516[gtid_10440 * 4] = arg_10457;
    }
    if (thread_active_10580) {
        *(__global float *) &mem_10519[gtid_10440 * 4] = arg_10458;
    }
    if (thread_active_10580) {
        *(__global bool *) &mem_10521[gtid_10440] = res_10460;
    }
    if (thread_active_10580) {
        *(__global float *) &mem_10524[gtid_10440 * 4] = res_10462;
    }
    if (thread_active_10580) {
        *(__global bool *) &mem_10526[gtid_10440] = cond_10463;
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
class forwardprojection_map:
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
                                       all_sizes={"group_size_10110": {"class": "group_size", "value": None},
                                        "group_size_10126": {"class": "group_size", "value": None},
                                        "group_size_10138": {"class": "group_size", "value": None},
                                        "group_size_10151": {"class": "group_size", "value": None},
                                        "group_size_10204": {"class": "group_size", "value": None},
                                        "group_size_10441": {"class": "group_size", "value": None},
                                        "group_size_10570": {"class": "group_size", "value": None}})
    self.kernel_replicate_9851_var = program.kernel_replicate_9851
    self.map_kernel_10116_var = program.map_kernel_10116
    self.map_kernel_10132_var = program.map_kernel_10132
    self.map_kernel_10144_var = program.map_kernel_10144
    self.map_kernel_10157_var = program.map_kernel_10157
    self.map_kernel_10210_var = program.map_kernel_10210
    self.map_kernel_10447_var = program.map_kernel_10447
  def futhark_main(self, angles_mem_sizze_10483, angles_mem_10484,
                   rays_mem_sizze_10485, rays_mem_10486, voxels_mem_sizze_10487,
                   voxels_mem_10488, sizze_9797, sizze_9798, sizze_9799,
                   stepsizze_9803):
    res_9804 = sitofp_i32_f32(sizze_9799)
    res_9805 = futhark_sqrt32(res_9804)
    res_9806 = fptosi_f32_i32(res_9805)
    res_9807 = sitofp_i32_f32(res_9806)
    res_9808 = (res_9807 / np.float32(2.0))
    group_sizze_10139 = self.sizes["group_size_10138"]
    y_10140 = (group_sizze_10139 - np.int32(1))
    x_10141 = (sizze_9797 + y_10140)
    num_groups_10142 = squot32(x_10141, group_sizze_10139)
    num_threads_10143 = (group_sizze_10139 * num_groups_10142)
    binop_x_10490 = sext_i32_i64(sizze_9797)
    bytes_10489 = (np.int64(4) * binop_x_10490)
    mem_10491 = opencl_alloc(self, bytes_10489, "mem_10491")
    mem_10494 = opencl_alloc(self, bytes_10489, "mem_10494")
    if ((1 * (num_groups_10142 * group_sizze_10139)) != 0):
      self.map_kernel_10144_var.set_args(np.int32(sizze_9797), angles_mem_10484,
                                         mem_10491, mem_10494)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10144_var,
                                 (np.long((num_groups_10142 * group_sizze_10139)),),
                                 (np.long(group_sizze_10139),))
      if synchronous:
        self.queue.finish()
    nesting_sizze_10125 = (sizze_9797 * sizze_9798)
    group_sizze_10127 = self.sizes["group_size_10126"]
    y_10128 = (group_sizze_10127 - np.int32(1))
    x_10129 = (nesting_sizze_10125 + y_10128)
    num_groups_10130 = squot32(x_10129, group_sizze_10127)
    num_threads_10131 = (group_sizze_10127 * num_groups_10130)
    binop_x_10497 = sext_i32_i64(nesting_sizze_10125)
    bytes_10495 = (np.int64(4) * binop_x_10497)
    mem_10498 = opencl_alloc(self, bytes_10495, "mem_10498")
    if ((1 * (num_groups_10130 * group_sizze_10127)) != 0):
      self.map_kernel_10132_var.set_args(np.int32(sizze_9797),
                                         np.int32(sizze_9798), mem_10491,
                                         mem_10498)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10132_var,
                                 (np.long((num_groups_10130 * group_sizze_10127)),),
                                 (np.long(group_sizze_10127),))
      if synchronous:
        self.queue.finish()
    mem_10491 = None
    group_sizze_10111 = self.sizes["group_size_10110"]
    y_10112 = (group_sizze_10111 - np.int32(1))
    x_10113 = (y_10112 + nesting_sizze_10125)
    num_groups_10114 = squot32(x_10113, group_sizze_10111)
    num_threads_10115 = (group_sizze_10111 * num_groups_10114)
    mem_10502 = opencl_alloc(self, bytes_10495, "mem_10502")
    if ((1 * (num_groups_10114 * group_sizze_10111)) != 0):
      self.map_kernel_10116_var.set_args(np.int32(sizze_9797),
                                         np.int32(sizze_9798), mem_10494,
                                         mem_10502)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10116_var,
                                 (np.long((num_groups_10114 * group_sizze_10111)),),
                                 (np.long(group_sizze_10111),))
      if synchronous:
        self.queue.finish()
    mem_10494 = None
    res_9821 = (np.float32(0.0) - res_9808)
    group_sizze_10152 = self.sizes["group_size_10151"]
    y_10153 = (group_sizze_10152 - np.int32(1))
    x_10154 = (nesting_sizze_10125 + y_10153)
    num_groups_10155 = squot32(x_10154, group_sizze_10152)
    num_threads_10156 = (group_sizze_10152 * num_groups_10155)
    mem_10505 = opencl_alloc(self, bytes_10495, "mem_10505")
    mem_10508 = opencl_alloc(self, bytes_10495, "mem_10508")
    if ((1 * (num_groups_10155 * group_sizze_10152)) != 0):
      self.map_kernel_10157_var.set_args(np.int32(sizze_9798),
                                         np.float32(res_9808),
                                         np.float32(res_9821),
                                         np.int32(nesting_sizze_10125),
                                         rays_mem_10486, mem_10498, mem_10502,
                                         mem_10505, mem_10508)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10157_var,
                                 (np.long((num_groups_10155 * group_sizze_10152)),),
                                 (np.long(group_sizze_10152),))
      if synchronous:
        self.queue.finish()
    res_9850 = sdiv32(nesting_sizze_10125, stepsizze_9803)
    mem_10511 = opencl_alloc(self, np.int64(4), "mem_10511")
    group_sizze_10570 = self.sizes["group_size_10570"]
    num_groups_10571 = squot32(((np.int32(1) + sext_i32_i32(group_sizze_10570)) - np.int32(1)),
                               sext_i32_i32(group_sizze_10570))
    if ((1 * (num_groups_10571 * group_sizze_10570)) != 0):
      self.kernel_replicate_9851_var.set_args(mem_10511)
      cl.enqueue_nd_range_kernel(self.queue, self.kernel_replicate_9851_var,
                                 (np.long((num_groups_10571 * group_sizze_10570)),),
                                 (np.long(group_sizze_10570),))
      if synchronous:
        self.queue.finish()
    loop_cond_9852 = slt32(np.int32(0), res_9850)
    x_9853 = (np.float32(2.0) * res_9808)
    x_9854 = (np.float32(2.0) * x_9853)
    arg_9855 = (x_9854 - np.float32(1.0))
    res_9856 = fptosi_f32_i32(arg_9855)
    arg_9857 = (res_9856 - np.int32(1))
    group_sizze_10442 = self.sizes["group_size_10441"]
    y_10443 = (group_sizze_10442 - np.int32(1))
    group_sizze_10205 = self.sizes["group_size_10204"]
    y_10206 = (group_sizze_10205 - np.int32(1))
    binop_x_10528 = sext_i32_i64(res_9856)
    bytes_10527 = (np.int64(4) * binop_x_10528)
    sizze_9866 = np.int32(1)
    output_mem_sizze_10512 = np.int64(4)
    output_mem_10513 = mem_10511
    loop_while_9867 = loop_cond_9852
    run_9869 = np.int32(0)
    while loop_while_9867:
      x_9870 = (np.int32(1) + run_9869)
      x_9871 = (stepsizze_9803 * x_9870)
      cond_9872 = sle32(nesting_sizze_10125, x_9871)
      if cond_9872:
        y_9874 = (stepsizze_9803 * run_9869)
        res_9875 = (nesting_sizze_10125 - y_9874)
        res_9873 = res_9875
      else:
        res_9873 = stepsizze_9803
      bounds_invalid_upwards_9876 = slt32(res_9873, np.int32(0))
      eq_x_zz_9879 = (np.int32(0) == res_9873)
      not_p_9880 = not(bounds_invalid_upwards_9876)
      p_and_eq_x_y_9881 = (eq_x_zz_9879 and not_p_9880)
      dim_zzero_9882 = (bounds_invalid_upwards_9876 or p_and_eq_x_y_9881)
      both_empty_9883 = (eq_x_zz_9879 and dim_zzero_9882)
      empty_or_match_9884 = (not_p_9880 or both_empty_9883)
      empty_or_match_cert_9885 = True
      assert empty_or_match_9884, ("Error at forwardprojection_map.fut:6:1-10:59 -> forwardprojection_map.fut:10:11-10:59 -> projection_lib.fut:205:189-205:197 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                                                  "*",
                                                                                                                                                                                                                  "[",
                                                                                                                                                                                                                  res_9873,
                                                                                                                                                                                                                  "]",
                                                                                                                                                                                                                  "intrinsics.i32"))
      x_9886 = (stepsizze_9803 * run_9869)
      x_10444 = (res_9873 + y_10443)
      num_groups_10445 = squot32(x_10444, group_sizze_10442)
      num_threads_10446 = (group_sizze_10442 * num_groups_10445)
      binop_x_10515 = sext_i32_i64(res_9873)
      bytes_10514 = (np.int64(4) * binop_x_10515)
      mem_10516 = opencl_alloc(self, bytes_10514, "mem_10516")
      mem_10519 = opencl_alloc(self, bytes_10514, "mem_10519")
      mem_10521 = opencl_alloc(self, binop_x_10515, "mem_10521")
      mem_10524 = opencl_alloc(self, bytes_10514, "mem_10524")
      mem_10526 = opencl_alloc(self, binop_x_10515, "mem_10526")
      if ((1 * (num_groups_10445 * group_sizze_10442)) != 0):
        self.map_kernel_10447_var.set_args(np.int32(sizze_9798),
                                           np.int32(res_9873), np.int32(x_9886),
                                           mem_10498, mem_10502, mem_10505,
                                           mem_10508, mem_10516, mem_10519,
                                           mem_10521, mem_10524, mem_10526)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10447_var,
                                   (np.long((num_groups_10445 * group_sizze_10442)),),
                                   (np.long(group_sizze_10442),))
        if synchronous:
          self.queue.finish()
      x_10207 = (res_9873 + y_10206)
      num_groups_10208 = squot32(x_10207, group_sizze_10205)
      num_threads_10209 = (group_sizze_10205 * num_groups_10208)
      mem_10535 = opencl_alloc(self, bytes_10514, "mem_10535")
      num_threads64_10550 = sext_i32_i64(num_threads_10209)
      total_sizze_10551 = (bytes_10527 * num_threads64_10550)
      mem_10529 = opencl_alloc(self, total_sizze_10551, "mem_10529")
      total_sizze_10552 = (bytes_10527 * num_threads64_10550)
      mem_10532 = opencl_alloc(self, total_sizze_10552, "mem_10532")
      if ((1 * (num_groups_10208 * group_sizze_10205)) != 0):
        self.map_kernel_10210_var.set_args(np.float32(res_9808),
                                           np.float32(res_9821),
                                           np.int32(res_9856),
                                           np.int32(arg_9857),
                                           np.int32(res_9873), voxels_mem_10488,
                                           mem_10516, mem_10519, mem_10521,
                                           mem_10524, mem_10526, mem_10529,
                                           mem_10532, mem_10535)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10210_var,
                                   (np.long((num_groups_10208 * group_sizze_10205)),),
                                   (np.long(group_sizze_10205),))
        if synchronous:
          self.queue.finish()
      mem_10516 = None
      mem_10519 = None
      mem_10521 = None
      mem_10524 = None
      mem_10526 = None
      mem_10529 = None
      mem_10532 = None
      conc_tmp_10089 = (sizze_9866 + res_9873)
      binop_x_10537 = sext_i32_i64(conc_tmp_10089)
      bytes_10536 = (np.int64(4) * binop_x_10537)
      mem_10538 = opencl_alloc(self, bytes_10536, "mem_10538")
      tmp_offs_10593 = np.int32(0)
      if ((sizze_9866 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10538, output_mem_10513,
                        dest_offset=np.long((tmp_offs_10593 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sizze_9866 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10593 = (tmp_offs_10593 + sizze_9866)
      if ((res_9873 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_10538, mem_10535,
                        dest_offset=np.long((tmp_offs_10593 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((res_9873 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_10593 = (tmp_offs_10593 + res_9873)
      mem_10535 = None
      loop_cond_10091 = slt32(x_9870, res_9850)
      sizze_tmp_10572 = conc_tmp_10089
      output_mem_sizze_tmp_10573 = bytes_10536
      output_mem_tmp_10574 = mem_10538
      loop_while_tmp_10575 = loop_cond_10091
      run_tmp_10577 = x_9870
      sizze_9866 = sizze_tmp_10572
      output_mem_sizze_10512 = output_mem_sizze_tmp_10573
      output_mem_10513 = output_mem_tmp_10574
      loop_while_9867 = loop_while_tmp_10575
      run_9869 = run_tmp_10577
    sizze_9862 = sizze_9866
    res_mem_sizze_10539 = output_mem_sizze_10512
    res_mem_10540 = output_mem_10513
    res_9863 = loop_while_9867
    res_9865 = run_9869
    mem_10498 = None
    mem_10502 = None
    mem_10505 = None
    mem_10508 = None
    mem_10511 = None
    j_m_i_10092 = (sizze_9862 - np.int32(1))
    x_10093 = abs(j_m_i_10092)
    empty_slice_10094 = (x_10093 == np.int32(0))
    m_10095 = (x_10093 - np.int32(1))
    i_p_m_t_s_10096 = (np.int32(1) + m_10095)
    zzero_leq_i_p_m_t_s_10097 = sle32(np.int32(0), i_p_m_t_s_10096)
    i_p_m_t_s_leq_w_10098 = slt32(i_p_m_t_s_10096, sizze_9862)
    i_lte_j_10099 = sle32(np.int32(1), sizze_9862)
    y_10100 = (zzero_leq_i_p_m_t_s_10097 and i_p_m_t_s_leq_w_10098)
    y_10101 = (i_lte_j_10099 and y_10100)
    ok_or_empty_10102 = (empty_slice_10094 or y_10101)
    index_certs_10103 = True
    assert ok_or_empty_10102, ("Error at forwardprojection_map.fut:6:1-10:59 -> forwardprojection_map.fut:10:11-10:59 -> projection_lib.fut:208:20-208:31 -> /futlib/array.fut:21:29-21:33: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                           np.int32(1),
                                                                                                                                                                                                           "] out of bounds for array of shape [",
                                                                                                                                                                                                           sizze_9862,
                                                                                                                                                                                                           "]."))
    binop_x_10542 = sext_i32_i64(x_10093)
    bytes_10541 = (np.int64(4) * binop_x_10542)
    mem_10543 = opencl_alloc(self, bytes_10541, "mem_10543")
    if ((x_10093 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_10543, res_mem_10540,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(4)),
                      byte_count=np.long((x_10093 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    res_mem_10540 = None
    out_arrsizze_10555 = x_10093
    out_memsizze_10554 = bytes_10541
    out_mem_10553 = mem_10543
    return (out_memsizze_10554, out_mem_10553, out_arrsizze_10555)
  def main(self, angles_mem_10484_ext, rays_mem_10486_ext, voxels_mem_10488_ext,
           stepsizze_9803_ext):
    try:
      assert ((type(angles_mem_10484_ext) in [np.ndarray,
                                              cl.array.Array]) and (angles_mem_10484_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9797 = np.int32(angles_mem_10484_ext.shape[0])
      angles_mem_sizze_10483 = np.int64(angles_mem_10484_ext.nbytes)
      if (type(angles_mem_10484_ext) == cl.array.Array):
        angles_mem_10484 = angles_mem_10484_ext.data
      else:
        angles_mem_10484 = opencl_alloc(self, angles_mem_sizze_10483,
                                        "angles_mem_10484")
        if (angles_mem_sizze_10483 != 0):
          cl.enqueue_copy(self.queue, angles_mem_10484,
                          normaliseArray(angles_mem_10484_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(angles_mem_10484_ext),
                                                                                                                            angles_mem_10484_ext))
    try:
      assert ((type(rays_mem_10486_ext) in [np.ndarray,
                                            cl.array.Array]) and (rays_mem_10486_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9798 = np.int32(rays_mem_10486_ext.shape[0])
      rays_mem_sizze_10485 = np.int64(rays_mem_10486_ext.nbytes)
      if (type(rays_mem_10486_ext) == cl.array.Array):
        rays_mem_10486 = rays_mem_10486_ext.data
      else:
        rays_mem_10486 = opencl_alloc(self, rays_mem_sizze_10485,
                                      "rays_mem_10486")
        if (rays_mem_sizze_10485 != 0):
          cl.enqueue_copy(self.queue, rays_mem_10486,
                          normaliseArray(rays_mem_10486_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(rays_mem_10486_ext),
                                                                                                                            rays_mem_10486_ext))
    try:
      assert ((type(voxels_mem_10488_ext) in [np.ndarray,
                                              cl.array.Array]) and (voxels_mem_10488_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9799 = np.int32(voxels_mem_10488_ext.shape[0])
      voxels_mem_sizze_10487 = np.int64(voxels_mem_10488_ext.nbytes)
      if (type(voxels_mem_10488_ext) == cl.array.Array):
        voxels_mem_10488 = voxels_mem_10488_ext.data
      else:
        voxels_mem_10488 = opencl_alloc(self, voxels_mem_sizze_10487,
                                        "voxels_mem_10488")
        if (voxels_mem_sizze_10487 != 0):
          cl.enqueue_copy(self.queue, voxels_mem_10488,
                          normaliseArray(voxels_mem_10488_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(voxels_mem_10488_ext),
                                                                                                                            voxels_mem_10488_ext))
    try:
      stepsizze_9803 = np.int32(ct.c_int32(stepsizze_9803_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(stepsizze_9803_ext),
                                                                                                                            stepsizze_9803_ext))
    (out_memsizze_10554, out_mem_10553,
     out_arrsizze_10555) = self.futhark_main(angles_mem_sizze_10483,
                                             angles_mem_10484,
                                             rays_mem_sizze_10485,
                                             rays_mem_10486,
                                             voxels_mem_sizze_10487,
                                             voxels_mem_10488, sizze_9797,
                                             sizze_9798, sizze_9799,
                                             stepsizze_9803)
    return cl.array.Array(self.queue, (out_arrsizze_10555,), ct.c_float,
                          data=out_mem_10553)