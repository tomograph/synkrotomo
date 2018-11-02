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
#define group_sizze_12568 (group_size_12567)
#define group_sizze_12839 (group_size_12838)
#define y_12842 (group_size_12838 - 1)
#define group_sizze_12939 (group_size_12938)
#define y_12942 (group_size_12938 - 1)
#define group_sizze_13068 (group_size_13067)
#define y_13071 (group_size_13067 - 1)
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
__kernel void kernel_replicate_11891(__global unsigned char *mem_13314)
{
    const uint replicate_gtid_11891 = get_global_id(0);
    
    if (replicate_gtid_11891 >= 1)
        return;
    *(__global float *) &mem_13314[replicate_gtid_11891 * 4] = 0.0F;
}
__kernel void kernel_replicate_12419(int32_t res_12409, __global
                                     unsigned char *mem_13423)
{
    const uint replicate_gtid_12419 = get_global_id(0);
    
    if (replicate_gtid_12419 >= res_12409)
        return;
    *(__global int32_t *) &mem_13423[replicate_gtid_12419 * 4] = 0;
}
__kernel void map_kernel_12479(int32_t sizze_11837, int32_t sizze_11838,
                               __global unsigned char *mem_13297, __global
                               unsigned char *mem_13305)
{
    int32_t wave_sizze_13507;
    int32_t group_sizze_13508;
    bool thread_active_13509;
    int32_t gtid_12470;
    int32_t gtid_12471;
    int32_t global_tid_12479;
    int32_t local_tid_12480;
    int32_t group_id_12481;
    
    global_tid_12479 = get_global_id(0);
    local_tid_12480 = get_local_id(0);
    group_sizze_13508 = get_local_size(0);
    wave_sizze_13507 = LOCKSTEP_WIDTH;
    group_id_12481 = get_group_id(0);
    gtid_12470 = squot32(global_tid_12479, sizze_11838);
    gtid_12471 = global_tid_12479 - squot32(global_tid_12479, sizze_11838) *
        sizze_11838;
    thread_active_13509 = slt32(gtid_12470, sizze_11837) && slt32(gtid_12471,
                                                                  sizze_11838);
    
    float res_12482;
    
    if (thread_active_13509) {
        res_12482 = *(__global float *) &mem_13297[gtid_12470 * 4];
    }
    if (thread_active_13509) {
        *(__global float *) &mem_13305[(gtid_12470 * sizze_11838 + gtid_12471) *
                                       4] = res_12482;
    }
}
__kernel void map_kernel_12495(int32_t sizze_11837, int32_t sizze_11838,
                               __global unsigned char *mem_13294, __global
                               unsigned char *mem_13301)
{
    int32_t wave_sizze_13504;
    int32_t group_sizze_13505;
    bool thread_active_13506;
    int32_t gtid_12486;
    int32_t gtid_12487;
    int32_t global_tid_12495;
    int32_t local_tid_12496;
    int32_t group_id_12497;
    
    global_tid_12495 = get_global_id(0);
    local_tid_12496 = get_local_id(0);
    group_sizze_13505 = get_local_size(0);
    wave_sizze_13504 = LOCKSTEP_WIDTH;
    group_id_12497 = get_group_id(0);
    gtid_12486 = squot32(global_tid_12495, sizze_11838);
    gtid_12487 = global_tid_12495 - squot32(global_tid_12495, sizze_11838) *
        sizze_11838;
    thread_active_13506 = slt32(gtid_12486, sizze_11837) && slt32(gtid_12487,
                                                                  sizze_11838);
    
    float res_12498;
    
    if (thread_active_13506) {
        res_12498 = *(__global float *) &mem_13294[gtid_12486 * 4];
    }
    if (thread_active_13506) {
        *(__global float *) &mem_13301[(gtid_12486 * sizze_11838 + gtid_12487) *
                                       4] = res_12498;
    }
}
__kernel void map_kernel_12507(int32_t sizze_11837, __global
                               unsigned char *angles_mem_13287, __global
                               unsigned char *mem_13294, __global
                               unsigned char *mem_13297)
{
    int32_t wave_sizze_13501;
    int32_t group_sizze_13502;
    bool thread_active_13503;
    int32_t gtid_12500;
    int32_t global_tid_12507;
    int32_t local_tid_12508;
    int32_t group_id_12509;
    
    global_tid_12507 = get_global_id(0);
    local_tid_12508 = get_local_id(0);
    group_sizze_13502 = get_local_size(0);
    wave_sizze_13501 = LOCKSTEP_WIDTH;
    group_id_12509 = get_group_id(0);
    gtid_12500 = global_tid_12507;
    thread_active_13503 = slt32(gtid_12500, sizze_11837);
    
    float x_12510;
    float res_12511;
    float res_12512;
    
    if (thread_active_13503) {
        x_12510 = *(__global float *) &angles_mem_13287[gtid_12500 * 4];
        res_12511 = futrts_sin32(x_12510);
        res_12512 = futrts_cos32(x_12510);
    }
    if (thread_active_13503) {
        *(__global float *) &mem_13294[gtid_12500 * 4] = res_12511;
    }
    if (thread_active_13503) {
        *(__global float *) &mem_13297[gtid_12500 * 4] = res_12512;
    }
}
__kernel void map_kernel_12520(int32_t sizze_11838, float res_11848,
                               float res_11861, int32_t nesting_sizze_12488,
                               __global unsigned char *rays_mem_13289, __global
                               unsigned char *mem_13301, __global
                               unsigned char *mem_13305, __global
                               unsigned char *mem_13308, __global
                               unsigned char *mem_13311)
{
    int32_t wave_sizze_13510;
    int32_t group_sizze_13511;
    bool thread_active_13512;
    int32_t gtid_12513;
    int32_t global_tid_12520;
    int32_t local_tid_12521;
    int32_t group_id_12522;
    
    global_tid_12520 = get_global_id(0);
    local_tid_12521 = get_local_id(0);
    group_sizze_13511 = get_local_size(0);
    wave_sizze_13510 = LOCKSTEP_WIDTH;
    group_id_12522 = get_group_id(0);
    gtid_12513 = global_tid_12520;
    thread_active_13512 = slt32(gtid_12513, nesting_sizze_12488);
    
    int32_t new_index_13230;
    int32_t binop_y_13232;
    int32_t new_index_13233;
    float x_12523;
    float x_12524;
    float x_12525;
    bool cond_12526;
    float res_12527;
    bool cond_12531;
    float res_12532;
    float res_12536;
    float res_12540;
    bool cond_12541;
    bool res_12542;
    bool x_12543;
    float res_12544;
    float res_12545;
    
    if (thread_active_13512) {
        new_index_13230 = squot32(gtid_12513, sizze_11838);
        binop_y_13232 = sizze_11838 * new_index_13230;
        new_index_13233 = gtid_12513 - binop_y_13232;
        x_12523 = *(__global float *) &mem_13301[(new_index_13230 *
                                                  sizze_11838 +
                                                  new_index_13233) * 4];
        x_12524 = *(__global float *) &mem_13305[(new_index_13230 *
                                                  sizze_11838 +
                                                  new_index_13233) * 4];
        x_12525 = *(__global float *) &rays_mem_13289[new_index_13233 * 4];
        cond_12526 = x_12523 == 0.0F;
        if (cond_12526) {
            res_12527 = x_12525;
        } else {
            float y_12528;
            float x_12529;
            float res_12530;
            
            y_12528 = res_11861 * x_12524;
            x_12529 = x_12525 - y_12528;
            res_12530 = x_12529 / x_12523;
            res_12527 = res_12530;
        }
        cond_12531 = x_12524 == 0.0F;
        if (cond_12531) {
            res_12532 = x_12525;
        } else {
            float y_12533;
            float x_12534;
            float res_12535;
            
            y_12533 = res_11861 * x_12523;
            x_12534 = x_12525 - y_12533;
            res_12535 = x_12534 / x_12524;
            res_12532 = res_12535;
        }
        if (cond_12531) {
            res_12536 = x_12525;
        } else {
            float y_12537;
            float x_12538;
            float res_12539;
            
            y_12537 = res_11848 * x_12523;
            x_12538 = x_12525 - y_12537;
            res_12539 = x_12538 / x_12524;
            res_12536 = res_12539;
        }
        res_12540 = (float) fabs(res_12527);
        cond_12541 = res_12540 <= res_11848;
        res_12542 = !cond_12526;
        x_12543 = cond_12541 && res_12542;
        if (x_12543) {
            res_12544 = res_11861;
            res_12545 = res_12527;
        } else {
            bool cond_12546;
            float res_12547;
            float res_12548;
            
            cond_12546 = res_12532 <= res_12536;
            if (cond_12546) {
                res_12547 = res_12532;
            } else {
                res_12547 = res_12536;
            }
            if (cond_12546) {
                res_12548 = res_11861;
            } else {
                res_12548 = res_11848;
            }
            res_12544 = res_12547;
            res_12545 = res_12548;
        }
    }
    if (thread_active_13512) {
        *(__global float *) &mem_13308[gtid_12513 * 4] = res_12544;
    }
    if (thread_active_13512) {
        *(__global float *) &mem_13311[gtid_12513 * 4] = res_12545;
    }
}
__kernel void map_kernel_12573(float res_11847, float res_11848,
                               float res_11861, int32_t res_11895,
                               int32_t i_11919, int32_t x_11921, __global
                               unsigned char *mem_13308, __global
                               unsigned char *mem_13311, __global
                               unsigned char *mem_13318, __global
                               unsigned char *mem_13320, __global
                               unsigned char *mem_13323, __global
                               unsigned char *mem_13326, __global
                               unsigned char *mem_13329, __global
                               unsigned char *mem_13350, __global
                               unsigned char *mem_13354, __global
                               unsigned char *mem_13358)
{
    int32_t wave_sizze_13526;
    int32_t group_sizze_13527;
    bool thread_active_13528;
    int32_t gtid_12566;
    int32_t global_tid_12573;
    int32_t local_tid_12574;
    int32_t group_id_12575;
    
    global_tid_12573 = get_global_id(0);
    local_tid_12574 = get_local_id(0);
    group_sizze_13527 = get_local_size(0);
    wave_sizze_13526 = LOCKSTEP_WIDTH;
    group_id_12575 = get_group_id(0);
    gtid_12566 = global_tid_12573;
    thread_active_13528 = slt32(gtid_12566, x_11921);
    
    int32_t j_p_i_t_s_13253;
    float x_12576;
    float x_12577;
    bool res_12578;
    bool res_12579;
    float res_12580;
    bool cond_12583;
    float res_12584;
    int32_t res_12585;
    float res_12586;
    bool res_12587;
    float res_12588;
    float res_12595;
    float res_12596;
    bool cond_12619;
    bool res_12620;
    bool x_12621;
    bool cond_12622;
    bool res_12623;
    bool x_12624;
    bool cond_12625;
    bool res_12626;
    bool x_12627;
    bool x_12628;
    bool x_12629;
    bool y_12630;
    bool res_12631;
    bool x_12632;
    float y_12633;
    bool res_12634;
    float res_12637;
    float res_12638;
    float res_12639;
    float res_12640;
    int32_t res_12641;
    int32_t res_12778;
    
    if (thread_active_13528) {
        j_p_i_t_s_13253 = i_11919 + gtid_12566;
        x_12576 = *(__global float *) &mem_13308[j_p_i_t_s_13253 * 4];
        x_12577 = *(__global float *) &mem_13311[j_p_i_t_s_13253 * 4];
        res_12578 = *(__global bool *) &mem_13318[gtid_12566];
        res_12579 = *(__global bool *) &mem_13320[gtid_12566];
        res_12580 = *(__global float *) &mem_13323[gtid_12566 * 4];
        for (int32_t i_13529 = 0; i_13529 < res_11895; i_13529++) {
            *(__global float *) &mem_13326[(group_id_12575 * (res_11895 *
                                                              group_sizze_12568) +
                                            i_13529 * group_sizze_12568 +
                                            local_tid_12574) * 4] = -1.0F;
        }
        for (int32_t i_13530 = 0; i_13530 < res_11895; i_13530++) {
            *(__global int32_t *) &mem_13329[(group_id_12575 * (res_11895 *
                                                                group_sizze_12568) +
                                              i_13530 * group_sizze_12568 +
                                              local_tid_12574) * 4] = -1;
        }
        cond_12583 = res_12580 < 0.0F;
        if (cond_12583) {
            res_12584 = -1.0F;
        } else {
            res_12584 = 1.0F;
        }
        res_12585 = fptosi_f32_i32(x_12576);
        res_12586 = sitofp_i32_f32(res_12585);
        res_12587 = 0.0F <= x_12576;
        if (res_12587) {
            bool res_12589;
            float res_12590;
            
            res_12589 = res_12586 < x_12576;
            if (res_12589) {
                res_12590 = res_12586;
            } else {
                res_12590 = x_12576;
            }
            res_12588 = res_12590;
        } else {
            bool res_12591;
            float res_12592;
            
            res_12591 = x_12576 < res_12586;
            if (res_12591) {
                int32_t res_12593;
                float res_12594;
                
                res_12593 = res_12585 - 1;
                res_12594 = sitofp_i32_f32(res_12593);
                res_12592 = res_12594;
            } else {
                res_12592 = x_12576;
            }
            res_12588 = res_12592;
        }
        res_12595 = 1.0F + res_12588;
        if (cond_12583) {
            int32_t res_12597;
            float res_12598;
            bool res_12599;
            float res_12600;
            float res_12607;
            
            res_12597 = fptosi_f32_i32(x_12577);
            res_12598 = sitofp_i32_f32(res_12597);
            res_12599 = 0.0F <= x_12577;
            if (res_12599) {
                bool res_12601;
                float res_12602;
                
                res_12601 = res_12598 < x_12577;
                if (res_12601) {
                    int32_t res_12603;
                    float res_12604;
                    
                    res_12603 = 1 + res_12597;
                    res_12604 = sitofp_i32_f32(res_12603);
                    res_12602 = res_12604;
                } else {
                    res_12602 = x_12577;
                }
                res_12600 = res_12602;
            } else {
                bool res_12605;
                float res_12606;
                
                res_12605 = x_12577 < res_12598;
                if (res_12605) {
                    res_12606 = res_12598;
                } else {
                    res_12606 = x_12577;
                }
                res_12600 = res_12606;
            }
            res_12607 = res_12600 - 1.0F;
            res_12596 = res_12607;
        } else {
            int32_t res_12608;
            float res_12609;
            bool res_12610;
            float res_12611;
            float res_12618;
            
            res_12608 = fptosi_f32_i32(x_12577);
            res_12609 = sitofp_i32_f32(res_12608);
            res_12610 = 0.0F <= x_12577;
            if (res_12610) {
                bool res_12612;
                float res_12613;
                
                res_12612 = res_12609 < x_12577;
                if (res_12612) {
                    res_12613 = res_12609;
                } else {
                    res_12613 = x_12577;
                }
                res_12611 = res_12613;
            } else {
                bool res_12614;
                float res_12615;
                
                res_12614 = x_12577 < res_12609;
                if (res_12614) {
                    int32_t res_12616;
                    float res_12617;
                    
                    res_12616 = res_12608 - 1;
                    res_12617 = sitofp_i32_f32(res_12616);
                    res_12615 = res_12617;
                } else {
                    res_12615 = x_12577;
                }
                res_12611 = res_12615;
            }
            res_12618 = 1.0F + res_12611;
            res_12596 = res_12618;
        }
        cond_12619 = res_11861 <= x_12576;
        res_12620 = x_12576 < res_11848;
        x_12621 = cond_12619 && res_12620;
        cond_12622 = res_11861 < x_12577;
        res_12623 = x_12577 <= res_11848;
        x_12624 = cond_12622 && res_12623;
        cond_12625 = res_11861 <= x_12577;
        res_12626 = x_12577 < res_11848;
        x_12627 = cond_12625 && res_12626;
        x_12628 = cond_12583 && x_12624;
        x_12629 = !cond_12583;
        y_12630 = x_12627 && x_12629;
        res_12631 = x_12628 || y_12630;
        x_12632 = x_12621 && res_12631;
        y_12633 = 1.0F / res_12580;
        
        bool loop_while_12642;
        float focusPoint_12645;
        float focusPoint_12646;
        float anchorX_12647;
        float anchorY_12648;
        int32_t write_index_12649;
        
        loop_while_12642 = x_12632;
        focusPoint_12645 = x_12576;
        focusPoint_12646 = x_12577;
        anchorX_12647 = res_12595;
        anchorY_12648 = res_12596;
        write_index_12649 = 0;
        while (loop_while_12642) {
            float arg_12650 = res_11848 + focusPoint_12646;
            int32_t res_12651 = fptosi_f32_i32(arg_12650);
            float res_12652 = sitofp_i32_f32(res_12651);
            bool res_12653 = 0.0F <= arg_12650;
            float res_12654;
            
            if (res_12653) {
                bool res_12655;
                float res_12656;
                
                res_12655 = res_12652 < arg_12650;
                if (res_12655) {
                    res_12656 = res_12652;
                } else {
                    res_12656 = arg_12650;
                }
                res_12654 = res_12656;
            } else {
                bool res_12657;
                float res_12658;
                
                res_12657 = arg_12650 < res_12652;
                if (res_12657) {
                    int32_t res_12659;
                    float res_12660;
                    
                    res_12659 = res_12651 - 1;
                    res_12660 = sitofp_i32_f32(res_12659);
                    res_12658 = res_12660;
                } else {
                    res_12658 = arg_12650;
                }
                res_12654 = res_12658;
            }
            
            int32_t res_12661 = fptosi_f32_i32(focusPoint_12646);
            float res_12662 = sitofp_i32_f32(res_12661);
            bool res_12663 = 0.0F <= focusPoint_12646;
            float res_12664;
            
            if (res_12663) {
                bool res_12665;
                float res_12666;
                
                res_12665 = res_12662 < focusPoint_12646;
                if (res_12665) {
                    res_12666 = res_12662;
                } else {
                    res_12666 = focusPoint_12646;
                }
                res_12664 = res_12666;
            } else {
                bool res_12667;
                float res_12668;
                
                res_12667 = focusPoint_12646 < res_12662;
                if (res_12667) {
                    int32_t res_12669;
                    float res_12670;
                    
                    res_12669 = res_12661 - 1;
                    res_12670 = sitofp_i32_f32(res_12669);
                    res_12668 = res_12670;
                } else {
                    res_12668 = focusPoint_12646;
                }
                res_12664 = res_12668;
            }
            
            float x_12671 = focusPoint_12646 - res_12664;
            bool res_12672 = x_12671 == 0.0F;
            bool x_12673 = cond_12583 && res_12672;
            float res_12674;
            
            if (x_12673) {
                float res_12675 = res_12654 - 1.0F;
                
                res_12674 = res_12675;
            } else {
                res_12674 = res_12654;
            }
            
            float arg_12676 = res_11848 + focusPoint_12645;
            int32_t res_12677 = fptosi_f32_i32(arg_12676);
            float res_12678 = sitofp_i32_f32(res_12677);
            bool res_12679 = 0.0F <= arg_12676;
            float res_12680;
            
            if (res_12679) {
                bool res_12681;
                float res_12682;
                
                res_12681 = res_12678 < arg_12676;
                if (res_12681) {
                    res_12682 = res_12678;
                } else {
                    res_12682 = arg_12676;
                }
                res_12680 = res_12682;
            } else {
                bool res_12683;
                float res_12684;
                
                res_12683 = arg_12676 < res_12678;
                if (res_12683) {
                    int32_t res_12685;
                    float res_12686;
                    
                    res_12685 = res_12677 - 1;
                    res_12686 = sitofp_i32_f32(res_12685);
                    res_12684 = res_12686;
                } else {
                    res_12684 = arg_12676;
                }
                res_12680 = res_12684;
            }
            
            float y_12687 = res_11847 * res_12674;
            float arg_12688 = res_12680 + y_12687;
            int32_t res_12689 = fptosi_f32_i32(arg_12688);
            float res_12690;
            
            if (res_12579) {
                res_12690 = 1.0F;
            } else {
                float res_12691;
                
                if (res_12578) {
                    res_12691 = 0.0F;
                } else {
                    float x_12692;
                    float res_12693;
                    
                    x_12692 = anchorX_12647 - focusPoint_12645;
                    res_12693 = res_12580 * x_12692;
                    res_12691 = res_12693;
                }
                res_12690 = res_12691;
            }
            
            float res_12694;
            
            if (res_12579) {
                res_12694 = 0.0F;
            } else {
                float res_12695;
                
                if (res_12578) {
                    res_12695 = 1.0F;
                } else {
                    float x_12696;
                    float res_12697;
                    
                    x_12696 = anchorY_12648 - focusPoint_12646;
                    res_12697 = y_12633 * x_12696;
                    res_12695 = res_12697;
                }
                res_12694 = res_12695;
            }
            
            float res_12698 = focusPoint_12646 + res_12690;
            float res_12699 = focusPoint_12645 + res_12694;
            float x_12700 = anchorX_12647 - focusPoint_12645;
            float x_12701 = fpow32(x_12700, 2.0F);
            float x_12702 = res_12698 - focusPoint_12646;
            float y_12703 = fpow32(x_12702, 2.0F);
            float arg_12704 = x_12701 + y_12703;
            float res_12705;
            
            res_12705 = futrts_sqrt32(arg_12704);
            
            float x_12706 = res_12699 - focusPoint_12645;
            float x_12707 = fpow32(x_12706, 2.0F);
            float x_12708 = anchorY_12648 - focusPoint_12646;
            float y_12709 = fpow32(x_12708, 2.0F);
            float arg_12710 = x_12707 + y_12709;
            float res_12711;
            
            res_12711 = futrts_sqrt32(arg_12710);
            
            float res_12714;
            float res_12715;
            float res_12716;
            float res_12717;
            int32_t res_12718;
            
            if (res_12578) {
                float res_12721;
                int32_t res_12722;
                
                *(__global float *) &mem_13326[(group_id_12575 * (res_11895 *
                                                                  group_sizze_12568) +
                                                write_index_12649 *
                                                group_sizze_12568 +
                                                local_tid_12574) * 4] =
                    res_12705;
                *(__global int32_t *) &mem_13329[(group_id_12575 * (res_11895 *
                                                                    group_sizze_12568) +
                                                  write_index_12649 *
                                                  group_sizze_12568 +
                                                  local_tid_12574) * 4] =
                    res_12689;
                res_12721 = 1.0F + anchorX_12647;
                res_12722 = 1 + write_index_12649;
                res_12714 = anchorX_12647;
                res_12715 = res_12698;
                res_12716 = res_12721;
                res_12717 = anchorY_12648;
                res_12718 = res_12722;
            } else {
                float res_12725;
                float res_12726;
                float res_12727;
                float res_12728;
                int32_t res_12729;
                
                if (res_12579) {
                    float res_12732;
                    int32_t res_12733;
                    
                    *(__global float *) &mem_13326[(group_id_12575 *
                                                    (res_11895 *
                                                     group_sizze_12568) +
                                                    write_index_12649 *
                                                    group_sizze_12568 +
                                                    local_tid_12574) * 4] =
                        res_12711;
                    *(__global int32_t *) &mem_13329[(group_id_12575 *
                                                      (res_11895 *
                                                       group_sizze_12568) +
                                                      write_index_12649 *
                                                      group_sizze_12568 +
                                                      local_tid_12574) * 4] =
                        res_12689;
                    res_12732 = res_12584 + anchorY_12648;
                    res_12733 = 1 + write_index_12649;
                    res_12725 = res_12699;
                    res_12726 = anchorY_12648;
                    res_12727 = anchorX_12647;
                    res_12728 = res_12732;
                    res_12729 = res_12733;
                } else {
                    float arg_12734;
                    float res_12735;
                    bool cond_12736;
                    float res_12739;
                    float res_12740;
                    float res_12741;
                    float res_12742;
                    int32_t res_12743;
                    
                    arg_12734 = res_12705 - res_12711;
                    res_12735 = (float) fabs(arg_12734);
                    cond_12736 = 1.0e-9F < res_12735;
                    if (cond_12736) {
                        bool cond_12744;
                        float res_12745;
                        float res_12746;
                        float res_12749;
                        float res_12750;
                        int32_t res_12751;
                        
                        cond_12744 = res_12705 < res_12711;
                        if (cond_12744) {
                            res_12745 = anchorX_12647;
                        } else {
                            res_12745 = res_12699;
                        }
                        if (cond_12744) {
                            res_12746 = res_12698;
                        } else {
                            res_12746 = anchorY_12648;
                        }
                        if (cond_12744) {
                            float res_12754;
                            int32_t res_12755;
                            
                            *(__global float *) &mem_13326[(group_id_12575 *
                                                            (res_11895 *
                                                             group_sizze_12568) +
                                                            write_index_12649 *
                                                            group_sizze_12568 +
                                                            local_tid_12574) *
                                                           4] = res_12705;
                            *(__global int32_t *) &mem_13329[(group_id_12575 *
                                                              (res_11895 *
                                                               group_sizze_12568) +
                                                              write_index_12649 *
                                                              group_sizze_12568 +
                                                              local_tid_12574) *
                                                             4] = res_12689;
                            res_12754 = 1.0F + anchorX_12647;
                            res_12755 = 1 + write_index_12649;
                            res_12749 = res_12754;
                            res_12750 = anchorY_12648;
                            res_12751 = res_12755;
                        } else {
                            float res_12758;
                            int32_t res_12759;
                            
                            *(__global float *) &mem_13326[(group_id_12575 *
                                                            (res_11895 *
                                                             group_sizze_12568) +
                                                            write_index_12649 *
                                                            group_sizze_12568 +
                                                            local_tid_12574) *
                                                           4] = res_12711;
                            *(__global int32_t *) &mem_13329[(group_id_12575 *
                                                              (res_11895 *
                                                               group_sizze_12568) +
                                                              write_index_12649 *
                                                              group_sizze_12568 +
                                                              local_tid_12574) *
                                                             4] = res_12689;
                            res_12758 = res_12584 + anchorY_12648;
                            res_12759 = 1 + write_index_12649;
                            res_12749 = anchorX_12647;
                            res_12750 = res_12758;
                            res_12751 = res_12759;
                        }
                        res_12739 = res_12745;
                        res_12740 = res_12746;
                        res_12741 = res_12749;
                        res_12742 = res_12750;
                        res_12743 = res_12751;
                    } else {
                        float res_12762;
                        float res_12763;
                        int32_t res_12764;
                        
                        *(__global float *) &mem_13326[(group_id_12575 *
                                                        (res_11895 *
                                                         group_sizze_12568) +
                                                        write_index_12649 *
                                                        group_sizze_12568 +
                                                        local_tid_12574) * 4] =
                            res_12705;
                        *(__global int32_t *) &mem_13329[(group_id_12575 *
                                                          (res_11895 *
                                                           group_sizze_12568) +
                                                          write_index_12649 *
                                                          group_sizze_12568 +
                                                          local_tid_12574) *
                                                         4] = res_12689;
                        res_12762 = 1.0F + anchorX_12647;
                        res_12763 = res_12584 + anchorY_12648;
                        res_12764 = 1 + write_index_12649;
                        res_12739 = anchorX_12647;
                        res_12740 = res_12698;
                        res_12741 = res_12762;
                        res_12742 = res_12763;
                        res_12743 = res_12764;
                    }
                    res_12725 = res_12739;
                    res_12726 = res_12740;
                    res_12727 = res_12741;
                    res_12728 = res_12742;
                    res_12729 = res_12743;
                }
                res_12714 = res_12725;
                res_12715 = res_12726;
                res_12716 = res_12727;
                res_12717 = res_12728;
                res_12718 = res_12729;
            }
            
            bool cond_12765 = res_11861 <= res_12714;
            bool res_12766 = res_12714 < res_11848;
            bool x_12767 = cond_12765 && res_12766;
            bool cond_12768 = res_11861 < res_12715;
            bool res_12769 = res_12715 <= res_11848;
            bool x_12770 = cond_12768 && res_12769;
            bool cond_12771 = res_11861 <= res_12715;
            bool res_12772 = res_12715 < res_11848;
            bool x_12773 = cond_12771 && res_12772;
            bool x_12774 = cond_12583 && x_12770;
            bool y_12775 = x_12629 && x_12773;
            bool res_12776 = x_12774 || y_12775;
            bool x_12777 = x_12767 && res_12776;
            bool loop_while_tmp_13531 = x_12777;
            float focusPoint_tmp_13534 = res_12714;
            float focusPoint_tmp_13535 = res_12715;
            float anchorX_tmp_13536 = res_12716;
            float anchorY_tmp_13537 = res_12717;
            int32_t write_index_tmp_13538;
            
            write_index_tmp_13538 = res_12718;
            loop_while_12642 = loop_while_tmp_13531;
            focusPoint_12645 = focusPoint_tmp_13534;
            focusPoint_12646 = focusPoint_tmp_13535;
            anchorX_12647 = anchorX_tmp_13536;
            anchorY_12648 = anchorY_tmp_13537;
            write_index_12649 = write_index_tmp_13538;
        }
        res_12634 = loop_while_12642;
        res_12637 = focusPoint_12645;
        res_12638 = focusPoint_12646;
        res_12639 = anchorX_12647;
        res_12640 = anchorY_12648;
        res_12641 = write_index_12649;
        
        int32_t x_12781 = 0;
        
        for (int32_t chunk_offset_12780 = 0; chunk_offset_12780 < res_11895;
             chunk_offset_12780++) {
            int32_t x_12788 = *(__global int32_t *) &mem_13329[(group_id_12575 *
                                                                (res_11895 *
                                                                 group_sizze_12568) +
                                                                chunk_offset_12780 *
                                                                group_sizze_12568 +
                                                                local_tid_12574) *
                                                               4];
            bool cond_12790 = x_12788 == -1;
            int32_t res_12791;
            
            if (cond_12790) {
                res_12791 = 0;
            } else {
                res_12791 = 1;
            }
            
            int32_t res_12793 = x_12781 + res_12791;
            int32_t x_tmp_13539 = res_12793;
            
            x_12781 = x_tmp_13539;
        }
        res_12778 = x_12781;
    }
    if (thread_active_13528) {
        *(__global int32_t *) &mem_13350[gtid_12566 * 4] = res_12778;
    }
    if (thread_active_13528) {
        for (int32_t i_13540 = 0; i_13540 < res_11895; i_13540++) {
            *(__global float *) &mem_13354[(i_13540 * x_11921 + gtid_12566) *
                                           4] = *(__global
                                                  float *) &mem_13326[(group_id_12575 *
                                                                       (res_11895 *
                                                                        group_sizze_12568) +
                                                                       i_13540 *
                                                                       group_sizze_12568 +
                                                                       local_tid_12574) *
                                                                      4];
        }
    }
    if (thread_active_13528) {
        for (int32_t i_13541 = 0; i_13541 < res_11895; i_13541++) {
            *(__global int32_t *) &mem_13358[(i_13541 * x_11921 + gtid_12566) *
                                             4] = *(__global
                                                    int32_t *) &mem_13329[(group_id_12575 *
                                                                           (res_11895 *
                                                                            group_sizze_12568) +
                                                                           i_13541 *
                                                                           group_sizze_12568 +
                                                                           local_tid_12574) *
                                                                          4];
        }
    }
}
__kernel void map_kernel_12828(int32_t sizze_11838, int32_t i_11919,
                               int32_t x_11921, __global
                               unsigned char *mem_13301, __global
                               unsigned char *mem_13305, __global
                               unsigned char *mem_13318, __global
                               unsigned char *mem_13320, __global
                               unsigned char *mem_13323)
{
    int32_t wave_sizze_13523;
    int32_t group_sizze_13524;
    bool thread_active_13525;
    int32_t gtid_12821;
    int32_t global_tid_12828;
    int32_t local_tid_12829;
    int32_t group_id_12830;
    
    global_tid_12828 = get_global_id(0);
    local_tid_12829 = get_local_id(0);
    group_sizze_13524 = get_local_size(0);
    wave_sizze_13523 = LOCKSTEP_WIDTH;
    group_id_12830 = get_group_id(0);
    gtid_12821 = global_tid_12828;
    thread_active_13525 = slt32(gtid_12821, x_11921);
    
    int32_t j_p_i_t_s_13245;
    int32_t new_index_13271;
    int32_t binop_y_13273;
    int32_t new_index_13274;
    float x_12831;
    float x_12832;
    bool res_12833;
    float res_12834;
    bool res_12835;
    float y_12836;
    float res_12837;
    
    if (thread_active_13525) {
        j_p_i_t_s_13245 = i_11919 + gtid_12821;
        new_index_13271 = squot32(j_p_i_t_s_13245, sizze_11838);
        binop_y_13273 = sizze_11838 * new_index_13271;
        new_index_13274 = j_p_i_t_s_13245 - binop_y_13273;
        x_12831 = *(__global float *) &mem_13301[(new_index_13271 *
                                                  sizze_11838 +
                                                  new_index_13274) * 4];
        x_12832 = *(__global float *) &mem_13305[(new_index_13271 *
                                                  sizze_11838 +
                                                  new_index_13274) * 4];
        res_12833 = x_12832 == 0.0F;
        res_12834 = (float) fabs(x_12832);
        res_12835 = res_12834 == 1.0F;
        y_12836 = 0.0F - x_12831;
        res_12837 = x_12832 / y_12836;
    }
    if (thread_active_13525) {
        *(__global bool *) &mem_13318[gtid_12821] = res_12833;
    }
    if (thread_active_13525) {
        *(__global bool *) &mem_13320[gtid_12821] = res_12835;
    }
    if (thread_active_13525) {
        *(__global float *) &mem_13323[gtid_12821 * 4] = res_12837;
    }
}
__kernel void map_kernel_12935(int32_t x_11921, int32_t y_12878, __global
                               unsigned char *mem_13361, __global
                               unsigned char *mem_13376, __global
                               unsigned char *mem_13379)
{
    int32_t wave_sizze_13569;
    int32_t group_sizze_13570;
    bool thread_active_13571;
    int32_t j_12920;
    int32_t global_tid_12935;
    int32_t local_tid_12936;
    int32_t group_id_12937;
    
    global_tid_12935 = get_global_id(0);
    local_tid_12936 = get_local_id(0);
    group_sizze_13570 = get_local_size(0);
    wave_sizze_13569 = LOCKSTEP_WIDTH;
    group_id_12937 = get_group_id(0);
    j_12920 = global_tid_12935;
    thread_active_13571 = slt32(j_12920, x_11921);
    
    int32_t x_12918;
    int32_t group_id_12925;
    bool cond_12926;
    int32_t final_result_12928;
    
    if (thread_active_13571) {
        x_12918 = *(__global int32_t *) &mem_13361[j_12920 * 4];
        group_id_12925 = squot32(j_12920, y_12878);
        cond_12926 = 0 == group_id_12925;
        if (cond_12926) {
            final_result_12928 = x_12918;
        } else {
            int32_t carry_in_index_12927;
            int32_t x_12917;
            int32_t res_12919;
            
            carry_in_index_12927 = group_id_12925 - 1;
            x_12917 = *(__global int32_t *) &mem_13376[carry_in_index_12927 *
                                                       4];
            res_12919 = x_12917 + x_12918;
            final_result_12928 = res_12919;
        }
    }
    if (thread_active_13571) {
        *(__global int32_t *) &mem_13379[j_12920 * 4] = final_result_12928;
    }
}
__kernel void map_kernel_13044(int32_t y_12984, int32_t convop_x_13352, __global
                               unsigned char *mem_13382, __global
                               unsigned char *mem_13407, __global
                               unsigned char *mem_13410)
{
    int32_t wave_sizze_13600;
    int32_t group_sizze_13601;
    bool thread_active_13602;
    int32_t j_13029;
    int32_t global_tid_13044;
    int32_t local_tid_13045;
    int32_t group_id_13046;
    
    global_tid_13044 = get_global_id(0);
    local_tid_13045 = get_local_id(0);
    group_sizze_13601 = get_local_size(0);
    wave_sizze_13600 = LOCKSTEP_WIDTH;
    group_id_13046 = get_group_id(0);
    j_13029 = global_tid_13044;
    thread_active_13602 = slt32(j_13029, convop_x_13352);
    
    int32_t y_13027;
    int32_t group_id_13034;
    bool cond_13035;
    int32_t final_result_13037;
    
    if (thread_active_13602) {
        y_13027 = *(__global int32_t *) &mem_13382[j_13029 * 4];
        group_id_13034 = squot32(j_13029, y_12984);
        cond_13035 = 0 == group_id_13034;
        if (cond_13035) {
            final_result_13037 = y_13027;
        } else {
            int32_t carry_in_index_13036;
            int32_t x_13026;
            int32_t zz_13028;
            
            carry_in_index_13036 = group_id_13034 - 1;
            x_13026 = *(__global int32_t *) &mem_13407[carry_in_index_13036 *
                                                       4];
            zz_13028 = x_13026 + y_13027;
            final_result_13037 = zz_13028;
        }
    }
    if (thread_active_13602) {
        *(__global int32_t *) &mem_13410[j_13029 * 4] = final_result_13037;
    }
}
__kernel void map_kernel_13054(int32_t res_11895, int32_t partition_sizze_12384,
                               int32_t convop_x_13352, __global
                               unsigned char *mem_13385, __global
                               unsigned char *mem_13389, __global
                               unsigned char *mem_13410, __global
                               unsigned char *mem_13413, __global
                               unsigned char *mem_13416, __global
                               unsigned char *mem_13420)
{
    int32_t wave_sizze_13603;
    int32_t group_sizze_13604;
    bool thread_active_13605;
    int32_t write_i_13047;
    int32_t global_tid_13054;
    int32_t local_tid_13055;
    int32_t group_id_13056;
    
    global_tid_13054 = get_global_id(0);
    local_tid_13055 = get_local_id(0);
    group_sizze_13604 = get_local_size(0);
    wave_sizze_13603 = LOCKSTEP_WIDTH;
    group_id_13056 = get_group_id(0);
    write_i_13047 = global_tid_13054;
    thread_active_13605 = slt32(write_i_13047, convop_x_13352);
    
    int32_t c_12390;
    int32_t offset_12391;
    int32_t new_index_13258;
    int32_t binop_y_13260;
    int32_t new_index_13261;
    int32_t v_12392;
    float v_12393;
    bool is_this_one_12394;
    int32_t this_offset_12395;
    int32_t total_res_12396;
    
    if (thread_active_13605) {
        c_12390 = *(__global int32_t *) &mem_13385[write_i_13047 * 4];
        offset_12391 = *(__global int32_t *) &mem_13410[write_i_13047 * 4];
        new_index_13258 = squot32(write_i_13047, res_11895);
        binop_y_13260 = res_11895 * new_index_13258;
        new_index_13261 = write_i_13047 - binop_y_13260;
        v_12392 = *(__global int32_t *) &mem_13389[(new_index_13258 *
                                                    res_11895 +
                                                    new_index_13261) * 4];
        v_12393 = *(__global float *) &mem_13420[(new_index_13258 * res_11895 +
                                                  new_index_13261) * 4];
        is_this_one_12394 = c_12390 == 0;
        this_offset_12395 = -1 + offset_12391;
        if (is_this_one_12394) {
            total_res_12396 = this_offset_12395;
        } else {
            total_res_12396 = -1;
        }
    }
    if (thread_active_13605 && (sle32(0, total_res_12396) &&
                                slt32(total_res_12396,
                                      partition_sizze_12384))) {
        *(__global int32_t *) &mem_13413[total_res_12396 * 4] = v_12392;
    }
    if (thread_active_13605 && (sle32(0, total_res_12396) &&
                                slt32(total_res_12396,
                                      partition_sizze_12384))) {
        *(__global float *) &mem_13416[total_res_12396 * 4] = v_12393;
    }
}
__kernel void map_kernel_13064(int32_t x_11921, int32_t res_12409, __global
                               unsigned char *mem_13379, __global
                               unsigned char *mem_13423)
{
    int32_t wave_sizze_13610;
    int32_t group_sizze_13611;
    bool thread_active_13612;
    int32_t write_i_13057;
    int32_t global_tid_13064;
    int32_t local_tid_13065;
    int32_t group_id_13066;
    
    global_tid_13064 = get_global_id(0);
    local_tid_13065 = get_local_id(0);
    group_sizze_13611 = get_local_size(0);
    wave_sizze_13610 = LOCKSTEP_WIDTH;
    group_id_13066 = get_group_id(0);
    write_i_13057 = global_tid_13064;
    thread_active_13612 = slt32(write_i_13057, x_11921);
    
    bool cond_12422;
    int32_t res_12423;
    
    if (thread_active_13612) {
        cond_12422 = write_i_13057 == 0;
        if (cond_12422) {
            res_12423 = 0;
        } else {
            int32_t i_12424;
            int32_t res_12425;
            
            i_12424 = write_i_13057 - 1;
            res_12425 = *(__global int32_t *) &mem_13379[i_12424 * 4];
            res_12423 = res_12425;
        }
    }
    if (thread_active_13612 && (sle32(0, res_12423) && slt32(res_12423,
                                                             res_12409))) {
        *(__global int32_t *) &mem_13423[res_12423 * 4] = 1;
    }
}
__kernel void map_kernel_13214(int32_t res_12409, int32_t y_13126, __global
                               unsigned char *mem_13426, __global
                               unsigned char *mem_13429, __global
                               unsigned char *mem_13456, __global
                               unsigned char *mem_13459, __global
                               unsigned char *mem_13462, __global
                               unsigned char *mem_13465)
{
    int32_t wave_sizze_13653;
    int32_t group_sizze_13654;
    bool thread_active_13655;
    int32_t j_13197;
    int32_t global_tid_13214;
    int32_t local_tid_13215;
    int32_t group_id_13216;
    
    global_tid_13214 = get_global_id(0);
    local_tid_13215 = get_local_id(0);
    group_sizze_13654 = get_local_size(0);
    wave_sizze_13653 = LOCKSTEP_WIDTH;
    group_id_13216 = get_group_id(0);
    j_13197 = global_tid_13214;
    thread_active_13655 = slt32(j_13197, res_12409);
    
    int32_t x_13191;
    float x_13192;
    int32_t group_id_13202;
    bool cond_13203;
    int32_t final_result_13206;
    float final_result_13207;
    
    if (thread_active_13655) {
        x_13191 = *(__global int32_t *) &mem_13426[j_13197 * 4];
        x_13192 = *(__global float *) &mem_13429[j_13197 * 4];
        group_id_13202 = squot32(j_13197, y_13126);
        cond_13203 = 0 == group_id_13202;
        if (cond_13203) {
            final_result_13206 = x_13191;
            final_result_13207 = x_13192;
        } else {
            int32_t carry_in_index_13204;
            int32_t x_13189;
            float x_13190;
            int32_t res_13193;
            bool cond_13194;
            float res_13195;
            
            carry_in_index_13204 = group_id_13202 - 1;
            x_13189 = *(__global int32_t *) &mem_13456[carry_in_index_13204 *
                                                       4];
            x_13190 = *(__global float *) &mem_13459[carry_in_index_13204 * 4];
            res_13193 = x_13189 | x_13191;
            cond_13194 = slt32(0, x_13191);
            if (cond_13194) {
                res_13195 = x_13192;
            } else {
                float res_13196 = x_13190 + x_13192;
                
                res_13195 = res_13196;
            }
            final_result_13206 = res_13193;
            final_result_13207 = res_13195;
        }
    }
    if (thread_active_13655) {
        *(__global int32_t *) &mem_13462[j_13197 * 4] = final_result_13206;
    }
    if (thread_active_13655) {
        *(__global float *) &mem_13465[j_13197 * 4] = final_result_13207;
    }
}
__kernel void map_kernel_13224(int32_t x_11921, __global
                               unsigned char *mem_13379, __global
                               unsigned char *mem_13465, __global
                               unsigned char *mem_13468)
{
    int32_t wave_sizze_13656;
    int32_t group_sizze_13657;
    bool thread_active_13658;
    int32_t gtid_13217;
    int32_t global_tid_13224;
    int32_t local_tid_13225;
    int32_t group_id_13226;
    
    global_tid_13224 = get_global_id(0);
    local_tid_13225 = get_local_id(0);
    group_sizze_13657 = get_local_size(0);
    wave_sizze_13656 = LOCKSTEP_WIDTH;
    group_id_13226 = get_group_id(0);
    gtid_13217 = global_tid_13224;
    thread_active_13658 = slt32(gtid_13217, x_11921);
    
    int32_t x_13227;
    int32_t res_13228;
    float res_13229;
    
    if (thread_active_13658) {
        x_13227 = *(__global int32_t *) &mem_13379[gtid_13217 * 4];
        res_13228 = x_13227 - 1;
        res_13229 = *(__global float *) &mem_13465[res_13228 * 4];
    }
    if (thread_active_13658) {
        *(__global float *) &mem_13468[gtid_13217 * 4] = res_13229;
    }
}
__kernel void scan1_kernel_12870(__local volatile int64_t *mem_aligned_0,
                                 int32_t x_11921, int32_t num_iterations_12875,
                                 int32_t y_12878, __global
                                 unsigned char *mem_13350, __global
                                 unsigned char *mem_13361, __global
                                 unsigned char *mem_13370)
{
    __local volatile char *restrict mem_13364 = mem_aligned_0;
    int32_t wave_sizze_13542;
    int32_t group_sizze_13543;
    bool thread_active_13544;
    int32_t global_tid_12870;
    int32_t local_tid_12871;
    int32_t group_id_12872;
    
    global_tid_12870 = get_global_id(0);
    local_tid_12871 = get_local_id(0);
    group_sizze_13543 = get_local_size(0);
    wave_sizze_13542 = LOCKSTEP_WIDTH;
    group_id_12872 = get_group_id(0);
    thread_active_13544 = 1;
    
    int32_t x_12879;
    bool is_first_thread_12893;
    int32_t result_12900;
    
    if (thread_active_13544) {
        x_12879 = group_id_12872 * y_12878;
        is_first_thread_12893 = local_tid_12871 == 0;
        
        int32_t x_merge_12876 = 0;
        
        for (int32_t i_12877 = 0; i_12877 < num_iterations_12875; i_12877++) {
            int32_t y_12880 = group_sizze_12839 * i_12877;
            int32_t offset_12881 = x_12879 + y_12880;
            int32_t j_12882 = local_tid_12871 + offset_12881;
            bool cond_12883 = slt32(j_12882, x_11921);
            int32_t foldres_12885;
            
            if (cond_12883) {
                int32_t to_scan_elem_12884;
                int32_t res_12857;
                
                to_scan_elem_12884 = *(__global int32_t *) &mem_13350[j_12882 *
                                                                      4];
                res_12857 = x_merge_12876 + to_scan_elem_12884;
                foldres_12885 = res_12857;
            } else {
                foldres_12885 = x_merge_12876;
            }
            for (int32_t comb_iter_13547 = 0; comb_iter_13547 <
                 squot32(group_sizze_12839 + group_sizze_12839 - 1,
                         group_sizze_12839); comb_iter_13547++) {
                int32_t combine_id_12886;
                int32_t flat_comb_id_13548 = comb_iter_13547 *
                        group_sizze_12839 + local_tid_12871;
                
                combine_id_12886 = flat_comb_id_13548;
                if (slt32(combine_id_12886, group_sizze_12839) && 1) {
                    *(__local int32_t *) &mem_13364[combine_id_12886 * 4] =
                        foldres_12885;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t my_index_12858;
            int32_t other_index_12859;
            int32_t x_12860;
            int32_t x_12861;
            int32_t my_index_13549;
            int32_t other_index_13550;
            int32_t x_13551;
            int32_t x_13552;
            
            my_index_12858 = local_tid_12871;
            if (slt32(local_tid_12871, group_sizze_12839)) {
                x_12861 = *(volatile __local
                            int32_t *) &mem_13364[local_tid_12871 *
                                                  sizeof(int32_t)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                int32_t skip_threads_13554 = 1;
                
                while (slt32(skip_threads_13554, 32)) {
                    if (slt32(local_tid_12871, group_sizze_12839) &&
                        sle32(skip_threads_13554, local_tid_12871 -
                              squot32(local_tid_12871, 32) * 32)) {
                        // read operands
                        {
                            x_12860 = *(volatile __local
                                        int32_t *) &mem_13364[(local_tid_12871 -
                                                               skip_threads_13554) *
                                                              sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            int32_t res_12862 = x_12860 + x_12861;
                            
                            x_12861 = res_12862;
                        }
                    }
                    if (sle32(wave_sizze_13542, skip_threads_13554)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (slt32(local_tid_12871, group_sizze_12839) &&
                        sle32(skip_threads_13554, local_tid_12871 -
                              squot32(local_tid_12871, 32) * 32)) {
                        // write result
                        {
                            *(volatile __local
                              int32_t *) &mem_13364[local_tid_12871 *
                                                    sizeof(int32_t)] = x_12861;
                        }
                    }
                    if (sle32(wave_sizze_13542, skip_threads_13554)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_13554 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_12871 - squot32(local_tid_12871, 32) * 32) ==
                    31 && slt32(local_tid_12871, group_sizze_12839)) {
                    *(volatile __local
                      int32_t *) &mem_13364[squot32(local_tid_12871, 32) *
                                            sizeof(int32_t)] = x_12861;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
            {
                if (squot32(local_tid_12871, 32) == 0 && slt32(local_tid_12871,
                                                               group_sizze_12839)) {
                    x_13552 = *(volatile __local
                                int32_t *) &mem_13364[local_tid_12871 *
                                                      sizeof(int32_t)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    int32_t skip_threads_13555 = 1;
                    
                    while (slt32(skip_threads_13555, 32)) {
                        if ((squot32(local_tid_12871, 32) == 0 &&
                             slt32(local_tid_12871, group_sizze_12839)) &&
                            sle32(skip_threads_13555, local_tid_12871 -
                                  squot32(local_tid_12871, 32) * 32)) {
                            // read operands
                            {
                                x_13551 = *(volatile __local
                                            int32_t *) &mem_13364[(local_tid_12871 -
                                                                   skip_threads_13555) *
                                                                  sizeof(int32_t)];
                            }
                            // perform operation
                            {
                                int32_t res_13553 = x_13551 + x_13552;
                                
                                x_13552 = res_13553;
                            }
                        }
                        if (sle32(wave_sizze_13542, skip_threads_13555)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if ((squot32(local_tid_12871, 32) == 0 &&
                             slt32(local_tid_12871, group_sizze_12839)) &&
                            sle32(skip_threads_13555, local_tid_12871 -
                                  squot32(local_tid_12871, 32) * 32)) {
                            // write result
                            {
                                *(volatile __local
                                  int32_t *) &mem_13364[local_tid_12871 *
                                                        sizeof(int32_t)] =
                                    x_13552;
                            }
                        }
                        if (sle32(wave_sizze_13542, skip_threads_13555)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_13555 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_12871, 32) == 0 ||
                      !slt32(local_tid_12871, group_sizze_12839))) {
                    // read operands
                    {
                        x_12860 = *(volatile __local
                                    int32_t *) &mem_13364[(squot32(local_tid_12871,
                                                                   32) - 1) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        int32_t res_12862 = x_12860 + x_12861;
                        
                        x_12861 = res_12862;
                    }
                    // write final result
                    {
                        *(volatile __local
                          int32_t *) &mem_13364[local_tid_12871 *
                                                sizeof(int32_t)] = x_12861;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_12871, 32) == 0) {
                    *(volatile __local int32_t *) &mem_13364[local_tid_12871 *
                                                             sizeof(int32_t)] =
                        x_12861;
                }
            }
            if (cond_12883) {
                int32_t scanned_elem_12890 = *(__local
                                               int32_t *) &mem_13364[local_tid_12871 *
                                                                     4];
                
                *(__global int32_t *) &mem_13361[j_12882 * 4] =
                    scanned_elem_12890;
            }
            
            int32_t new_scan_carry_12895;
            
            if (is_first_thread_12893) {
                int32_t carry_12894 = *(__local int32_t *) &mem_13364[y_12842 *
                                                                      4];
                
                new_scan_carry_12895 = carry_12894;
            } else {
                new_scan_carry_12895 = 0;
            }
            
            int32_t new_carry_sync_12898;
            
            new_carry_sync_12898 = new_scan_carry_12895;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_merge_tmp_13546 = new_carry_sync_12898;
            
            x_merge_12876 = x_merge_tmp_13546;
        }
        result_12900 = x_merge_12876;
    }
    if (local_tid_12871 == 0) {
        *(__global int32_t *) &mem_13370[group_id_12872 * 4] = result_12900;
    }
}
__kernel void scan1_kernel_12976(__local volatile int64_t *mem_aligned_0,
                                 int32_t res_11895,
                                 int32_t num_iterations_12981, int32_t y_12984,
                                 int32_t convop_x_13352, __global
                                 unsigned char *mem_13382, __global
                                 unsigned char *mem_13385, __global
                                 unsigned char *mem_13389, __global
                                 unsigned char *mem_13401)
{
    __local volatile char *restrict mem_13394 = mem_aligned_0;
    int32_t wave_sizze_13572;
    int32_t group_sizze_13573;
    bool thread_active_13574;
    int32_t global_tid_12976;
    int32_t local_tid_12977;
    int32_t group_id_12978;
    
    global_tid_12976 = get_global_id(0);
    local_tid_12977 = get_local_id(0);
    group_sizze_13573 = get_local_size(0);
    wave_sizze_13572 = LOCKSTEP_WIDTH;
    group_id_12978 = get_group_id(0);
    thread_active_13574 = 1;
    
    int32_t x_12985;
    bool is_first_thread_13001;
    int32_t result_13009;
    
    if (thread_active_13574) {
        x_12985 = group_id_12978 * y_12984;
        is_first_thread_13001 = local_tid_12977 == 0;
        
        int32_t x_merge_12982 = 0;
        
        for (int32_t i_12983 = 0; i_12983 < num_iterations_12981; i_12983++) {
            int32_t y_12986 = group_sizze_12939 * i_12983;
            int32_t offset_12987 = x_12985 + y_12986;
            int32_t j_12988 = local_tid_12977 + offset_12987;
            bool cond_12989 = slt32(j_12988, convop_x_13352);
            int32_t foldres_12992;
            
            if (cond_12989) {
                int32_t new_index_13254;
                int32_t binop_y_13256;
                int32_t new_index_13257;
                int32_t res_elem_12990;
                bool res_12956;
                bool res_12957;
                int32_t part_res_12958;
                int32_t part_res_12959;
                int32_t zz_12961;
                
                new_index_13254 = squot32(j_12988, res_11895);
                binop_y_13256 = res_11895 * new_index_13254;
                new_index_13257 = j_12988 - binop_y_13256;
                res_elem_12990 = *(__global
                                   int32_t *) &mem_13389[(new_index_13254 *
                                                          res_11895 +
                                                          new_index_13257) * 4];
                res_12956 = res_elem_12990 == -1;
                res_12957 = !res_12956;
                if (res_12957) {
                    part_res_12958 = 0;
                } else {
                    part_res_12958 = 1;
                }
                if (res_12957) {
                    part_res_12959 = 1;
                } else {
                    part_res_12959 = 0;
                }
                zz_12961 = part_res_12959 + x_merge_12982;
                *(__global int32_t *) &mem_13385[j_12988 * 4] = part_res_12958;
                foldres_12992 = zz_12961;
            } else {
                foldres_12992 = x_merge_12982;
            }
            for (int32_t comb_iter_13578 = 0; comb_iter_13578 <
                 squot32(group_sizze_12939 + group_sizze_12939 - 1,
                         group_sizze_12939); comb_iter_13578++) {
                int32_t combine_id_12994;
                int32_t flat_comb_id_13579 = comb_iter_13578 *
                        group_sizze_12939 + local_tid_12977;
                
                combine_id_12994 = flat_comb_id_13579;
                if (slt32(combine_id_12994, group_sizze_12939) && 1) {
                    *(__local int32_t *) &mem_13394[combine_id_12994 * 4] =
                        foldres_12992;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t my_index_12962;
            int32_t other_index_12963;
            int32_t x_12964;
            int32_t y_12965;
            int32_t my_index_13580;
            int32_t other_index_13581;
            int32_t x_13582;
            int32_t y_13583;
            
            my_index_12962 = local_tid_12977;
            if (slt32(local_tid_12977, group_sizze_12939)) {
                y_12965 = *(volatile __local
                            int32_t *) &mem_13394[local_tid_12977 *
                                                  sizeof(int32_t)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                int32_t skip_threads_13585 = 1;
                
                while (slt32(skip_threads_13585, 32)) {
                    if (slt32(local_tid_12977, group_sizze_12939) &&
                        sle32(skip_threads_13585, local_tid_12977 -
                              squot32(local_tid_12977, 32) * 32)) {
                        // read operands
                        {
                            x_12964 = *(volatile __local
                                        int32_t *) &mem_13394[(local_tid_12977 -
                                                               skip_threads_13585) *
                                                              sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            int32_t zz_12966 = x_12964 + y_12965;
                            
                            y_12965 = zz_12966;
                        }
                    }
                    if (sle32(wave_sizze_13572, skip_threads_13585)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (slt32(local_tid_12977, group_sizze_12939) &&
                        sle32(skip_threads_13585, local_tid_12977 -
                              squot32(local_tid_12977, 32) * 32)) {
                        // write result
                        {
                            *(volatile __local
                              int32_t *) &mem_13394[local_tid_12977 *
                                                    sizeof(int32_t)] = y_12965;
                        }
                    }
                    if (sle32(wave_sizze_13572, skip_threads_13585)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_13585 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_12977 - squot32(local_tid_12977, 32) * 32) ==
                    31 && slt32(local_tid_12977, group_sizze_12939)) {
                    *(volatile __local
                      int32_t *) &mem_13394[squot32(local_tid_12977, 32) *
                                            sizeof(int32_t)] = y_12965;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
            {
                if (squot32(local_tid_12977, 32) == 0 && slt32(local_tid_12977,
                                                               group_sizze_12939)) {
                    y_13583 = *(volatile __local
                                int32_t *) &mem_13394[local_tid_12977 *
                                                      sizeof(int32_t)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    int32_t skip_threads_13586 = 1;
                    
                    while (slt32(skip_threads_13586, 32)) {
                        if ((squot32(local_tid_12977, 32) == 0 &&
                             slt32(local_tid_12977, group_sizze_12939)) &&
                            sle32(skip_threads_13586, local_tid_12977 -
                                  squot32(local_tid_12977, 32) * 32)) {
                            // read operands
                            {
                                x_13582 = *(volatile __local
                                            int32_t *) &mem_13394[(local_tid_12977 -
                                                                   skip_threads_13586) *
                                                                  sizeof(int32_t)];
                            }
                            // perform operation
                            {
                                int32_t zz_13584 = x_13582 + y_13583;
                                
                                y_13583 = zz_13584;
                            }
                        }
                        if (sle32(wave_sizze_13572, skip_threads_13586)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if ((squot32(local_tid_12977, 32) == 0 &&
                             slt32(local_tid_12977, group_sizze_12939)) &&
                            sle32(skip_threads_13586, local_tid_12977 -
                                  squot32(local_tid_12977, 32) * 32)) {
                            // write result
                            {
                                *(volatile __local
                                  int32_t *) &mem_13394[local_tid_12977 *
                                                        sizeof(int32_t)] =
                                    y_13583;
                            }
                        }
                        if (sle32(wave_sizze_13572, skip_threads_13586)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_13586 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_12977, 32) == 0 ||
                      !slt32(local_tid_12977, group_sizze_12939))) {
                    // read operands
                    {
                        x_12964 = *(volatile __local
                                    int32_t *) &mem_13394[(squot32(local_tid_12977,
                                                                   32) - 1) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        int32_t zz_12966 = x_12964 + y_12965;
                        
                        y_12965 = zz_12966;
                    }
                    // write final result
                    {
                        *(volatile __local
                          int32_t *) &mem_13394[local_tid_12977 *
                                                sizeof(int32_t)] = y_12965;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_12977, 32) == 0) {
                    *(volatile __local int32_t *) &mem_13394[local_tid_12977 *
                                                             sizeof(int32_t)] =
                        y_12965;
                }
            }
            if (cond_12989) {
                int32_t scanned_elem_12998 = *(__local
                                               int32_t *) &mem_13394[local_tid_12977 *
                                                                     4];
                
                *(__global int32_t *) &mem_13382[j_12988 * 4] =
                    scanned_elem_12998;
            }
            
            int32_t new_scan_carry_13003;
            
            if (is_first_thread_13001) {
                int32_t carry_13002 = *(__local int32_t *) &mem_13394[y_12942 *
                                                                      4];
                
                new_scan_carry_13003 = carry_13002;
            } else {
                new_scan_carry_13003 = 0;
            }
            
            int32_t new_carry_sync_13006;
            
            new_carry_sync_13006 = new_scan_carry_13003;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_merge_tmp_13577 = new_carry_sync_13006;
            
            x_merge_12982 = x_merge_tmp_13577;
        }
        result_13009 = x_merge_12982;
    }
    if (local_tid_12977 == 0) {
        *(__global int32_t *) &mem_13401[group_id_12978 * 4] = result_13009;
    }
}
__kernel void scan1_kernel_13117(__local volatile int64_t *mem_aligned_0,
                                 __local volatile int64_t *mem_aligned_1,
                                 int32_t res_12409,
                                 int32_t num_iterations_13122, int32_t y_13126,
                                 __global unsigned char *voxels_mem_13291,
                                 __global unsigned char *mem_13413, __global
                                 unsigned char *mem_13416, __global
                                 unsigned char *mem_13423, __global
                                 unsigned char *mem_13426, __global
                                 unsigned char *mem_13429, __global
                                 unsigned char *mem_13444, __global
                                 unsigned char *mem_13447)
{
    __local volatile char *restrict mem_13432 = mem_aligned_0;
    __local volatile char *restrict mem_13435 = mem_aligned_1;
    int32_t wave_sizze_13613;
    int32_t group_sizze_13614;
    bool thread_active_13615;
    int32_t global_tid_13117;
    int32_t local_tid_13118;
    int32_t group_id_13119;
    
    global_tid_13117 = get_global_id(0);
    local_tid_13118 = get_local_id(0);
    group_sizze_13614 = get_local_size(0);
    wave_sizze_13613 = LOCKSTEP_WIDTH;
    group_id_13119 = get_group_id(0);
    thread_active_13615 = 1;
    
    int32_t x_13127;
    bool is_first_thread_13149;
    int32_t result_13160;
    float result_13161;
    
    if (thread_active_13615) {
        x_13127 = group_id_13119 * y_13126;
        is_first_thread_13149 = local_tid_13118 == 0;
        
        int32_t x_merge_13123;
        float x_merge_13124;
        
        x_merge_13123 = 0;
        x_merge_13124 = 0.0F;
        for (int32_t i_13125 = 0; i_13125 < num_iterations_13122; i_13125++) {
            int32_t y_13128 = group_sizze_13068 * i_13125;
            int32_t offset_13129 = x_13127 + y_13128;
            int32_t j_13130 = local_tid_13118 + offset_13129;
            bool cond_13131 = slt32(j_13130, res_12409);
            int32_t foldres_13135;
            float foldres_13136;
            
            if (cond_13131) {
                int32_t reshape_outer_elem_13132;
                float reshape_outer_elem_13133;
                int32_t res_elem_13134;
                float y_13088;
                float res_13089;
                int32_t res_13092;
                bool cond_13093;
                float res_13094;
                
                reshape_outer_elem_13132 = *(__global
                                             int32_t *) &mem_13413[j_13130 * 4];
                reshape_outer_elem_13133 = *(__global
                                             float *) &mem_13416[j_13130 * 4];
                res_elem_13134 = *(__global int32_t *) &mem_13423[j_13130 * 4];
                y_13088 = *(__global
                            float *) &voxels_mem_13291[reshape_outer_elem_13132 *
                                                       4];
                res_13089 = y_13088 * reshape_outer_elem_13133;
                res_13092 = x_merge_13123 | res_elem_13134;
                cond_13093 = slt32(0, res_elem_13134);
                if (cond_13093) {
                    res_13094 = res_13089;
                } else {
                    float res_13095 = res_13089 + x_merge_13124;
                    
                    res_13094 = res_13095;
                }
                foldres_13135 = res_13092;
                foldres_13136 = res_13094;
            } else {
                foldres_13135 = x_merge_13123;
                foldres_13136 = x_merge_13124;
            }
            for (int32_t comb_iter_13620 = 0; comb_iter_13620 <
                 squot32(group_sizze_13068 + group_sizze_13068 - 1,
                         group_sizze_13068); comb_iter_13620++) {
                int32_t combine_id_13137;
                int32_t flat_comb_id_13621 = comb_iter_13620 *
                        group_sizze_13068 + local_tid_13118;
                
                combine_id_13137 = flat_comb_id_13621;
                if (slt32(combine_id_13137, group_sizze_13068) && 1) {
                    *(__local int32_t *) &mem_13432[combine_id_13137 * 4] =
                        foldres_13135;
                    *(__local float *) &mem_13435[combine_id_13137 * 4] =
                        foldres_13136;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t my_index_13096;
            int32_t other_index_13097;
            int32_t x_13098;
            float x_13099;
            int32_t x_13100;
            float x_13101;
            int32_t my_index_13622;
            int32_t other_index_13623;
            int32_t x_13624;
            float x_13625;
            int32_t x_13626;
            float x_13627;
            
            my_index_13096 = local_tid_13118;
            if (slt32(local_tid_13118, group_sizze_13068)) {
                x_13100 = *(volatile __local
                            int32_t *) &mem_13432[local_tid_13118 *
                                                  sizeof(int32_t)];
                x_13101 = *(volatile __local
                            float *) &mem_13435[local_tid_13118 *
                                                sizeof(float)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                int32_t skip_threads_13632 = 1;
                
                while (slt32(skip_threads_13632, 32)) {
                    if (slt32(local_tid_13118, group_sizze_13068) &&
                        sle32(skip_threads_13632, local_tid_13118 -
                              squot32(local_tid_13118, 32) * 32)) {
                        // read operands
                        {
                            x_13098 = *(volatile __local
                                        int32_t *) &mem_13432[(local_tid_13118 -
                                                               skip_threads_13632) *
                                                              sizeof(int32_t)];
                            x_13099 = *(volatile __local
                                        float *) &mem_13435[(local_tid_13118 -
                                                             skip_threads_13632) *
                                                            sizeof(float)];
                        }
                        // perform operation
                        {
                            int32_t res_13102;
                            bool cond_13103;
                            float res_13104;
                            
                            res_13102 = x_13098 | x_13100;
                            cond_13103 = slt32(0, x_13100);
                            if (cond_13103) {
                                res_13104 = x_13101;
                            } else {
                                float res_13105 = x_13099 + x_13101;
                                
                                res_13104 = res_13105;
                            }
                            x_13100 = res_13102;
                            x_13101 = res_13104;
                        }
                    }
                    if (sle32(wave_sizze_13613, skip_threads_13632)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (slt32(local_tid_13118, group_sizze_13068) &&
                        sle32(skip_threads_13632, local_tid_13118 -
                              squot32(local_tid_13118, 32) * 32)) {
                        // write result
                        {
                            *(volatile __local
                              int32_t *) &mem_13432[local_tid_13118 *
                                                    sizeof(int32_t)] = x_13100;
                            *(volatile __local
                              float *) &mem_13435[local_tid_13118 *
                                                  sizeof(float)] = x_13101;
                        }
                    }
                    if (sle32(wave_sizze_13613, skip_threads_13632)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_13632 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_13118 - squot32(local_tid_13118, 32) * 32) ==
                    31 && slt32(local_tid_13118, group_sizze_13068)) {
                    *(volatile __local
                      int32_t *) &mem_13432[squot32(local_tid_13118, 32) *
                                            sizeof(int32_t)] = x_13100;
                    *(volatile __local
                      float *) &mem_13435[squot32(local_tid_13118, 32) *
                                          sizeof(float)] = x_13101;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
            {
                if (squot32(local_tid_13118, 32) == 0 && slt32(local_tid_13118,
                                                               group_sizze_13068)) {
                    x_13626 = *(volatile __local
                                int32_t *) &mem_13432[local_tid_13118 *
                                                      sizeof(int32_t)];
                    x_13627 = *(volatile __local
                                float *) &mem_13435[local_tid_13118 *
                                                    sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    int32_t skip_threads_13633 = 1;
                    
                    while (slt32(skip_threads_13633, 32)) {
                        if ((squot32(local_tid_13118, 32) == 0 &&
                             slt32(local_tid_13118, group_sizze_13068)) &&
                            sle32(skip_threads_13633, local_tid_13118 -
                                  squot32(local_tid_13118, 32) * 32)) {
                            // read operands
                            {
                                x_13624 = *(volatile __local
                                            int32_t *) &mem_13432[(local_tid_13118 -
                                                                   skip_threads_13633) *
                                                                  sizeof(int32_t)];
                                x_13625 = *(volatile __local
                                            float *) &mem_13435[(local_tid_13118 -
                                                                 skip_threads_13633) *
                                                                sizeof(float)];
                            }
                            // perform operation
                            {
                                int32_t res_13628;
                                bool cond_13629;
                                float res_13630;
                                
                                res_13628 = x_13624 | x_13626;
                                cond_13629 = slt32(0, x_13626);
                                if (cond_13629) {
                                    res_13630 = x_13627;
                                } else {
                                    float res_13631 = x_13625 + x_13627;
                                    
                                    res_13630 = res_13631;
                                }
                                x_13626 = res_13628;
                                x_13627 = res_13630;
                            }
                        }
                        if (sle32(wave_sizze_13613, skip_threads_13633)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if ((squot32(local_tid_13118, 32) == 0 &&
                             slt32(local_tid_13118, group_sizze_13068)) &&
                            sle32(skip_threads_13633, local_tid_13118 -
                                  squot32(local_tid_13118, 32) * 32)) {
                            // write result
                            {
                                *(volatile __local
                                  int32_t *) &mem_13432[local_tid_13118 *
                                                        sizeof(int32_t)] =
                                    x_13626;
                                *(volatile __local
                                  float *) &mem_13435[local_tid_13118 *
                                                      sizeof(float)] = x_13627;
                            }
                        }
                        if (sle32(wave_sizze_13613, skip_threads_13633)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_13633 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_13118, 32) == 0 ||
                      !slt32(local_tid_13118, group_sizze_13068))) {
                    // read operands
                    {
                        x_13098 = *(volatile __local
                                    int32_t *) &mem_13432[(squot32(local_tid_13118,
                                                                   32) - 1) *
                                                          sizeof(int32_t)];
                        x_13099 = *(volatile __local
                                    float *) &mem_13435[(squot32(local_tid_13118,
                                                                 32) - 1) *
                                                        sizeof(float)];
                    }
                    // perform operation
                    {
                        int32_t res_13102;
                        bool cond_13103;
                        float res_13104;
                        
                        res_13102 = x_13098 | x_13100;
                        cond_13103 = slt32(0, x_13100);
                        if (cond_13103) {
                            res_13104 = x_13101;
                        } else {
                            float res_13105 = x_13099 + x_13101;
                            
                            res_13104 = res_13105;
                        }
                        x_13100 = res_13102;
                        x_13101 = res_13104;
                    }
                    // write final result
                    {
                        *(volatile __local
                          int32_t *) &mem_13432[local_tid_13118 *
                                                sizeof(int32_t)] = x_13100;
                        *(volatile __local float *) &mem_13435[local_tid_13118 *
                                                               sizeof(float)] =
                            x_13101;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_13118, 32) == 0) {
                    *(volatile __local int32_t *) &mem_13432[local_tid_13118 *
                                                             sizeof(int32_t)] =
                        x_13100;
                    *(volatile __local float *) &mem_13435[local_tid_13118 *
                                                           sizeof(float)] =
                        x_13101;
                }
            }
            if (cond_13131) {
                int32_t scanned_elem_13143;
                float scanned_elem_13144;
                
                scanned_elem_13143 = *(__local
                                       int32_t *) &mem_13432[local_tid_13118 *
                                                             4];
                scanned_elem_13144 = *(__local
                                       float *) &mem_13435[local_tid_13118 * 4];
                *(__global int32_t *) &mem_13426[j_13130 * 4] =
                    scanned_elem_13143;
                *(__global float *) &mem_13429[j_13130 * 4] =
                    scanned_elem_13144;
            }
            
            int32_t new_scan_carry_13152;
            float new_scan_carry_13153;
            
            if (is_first_thread_13149) {
                int32_t carry_13150;
                float carry_13151;
                
                carry_13150 = *(__local int32_t *) &mem_13432[y_13071 * 4];
                carry_13151 = *(__local float *) &mem_13435[y_13071 * 4];
                new_scan_carry_13152 = carry_13150;
                new_scan_carry_13153 = carry_13151;
            } else {
                new_scan_carry_13152 = 0;
                new_scan_carry_13153 = 0.0F;
            }
            
            int32_t new_carry_sync_13156;
            float new_carry_sync_13157;
            
            new_carry_sync_13156 = new_scan_carry_13152;
            new_carry_sync_13157 = new_scan_carry_13153;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_merge_tmp_13618 = new_carry_sync_13156;
            float x_merge_tmp_13619;
            
            x_merge_tmp_13619 = new_carry_sync_13157;
            x_merge_13123 = x_merge_tmp_13618;
            x_merge_13124 = x_merge_tmp_13619;
        }
        result_13160 = x_merge_13123;
        result_13161 = x_merge_13124;
    }
    if (local_tid_13118 == 0) {
        *(__global int32_t *) &mem_13444[group_id_13119 * 4] = result_13160;
    }
    if (local_tid_13118 == 0) {
        *(__global float *) &mem_13447[group_id_13119 * 4] = result_13161;
    }
}
__kernel void scan2_kernel_12906(__local volatile int64_t *mem_aligned_0,
                                 int32_t num_groups_12846, __global
                                 unsigned char *mem_13370, __global
                                 unsigned char *mem_13376)
{
    __local volatile char *restrict mem_13373 = mem_aligned_0;
    int32_t wave_sizze_13557;
    int32_t group_sizze_13558;
    bool thread_active_13559;
    int32_t global_tid_12906;
    int32_t local_tid_12907;
    int32_t group_id_12908;
    
    global_tid_12906 = get_global_id(0);
    local_tid_12907 = get_local_id(0);
    group_sizze_13558 = get_local_size(0);
    wave_sizze_13557 = LOCKSTEP_WIDTH;
    group_id_12908 = get_group_id(0);
    thread_active_13559 = 1;
    for (int32_t comb_iter_13560 = 0; comb_iter_13560 <
         squot32(num_groups_12846 + num_groups_12846 - 1, num_groups_12846);
         comb_iter_13560++) {
        int32_t combine_id_12909;
        int32_t flat_comb_id_13561 = comb_iter_13560 * num_groups_12846 +
                local_tid_12907;
        
        combine_id_12909 = flat_comb_id_13561;
        if (slt32(combine_id_12909, num_groups_12846) && 1) {
            int32_t res_scan_carry_out_elem_12910 = *(__global
                                                      int32_t *) &mem_13370[combine_id_12909 *
                                                                            4];
            
            *(__local int32_t *) &mem_13373[combine_id_12909 * 4] =
                res_scan_carry_out_elem_12910;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t my_index_12901;
    int32_t other_index_12902;
    int32_t x_12903;
    int32_t x_12904;
    int32_t my_index_13562;
    int32_t other_index_13563;
    int32_t x_13564;
    int32_t x_13565;
    
    my_index_12901 = local_tid_12907;
    if (slt32(local_tid_12907, num_groups_12846)) {
        x_12904 = *(volatile __local int32_t *) &mem_13373[local_tid_12907 *
                                                           sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        int32_t skip_threads_13567 = 1;
        
        while (slt32(skip_threads_13567, 32)) {
            if (slt32(local_tid_12907, num_groups_12846) &&
                sle32(skip_threads_13567, local_tid_12907 -
                      squot32(local_tid_12907, 32) * 32)) {
                // read operands
                {
                    x_12903 = *(volatile __local
                                int32_t *) &mem_13373[(local_tid_12907 -
                                                       skip_threads_13567) *
                                                      sizeof(int32_t)];
                }
                // perform operation
                {
                    int32_t res_12905;
                    
                    if (thread_active_13559) {
                        res_12905 = x_12903 + x_12904;
                    }
                    x_12904 = res_12905;
                }
            }
            if (sle32(wave_sizze_13557, skip_threads_13567)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (slt32(local_tid_12907, num_groups_12846) &&
                sle32(skip_threads_13567, local_tid_12907 -
                      squot32(local_tid_12907, 32) * 32)) {
                // write result
                {
                    *(volatile __local int32_t *) &mem_13373[local_tid_12907 *
                                                             sizeof(int32_t)] =
                        x_12904;
                }
            }
            if (sle32(wave_sizze_13557, skip_threads_13567)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_13567 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_12907 - squot32(local_tid_12907, 32) * 32) == 31 &&
            slt32(local_tid_12907, num_groups_12846)) {
            *(volatile __local int32_t *) &mem_13373[squot32(local_tid_12907,
                                                             32) *
                                                     sizeof(int32_t)] = x_12904;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_tid_12907, 32) == 0 && slt32(local_tid_12907,
                                                       num_groups_12846)) {
            x_13565 = *(volatile __local int32_t *) &mem_13373[local_tid_12907 *
                                                               sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            int32_t skip_threads_13568 = 1;
            
            while (slt32(skip_threads_13568, 32)) {
                if ((squot32(local_tid_12907, 32) == 0 && slt32(local_tid_12907,
                                                                num_groups_12846)) &&
                    sle32(skip_threads_13568, local_tid_12907 -
                          squot32(local_tid_12907, 32) * 32)) {
                    // read operands
                    {
                        x_13564 = *(volatile __local
                                    int32_t *) &mem_13373[(local_tid_12907 -
                                                           skip_threads_13568) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        int32_t res_13566;
                        
                        if (thread_active_13559) {
                            res_13566 = x_13564 + x_13565;
                        }
                        x_13565 = res_13566;
                    }
                }
                if (sle32(wave_sizze_13557, skip_threads_13568)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if ((squot32(local_tid_12907, 32) == 0 && slt32(local_tid_12907,
                                                                num_groups_12846)) &&
                    sle32(skip_threads_13568, local_tid_12907 -
                          squot32(local_tid_12907, 32) * 32)) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &mem_13373[local_tid_12907 *
                                                sizeof(int32_t)] = x_13565;
                    }
                }
                if (sle32(wave_sizze_13557, skip_threads_13568)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_13568 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_12907, 32) == 0 || !slt32(local_tid_12907,
                                                          num_groups_12846))) {
            // read operands
            {
                x_12903 = *(volatile __local
                            int32_t *) &mem_13373[(squot32(local_tid_12907,
                                                           32) - 1) *
                                                  sizeof(int32_t)];
            }
            // perform operation
            {
                int32_t res_12905;
                
                if (thread_active_13559) {
                    res_12905 = x_12903 + x_12904;
                }
                x_12904 = res_12905;
            }
            // write final result
            {
                *(volatile __local int32_t *) &mem_13373[local_tid_12907 *
                                                         sizeof(int32_t)] =
                    x_12904;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_12907, 32) == 0) {
            *(volatile __local int32_t *) &mem_13373[local_tid_12907 *
                                                     sizeof(int32_t)] = x_12904;
        }
    }
    
    int32_t scanned_elem_12913;
    
    if (thread_active_13559) {
        scanned_elem_12913 = *(__local int32_t *) &mem_13373[local_tid_12907 *
                                                             4];
    }
    *(__global int32_t *) &mem_13376[global_tid_12906 * 4] = scanned_elem_12913;
}
__kernel void scan2_kernel_13015(__local volatile int64_t *mem_aligned_0,
                                 int32_t num_groups_12946, __global
                                 unsigned char *mem_13401, __global
                                 unsigned char *mem_13407)
{
    __local volatile char *restrict mem_13404 = mem_aligned_0;
    int32_t wave_sizze_13588;
    int32_t group_sizze_13589;
    bool thread_active_13590;
    int32_t global_tid_13015;
    int32_t local_tid_13016;
    int32_t group_id_13017;
    
    global_tid_13015 = get_global_id(0);
    local_tid_13016 = get_local_id(0);
    group_sizze_13589 = get_local_size(0);
    wave_sizze_13588 = LOCKSTEP_WIDTH;
    group_id_13017 = get_group_id(0);
    thread_active_13590 = 1;
    for (int32_t comb_iter_13591 = 0; comb_iter_13591 <
         squot32(num_groups_12946 + num_groups_12946 - 1, num_groups_12946);
         comb_iter_13591++) {
        int32_t combine_id_13018;
        int32_t flat_comb_id_13592 = comb_iter_13591 * num_groups_12946 +
                local_tid_13016;
        
        combine_id_13018 = flat_comb_id_13592;
        if (slt32(combine_id_13018, num_groups_12946) && 1) {
            int32_t offsets_scan_carry_out_elem_13019 = *(__global
                                                          int32_t *) &mem_13401[combine_id_13018 *
                                                                                4];
            
            *(__local int32_t *) &mem_13404[combine_id_13018 * 4] =
                offsets_scan_carry_out_elem_13019;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t my_index_13010;
    int32_t other_index_13011;
    int32_t x_13012;
    int32_t y_13013;
    int32_t my_index_13593;
    int32_t other_index_13594;
    int32_t x_13595;
    int32_t y_13596;
    
    my_index_13010 = local_tid_13016;
    if (slt32(local_tid_13016, num_groups_12946)) {
        y_13013 = *(volatile __local int32_t *) &mem_13404[local_tid_13016 *
                                                           sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        int32_t skip_threads_13598 = 1;
        
        while (slt32(skip_threads_13598, 32)) {
            if (slt32(local_tid_13016, num_groups_12946) &&
                sle32(skip_threads_13598, local_tid_13016 -
                      squot32(local_tid_13016, 32) * 32)) {
                // read operands
                {
                    x_13012 = *(volatile __local
                                int32_t *) &mem_13404[(local_tid_13016 -
                                                       skip_threads_13598) *
                                                      sizeof(int32_t)];
                }
                // perform operation
                {
                    int32_t zz_13014;
                    
                    if (thread_active_13590) {
                        zz_13014 = x_13012 + y_13013;
                    }
                    y_13013 = zz_13014;
                }
            }
            if (sle32(wave_sizze_13588, skip_threads_13598)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (slt32(local_tid_13016, num_groups_12946) &&
                sle32(skip_threads_13598, local_tid_13016 -
                      squot32(local_tid_13016, 32) * 32)) {
                // write result
                {
                    *(volatile __local int32_t *) &mem_13404[local_tid_13016 *
                                                             sizeof(int32_t)] =
                        y_13013;
                }
            }
            if (sle32(wave_sizze_13588, skip_threads_13598)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_13598 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_13016 - squot32(local_tid_13016, 32) * 32) == 31 &&
            slt32(local_tid_13016, num_groups_12946)) {
            *(volatile __local int32_t *) &mem_13404[squot32(local_tid_13016,
                                                             32) *
                                                     sizeof(int32_t)] = y_13013;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_tid_13016, 32) == 0 && slt32(local_tid_13016,
                                                       num_groups_12946)) {
            y_13596 = *(volatile __local int32_t *) &mem_13404[local_tid_13016 *
                                                               sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            int32_t skip_threads_13599 = 1;
            
            while (slt32(skip_threads_13599, 32)) {
                if ((squot32(local_tid_13016, 32) == 0 && slt32(local_tid_13016,
                                                                num_groups_12946)) &&
                    sle32(skip_threads_13599, local_tid_13016 -
                          squot32(local_tid_13016, 32) * 32)) {
                    // read operands
                    {
                        x_13595 = *(volatile __local
                                    int32_t *) &mem_13404[(local_tid_13016 -
                                                           skip_threads_13599) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        int32_t zz_13597;
                        
                        if (thread_active_13590) {
                            zz_13597 = x_13595 + y_13596;
                        }
                        y_13596 = zz_13597;
                    }
                }
                if (sle32(wave_sizze_13588, skip_threads_13599)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if ((squot32(local_tid_13016, 32) == 0 && slt32(local_tid_13016,
                                                                num_groups_12946)) &&
                    sle32(skip_threads_13599, local_tid_13016 -
                          squot32(local_tid_13016, 32) * 32)) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &mem_13404[local_tid_13016 *
                                                sizeof(int32_t)] = y_13596;
                    }
                }
                if (sle32(wave_sizze_13588, skip_threads_13599)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_13599 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_13016, 32) == 0 || !slt32(local_tid_13016,
                                                          num_groups_12946))) {
            // read operands
            {
                x_13012 = *(volatile __local
                            int32_t *) &mem_13404[(squot32(local_tid_13016,
                                                           32) - 1) *
                                                  sizeof(int32_t)];
            }
            // perform operation
            {
                int32_t zz_13014;
                
                if (thread_active_13590) {
                    zz_13014 = x_13012 + y_13013;
                }
                y_13013 = zz_13014;
            }
            // write final result
            {
                *(volatile __local int32_t *) &mem_13404[local_tid_13016 *
                                                         sizeof(int32_t)] =
                    y_13013;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_13016, 32) == 0) {
            *(volatile __local int32_t *) &mem_13404[local_tid_13016 *
                                                     sizeof(int32_t)] = y_13013;
        }
    }
    
    int32_t scanned_elem_13022;
    
    if (thread_active_13590) {
        scanned_elem_13022 = *(__local int32_t *) &mem_13404[local_tid_13016 *
                                                             4];
    }
    *(__global int32_t *) &mem_13407[global_tid_13015 * 4] = scanned_elem_13022;
}
__kernel void scan2_kernel_13172(__local volatile int64_t *mem_aligned_0,
                                 __local volatile int64_t *mem_aligned_1,
                                 int32_t num_groups_13075, __global
                                 unsigned char *mem_13444, __global
                                 unsigned char *mem_13447, __global
                                 unsigned char *mem_13456, __global
                                 unsigned char *mem_13459)
{
    __local volatile char *restrict mem_13450 = mem_aligned_0;
    __local volatile char *restrict mem_13453 = mem_aligned_1;
    int32_t wave_sizze_13636;
    int32_t group_sizze_13637;
    bool thread_active_13638;
    int32_t global_tid_13172;
    int32_t local_tid_13173;
    int32_t group_id_13174;
    
    global_tid_13172 = get_global_id(0);
    local_tid_13173 = get_local_id(0);
    group_sizze_13637 = get_local_size(0);
    wave_sizze_13636 = LOCKSTEP_WIDTH;
    group_id_13174 = get_group_id(0);
    thread_active_13638 = 1;
    for (int32_t comb_iter_13639 = 0; comb_iter_13639 <
         squot32(num_groups_13075 + num_groups_13075 - 1, num_groups_13075);
         comb_iter_13639++) {
        int32_t combine_id_13175;
        int32_t flat_comb_id_13640 = comb_iter_13639 * num_groups_13075 +
                local_tid_13173;
        
        combine_id_13175 = flat_comb_id_13640;
        if (slt32(combine_id_13175, num_groups_13075) && 1) {
            int32_t res_scan_carry_out_elem_13176 = *(__global
                                                      int32_t *) &mem_13444[combine_id_13175 *
                                                                            4];
            float res_scan_carry_out_elem_13177 = *(__global
                                                    float *) &mem_13447[combine_id_13175 *
                                                                        4];
            
            *(__local int32_t *) &mem_13450[combine_id_13175 * 4] =
                res_scan_carry_out_elem_13176;
            *(__local float *) &mem_13453[combine_id_13175 * 4] =
                res_scan_carry_out_elem_13177;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t my_index_13162;
    int32_t other_index_13163;
    int32_t x_13164;
    float x_13165;
    int32_t x_13166;
    float x_13167;
    int32_t my_index_13641;
    int32_t other_index_13642;
    int32_t x_13643;
    float x_13644;
    int32_t x_13645;
    float x_13646;
    
    my_index_13162 = local_tid_13173;
    if (slt32(local_tid_13173, num_groups_13075)) {
        x_13166 = *(volatile __local int32_t *) &mem_13450[local_tid_13173 *
                                                           sizeof(int32_t)];
        x_13167 = *(volatile __local float *) &mem_13453[local_tid_13173 *
                                                         sizeof(float)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        int32_t skip_threads_13651 = 1;
        
        while (slt32(skip_threads_13651, 32)) {
            if (slt32(local_tid_13173, num_groups_13075) &&
                sle32(skip_threads_13651, local_tid_13173 -
                      squot32(local_tid_13173, 32) * 32)) {
                // read operands
                {
                    x_13164 = *(volatile __local
                                int32_t *) &mem_13450[(local_tid_13173 -
                                                       skip_threads_13651) *
                                                      sizeof(int32_t)];
                    x_13165 = *(volatile __local
                                float *) &mem_13453[(local_tid_13173 -
                                                     skip_threads_13651) *
                                                    sizeof(float)];
                }
                // perform operation
                {
                    int32_t res_13168;
                    bool cond_13169;
                    float res_13170;
                    
                    if (thread_active_13638) {
                        res_13168 = x_13164 | x_13166;
                        cond_13169 = slt32(0, x_13166);
                        if (cond_13169) {
                            res_13170 = x_13167;
                        } else {
                            float res_13171 = x_13165 + x_13167;
                            
                            res_13170 = res_13171;
                        }
                    }
                    x_13166 = res_13168;
                    x_13167 = res_13170;
                }
            }
            if (sle32(wave_sizze_13636, skip_threads_13651)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (slt32(local_tid_13173, num_groups_13075) &&
                sle32(skip_threads_13651, local_tid_13173 -
                      squot32(local_tid_13173, 32) * 32)) {
                // write result
                {
                    *(volatile __local int32_t *) &mem_13450[local_tid_13173 *
                                                             sizeof(int32_t)] =
                        x_13166;
                    *(volatile __local float *) &mem_13453[local_tid_13173 *
                                                           sizeof(float)] =
                        x_13167;
                }
            }
            if (sle32(wave_sizze_13636, skip_threads_13651)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_13651 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_13173 - squot32(local_tid_13173, 32) * 32) == 31 &&
            slt32(local_tid_13173, num_groups_13075)) {
            *(volatile __local int32_t *) &mem_13450[squot32(local_tid_13173,
                                                             32) *
                                                     sizeof(int32_t)] = x_13166;
            *(volatile __local float *) &mem_13453[squot32(local_tid_13173,
                                                           32) *
                                                   sizeof(float)] = x_13167;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_tid_13173, 32) == 0 && slt32(local_tid_13173,
                                                       num_groups_13075)) {
            x_13645 = *(volatile __local int32_t *) &mem_13450[local_tid_13173 *
                                                               sizeof(int32_t)];
            x_13646 = *(volatile __local float *) &mem_13453[local_tid_13173 *
                                                             sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            int32_t skip_threads_13652 = 1;
            
            while (slt32(skip_threads_13652, 32)) {
                if ((squot32(local_tid_13173, 32) == 0 && slt32(local_tid_13173,
                                                                num_groups_13075)) &&
                    sle32(skip_threads_13652, local_tid_13173 -
                          squot32(local_tid_13173, 32) * 32)) {
                    // read operands
                    {
                        x_13643 = *(volatile __local
                                    int32_t *) &mem_13450[(local_tid_13173 -
                                                           skip_threads_13652) *
                                                          sizeof(int32_t)];
                        x_13644 = *(volatile __local
                                    float *) &mem_13453[(local_tid_13173 -
                                                         skip_threads_13652) *
                                                        sizeof(float)];
                    }
                    // perform operation
                    {
                        int32_t res_13647;
                        bool cond_13648;
                        float res_13649;
                        
                        if (thread_active_13638) {
                            res_13647 = x_13643 | x_13645;
                            cond_13648 = slt32(0, x_13645);
                            if (cond_13648) {
                                res_13649 = x_13646;
                            } else {
                                float res_13650 = x_13644 + x_13646;
                                
                                res_13649 = res_13650;
                            }
                        }
                        x_13645 = res_13647;
                        x_13646 = res_13649;
                    }
                }
                if (sle32(wave_sizze_13636, skip_threads_13652)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if ((squot32(local_tid_13173, 32) == 0 && slt32(local_tid_13173,
                                                                num_groups_13075)) &&
                    sle32(skip_threads_13652, local_tid_13173 -
                          squot32(local_tid_13173, 32) * 32)) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &mem_13450[local_tid_13173 *
                                                sizeof(int32_t)] = x_13645;
                        *(volatile __local float *) &mem_13453[local_tid_13173 *
                                                               sizeof(float)] =
                            x_13646;
                    }
                }
                if (sle32(wave_sizze_13636, skip_threads_13652)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_13652 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_13173, 32) == 0 || !slt32(local_tid_13173,
                                                          num_groups_13075))) {
            // read operands
            {
                x_13164 = *(volatile __local
                            int32_t *) &mem_13450[(squot32(local_tid_13173,
                                                           32) - 1) *
                                                  sizeof(int32_t)];
                x_13165 = *(volatile __local
                            float *) &mem_13453[(squot32(local_tid_13173, 32) -
                                                 1) * sizeof(float)];
            }
            // perform operation
            {
                int32_t res_13168;
                bool cond_13169;
                float res_13170;
                
                if (thread_active_13638) {
                    res_13168 = x_13164 | x_13166;
                    cond_13169 = slt32(0, x_13166);
                    if (cond_13169) {
                        res_13170 = x_13167;
                    } else {
                        float res_13171 = x_13165 + x_13167;
                        
                        res_13170 = res_13171;
                    }
                }
                x_13166 = res_13168;
                x_13167 = res_13170;
            }
            // write final result
            {
                *(volatile __local int32_t *) &mem_13450[local_tid_13173 *
                                                         sizeof(int32_t)] =
                    x_13166;
                *(volatile __local float *) &mem_13453[local_tid_13173 *
                                                       sizeof(float)] = x_13167;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_13173, 32) == 0) {
            *(volatile __local int32_t *) &mem_13450[local_tid_13173 *
                                                     sizeof(int32_t)] = x_13166;
            *(volatile __local float *) &mem_13453[local_tid_13173 *
                                                   sizeof(float)] = x_13167;
        }
    }
    
    int32_t scanned_elem_13182;
    float scanned_elem_13183;
    
    if (thread_active_13638) {
        scanned_elem_13182 = *(__local int32_t *) &mem_13450[local_tid_13173 *
                                                             4];
        scanned_elem_13183 = *(__local float *) &mem_13453[local_tid_13173 * 4];
    }
    *(__global int32_t *) &mem_13456[global_tid_13172 * 4] = scanned_elem_13182;
    *(__global float *) &mem_13459[global_tid_13172 * 4] = scanned_elem_13183;
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
class forwardprojection_semiflat:
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
                                       all_sizes={"group_size_12473": {"class": "group_size", "value": None},
                                        "group_size_12489": {"class": "group_size", "value": None},
                                        "group_size_12501": {"class": "group_size", "value": None},
                                        "group_size_12514": {"class": "group_size", "value": None},
                                        "group_size_12567": {"class": "group_size", "value": None},
                                        "group_size_12822": {"class": "group_size", "value": None},
                                        "group_size_12838": {"class": "group_size", "value": None},
                                        "max_num_groups_12840": {"class": "num_groups", "value": None},
                                        "group_size_12929": {"class": "group_size", "value": None},
                                        "group_size_12938": {"class": "group_size", "value": None},
                                        "max_num_groups_12940": {"class": "num_groups", "value": None},
                                        "group_size_13038": {"class": "group_size", "value": None},
                                        "group_size_13048": {"class": "group_size", "value": None},
                                        "group_size_13058": {"class": "group_size", "value": None},
                                        "group_size_13067": {"class": "group_size", "value": None},
                                        "max_num_groups_13069": {"class": "num_groups", "value": None},
                                        "group_size_13208": {"class": "group_size", "value": None},
                                        "group_size_13218": {"class": "group_size", "value": None},
                                        "group_size_13515": {"class": "group_size", "value": None},
                                        "group_size_13608": {"class": "group_size", "value": None}})
    self.fut_kernel_map_transpose_f32_var = program.fut_kernel_map_transpose_f32
    self.fut_kernel_map_transpose_i32_var = program.fut_kernel_map_transpose_i32
    self.fut_kernel_map_transpose_lowheight_f32_var = program.fut_kernel_map_transpose_lowheight_f32
    self.fut_kernel_map_transpose_lowheight_i32_var = program.fut_kernel_map_transpose_lowheight_i32
    self.fut_kernel_map_transpose_lowwidth_f32_var = program.fut_kernel_map_transpose_lowwidth_f32
    self.fut_kernel_map_transpose_lowwidth_i32_var = program.fut_kernel_map_transpose_lowwidth_i32
    self.fut_kernel_map_transpose_small_f32_var = program.fut_kernel_map_transpose_small_f32
    self.fut_kernel_map_transpose_small_i32_var = program.fut_kernel_map_transpose_small_i32
    self.kernel_replicate_11891_var = program.kernel_replicate_11891
    self.kernel_replicate_12419_var = program.kernel_replicate_12419
    self.map_kernel_12479_var = program.map_kernel_12479
    self.map_kernel_12495_var = program.map_kernel_12495
    self.map_kernel_12507_var = program.map_kernel_12507
    self.map_kernel_12520_var = program.map_kernel_12520
    self.map_kernel_12573_var = program.map_kernel_12573
    self.map_kernel_12828_var = program.map_kernel_12828
    self.map_kernel_12935_var = program.map_kernel_12935
    self.map_kernel_13044_var = program.map_kernel_13044
    self.map_kernel_13054_var = program.map_kernel_13054
    self.map_kernel_13064_var = program.map_kernel_13064
    self.map_kernel_13214_var = program.map_kernel_13214
    self.map_kernel_13224_var = program.map_kernel_13224
    self.scan1_kernel_12870_var = program.scan1_kernel_12870
    self.scan1_kernel_12976_var = program.scan1_kernel_12976
    self.scan1_kernel_13117_var = program.scan1_kernel_13117
    self.scan2_kernel_12906_var = program.scan2_kernel_12906
    self.scan2_kernel_13015_var = program.scan2_kernel_13015
    self.scan2_kernel_13172_var = program.scan2_kernel_13172
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
  def futhark_main(self, angles_mem_sizze_13286, angles_mem_13287,
                   rays_mem_sizze_13288, rays_mem_13289, voxels_mem_sizze_13290,
                   voxels_mem_13291, sizze_11837, sizze_11838, sizze_11839,
                   stepsizze_11843):
    res_11844 = sitofp_i32_f32(sizze_11839)
    res_11845 = futhark_sqrt32(res_11844)
    res_11846 = fptosi_f32_i32(res_11845)
    res_11847 = sitofp_i32_f32(res_11846)
    res_11848 = (res_11847 / np.float32(2.0))
    group_sizze_12502 = self.sizes["group_size_12501"]
    y_12503 = (group_sizze_12502 - np.int32(1))
    x_12504 = (sizze_11837 + y_12503)
    num_groups_12505 = squot32(x_12504, group_sizze_12502)
    num_threads_12506 = (group_sizze_12502 * num_groups_12505)
    binop_x_13293 = sext_i32_i64(sizze_11837)
    bytes_13292 = (np.int64(4) * binop_x_13293)
    mem_13294 = opencl_alloc(self, bytes_13292, "mem_13294")
    mem_13297 = opencl_alloc(self, bytes_13292, "mem_13297")
    if ((1 * (num_groups_12505 * group_sizze_12502)) != 0):
      self.map_kernel_12507_var.set_args(np.int32(sizze_11837),
                                         angles_mem_13287, mem_13294, mem_13297)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_12507_var,
                                 (np.long((num_groups_12505 * group_sizze_12502)),),
                                 (np.long(group_sizze_12502),))
      if synchronous:
        self.queue.finish()
    nesting_sizze_12488 = (sizze_11837 * sizze_11838)
    group_sizze_12490 = self.sizes["group_size_12489"]
    y_12491 = (group_sizze_12490 - np.int32(1))
    x_12492 = (nesting_sizze_12488 + y_12491)
    num_groups_12493 = squot32(x_12492, group_sizze_12490)
    num_threads_12494 = (group_sizze_12490 * num_groups_12493)
    binop_x_13300 = sext_i32_i64(nesting_sizze_12488)
    bytes_13298 = (np.int64(4) * binop_x_13300)
    mem_13301 = opencl_alloc(self, bytes_13298, "mem_13301")
    if ((1 * (num_groups_12493 * group_sizze_12490)) != 0):
      self.map_kernel_12495_var.set_args(np.int32(sizze_11837),
                                         np.int32(sizze_11838), mem_13294,
                                         mem_13301)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_12495_var,
                                 (np.long((num_groups_12493 * group_sizze_12490)),),
                                 (np.long(group_sizze_12490),))
      if synchronous:
        self.queue.finish()
    mem_13294 = None
    group_sizze_12474 = self.sizes["group_size_12473"]
    y_12475 = (group_sizze_12474 - np.int32(1))
    x_12476 = (y_12475 + nesting_sizze_12488)
    num_groups_12477 = squot32(x_12476, group_sizze_12474)
    num_threads_12478 = (group_sizze_12474 * num_groups_12477)
    mem_13305 = opencl_alloc(self, bytes_13298, "mem_13305")
    if ((1 * (num_groups_12477 * group_sizze_12474)) != 0):
      self.map_kernel_12479_var.set_args(np.int32(sizze_11837),
                                         np.int32(sizze_11838), mem_13297,
                                         mem_13305)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_12479_var,
                                 (np.long((num_groups_12477 * group_sizze_12474)),),
                                 (np.long(group_sizze_12474),))
      if synchronous:
        self.queue.finish()
    mem_13297 = None
    res_11861 = (np.float32(0.0) - res_11848)
    group_sizze_12515 = self.sizes["group_size_12514"]
    y_12516 = (group_sizze_12515 - np.int32(1))
    x_12517 = (nesting_sizze_12488 + y_12516)
    num_groups_12518 = squot32(x_12517, group_sizze_12515)
    num_threads_12519 = (group_sizze_12515 * num_groups_12518)
    mem_13308 = opencl_alloc(self, bytes_13298, "mem_13308")
    mem_13311 = opencl_alloc(self, bytes_13298, "mem_13311")
    if ((1 * (num_groups_12518 * group_sizze_12515)) != 0):
      self.map_kernel_12520_var.set_args(np.int32(sizze_11838),
                                         np.float32(res_11848),
                                         np.float32(res_11861),
                                         np.int32(nesting_sizze_12488),
                                         rays_mem_13289, mem_13301, mem_13305,
                                         mem_13308, mem_13311)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_12520_var,
                                 (np.long((num_groups_12518 * group_sizze_12515)),),
                                 (np.long(group_sizze_12515),))
      if synchronous:
        self.queue.finish()
    res_11890 = sdiv32(nesting_sizze_12488, stepsizze_11843)
    mem_13314 = opencl_alloc(self, np.int64(4), "mem_13314")
    group_sizze_13515 = self.sizes["group_size_13515"]
    num_groups_13516 = squot32(((np.int32(1) + sext_i32_i32(group_sizze_13515)) - np.int32(1)),
                               sext_i32_i32(group_sizze_13515))
    if ((1 * (num_groups_13516 * group_sizze_13515)) != 0):
      self.kernel_replicate_11891_var.set_args(mem_13314)
      cl.enqueue_nd_range_kernel(self.queue, self.kernel_replicate_11891_var,
                                 (np.long((num_groups_13516 * group_sizze_13515)),),
                                 (np.long(group_sizze_13515),))
      if synchronous:
        self.queue.finish()
    loop_cond_11892 = slt32(np.int32(0), res_11890)
    x_11893 = (np.float32(2.0) * res_11847)
    arg_11894 = (x_11893 - np.float32(1.0))
    res_11895 = fptosi_f32_i32(arg_11894)
    group_sizze_12823 = self.sizes["group_size_12822"]
    y_12824 = (group_sizze_12823 - np.int32(1))
    group_sizze_12568 = self.sizes["group_size_12567"]
    y_12569 = (group_sizze_12568 - np.int32(1))
    group_sizze_12839 = self.sizes["group_size_12838"]
    max_num_groups_12841 = self.sizes["max_num_groups_12840"]
    y_12842 = (group_sizze_12839 - np.int32(1))
    group_sizze_12930 = self.sizes["group_size_12929"]
    y_12931 = (group_sizze_12930 - np.int32(1))
    group_sizze_12939 = self.sizes["group_size_12938"]
    max_num_groups_12941 = self.sizes["max_num_groups_12940"]
    y_12942 = (group_sizze_12939 - np.int32(1))
    group_sizze_13039 = self.sizes["group_size_13038"]
    y_13040 = (group_sizze_13039 - np.int32(1))
    group_sizze_13049 = self.sizes["group_size_13048"]
    y_13050 = (group_sizze_13049 - np.int32(1))
    group_sizze_13059 = self.sizes["group_size_13058"]
    y_13060 = (group_sizze_13059 - np.int32(1))
    group_sizze_13068 = self.sizes["group_size_13067"]
    max_num_groups_13070 = self.sizes["max_num_groups_13069"]
    y_13071 = (group_sizze_13068 - np.int32(1))
    group_sizze_13209 = self.sizes["group_size_13208"]
    y_13210 = (group_sizze_13209 - np.int32(1))
    group_sizze_13219 = self.sizes["group_size_13218"]
    y_13220 = (group_sizze_13219 - np.int32(1))
    binop_x_13325 = sext_i32_i64(res_11895)
    bytes_13324 = (np.int64(4) * binop_x_13325)
    binop_x_13363 = sext_i32_i64(group_sizze_12839)
    bytes_13362 = (np.int64(4) * binop_x_13363)
    binop_x_13393 = sext_i32_i64(group_sizze_12939)
    bytes_13392 = (np.int64(4) * binop_x_13393)
    binop_x_13431 = sext_i32_i64(group_sizze_13068)
    bytes_13430 = (np.int64(4) * binop_x_13431)
    sizze_11909 = np.int32(1)
    output_mem_sizze_13315 = np.int64(4)
    output_mem_13316 = mem_13314
    loop_while_11910 = loop_cond_11892
    run_11912 = np.int32(0)
    while loop_while_11910:
      x_11913 = (np.int32(1) + run_11912)
      x_11914 = (stepsizze_11843 * x_11913)
      cond_11915 = sle32(nesting_sizze_12488, x_11914)
      if cond_11915:
        y_11917 = (stepsizze_11843 * run_11912)
        res_11918 = (nesting_sizze_12488 - y_11917)
        res_11916 = res_11918
      else:
        res_11916 = stepsizze_11843
      i_11919 = (stepsizze_11843 * run_11912)
      j_11920 = (res_11916 + i_11919)
      x_11921 = abs(res_11916)
      empty_slice_11922 = (x_11921 == np.int32(0))
      m_11923 = (x_11921 - np.int32(1))
      i_p_m_t_s_11924 = (i_11919 + m_11923)
      zzero_leq_i_p_m_t_s_11925 = sle32(np.int32(0), i_p_m_t_s_11924)
      i_p_m_t_s_leq_w_11926 = slt32(i_p_m_t_s_11924, nesting_sizze_12488)
      zzero_lte_i_11927 = sle32(np.int32(0), i_11919)
      i_lte_j_11928 = sle32(i_11919, j_11920)
      y_11929 = (i_p_m_t_s_leq_w_11926 and zzero_lte_i_11927)
      y_11930 = (zzero_leq_i_p_m_t_s_11925 and y_11929)
      y_11931 = (i_lte_j_11928 and y_11930)
      forwards_ok_11932 = (zzero_lte_i_11927 and y_11931)
      ok_or_empty_11933 = (empty_slice_11922 or forwards_ok_11932)
      index_certs_11934 = True
      assert ok_or_empty_11933, ("Error at forwardprojection_semiflat.fut:32:1-36:64 -> forwardprojection_semiflat.fut:36:11-36:64 -> projection_lib.fut:306:92-306:142: %s%d%s%d%s%d%s" % ("Index [",
                                                                                                                                                                                            i_11919,
                                                                                                                                                                                            ":",
                                                                                                                                                                                            j_11920,
                                                                                                                                                                                            "] out of bounds for array of shape [",
                                                                                                                                                                                            nesting_sizze_12488,
                                                                                                                                                                                            "]."))
      x_12825 = (x_11921 + y_12824)
      num_groups_12826 = squot32(x_12825, group_sizze_12823)
      num_threads_12827 = (group_sizze_12823 * num_groups_12826)
      bytes_13317 = sext_i32_i64(x_11921)
      mem_13318 = opencl_alloc(self, bytes_13317, "mem_13318")
      mem_13320 = opencl_alloc(self, bytes_13317, "mem_13320")
      bytes_13321 = (np.int64(4) * bytes_13317)
      mem_13323 = opencl_alloc(self, bytes_13321, "mem_13323")
      if ((1 * (num_groups_12826 * group_sizze_12823)) != 0):
        self.map_kernel_12828_var.set_args(np.int32(sizze_11838),
                                           np.int32(i_11919), np.int32(x_11921),
                                           mem_13301, mem_13305, mem_13318,
                                           mem_13320, mem_13323)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_12828_var,
                                   (np.long((num_groups_12826 * group_sizze_12823)),),
                                   (np.long(group_sizze_12823),))
        if synchronous:
          self.queue.finish()
      x_12570 = (x_11921 + y_12569)
      num_groups_12571 = squot32(x_12570, group_sizze_12568)
      num_threads_12572 = (group_sizze_12568 * num_groups_12571)
      mem_13350 = opencl_alloc(self, bytes_13321, "mem_13350")
      convop_x_13352 = (res_11895 * x_11921)
      binop_x_13353 = sext_i32_i64(convop_x_13352)
      bytes_13351 = (np.int64(4) * binop_x_13353)
      mem_13354 = opencl_alloc(self, bytes_13351, "mem_13354")
      mem_13358 = opencl_alloc(self, bytes_13351, "mem_13358")
      num_threads64_13483 = sext_i32_i64(num_threads_12572)
      total_sizze_13484 = (bytes_13324 * num_threads64_13483)
      mem_13326 = opencl_alloc(self, total_sizze_13484, "mem_13326")
      total_sizze_13485 = (bytes_13324 * num_threads64_13483)
      mem_13329 = opencl_alloc(self, total_sizze_13485, "mem_13329")
      if ((1 * (num_groups_12571 * group_sizze_12568)) != 0):
        self.map_kernel_12573_var.set_args(np.float32(res_11847),
                                           np.float32(res_11848),
                                           np.float32(res_11861),
                                           np.int32(res_11895),
                                           np.int32(i_11919), np.int32(x_11921),
                                           mem_13308, mem_13311, mem_13318,
                                           mem_13320, mem_13323, mem_13326,
                                           mem_13329, mem_13350, mem_13354,
                                           mem_13358)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_12573_var,
                                   (np.long((num_groups_12571 * group_sizze_12568)),),
                                   (np.long(group_sizze_12568),))
        if synchronous:
          self.queue.finish()
      mem_13318 = None
      mem_13320 = None
      mem_13323 = None
      mem_13326 = None
      mem_13329 = None
      x_12843 = (x_11921 + y_12842)
      w_div_group_sizze_12844 = squot32(x_12843, group_sizze_12839)
      num_groups_maybe_zzero_12845 = smin32(max_num_groups_12841,
                                            w_div_group_sizze_12844)
      num_groups_12846 = smax32(np.int32(1), num_groups_maybe_zzero_12845)
      num_threads_12847 = (group_sizze_12839 * num_groups_12846)
      mem_13361 = opencl_alloc(self, bytes_13321, "mem_13361")
      y_12873 = (num_threads_12847 - np.int32(1))
      x_12874 = (x_11921 + y_12873)
      num_iterations_12875 = squot32(x_12874, num_threads_12847)
      y_12878 = (group_sizze_12839 * num_iterations_12875)
      binop_x_13369 = sext_i32_i64(num_groups_12846)
      bytes_13368 = (np.int64(4) * binop_x_13369)
      mem_13370 = opencl_alloc(self, bytes_13368, "mem_13370")
      if ((1 * (num_groups_12846 * group_sizze_12839)) != 0):
        self.scan1_kernel_12870_var.set_args(cl.LocalMemory(np.long(bytes_13362)),
                                             np.int32(x_11921),
                                             np.int32(num_iterations_12875),
                                             np.int32(y_12878), mem_13350,
                                             mem_13361, mem_13370)
        cl.enqueue_nd_range_kernel(self.queue, self.scan1_kernel_12870_var,
                                   (np.long((num_groups_12846 * group_sizze_12839)),),
                                   (np.long(group_sizze_12839),))
        if synchronous:
          self.queue.finish()
      mem_13350 = None
      mem_13376 = opencl_alloc(self, bytes_13368, "mem_13376")
      if ((1 * num_groups_12846) != 0):
        self.scan2_kernel_12906_var.set_args(cl.LocalMemory(np.long(bytes_13368)),
                                             np.int32(num_groups_12846),
                                             mem_13370, mem_13376)
        cl.enqueue_nd_range_kernel(self.queue, self.scan2_kernel_12906_var,
                                   (np.long(num_groups_12846),),
                                   (np.long(num_groups_12846),))
        if synchronous:
          self.queue.finish()
      mem_13370 = None
      mem_13373 = None
      x_12932 = (x_11921 + y_12931)
      num_groups_12933 = squot32(x_12932, group_sizze_12930)
      num_threads_12934 = (group_sizze_12930 * num_groups_12933)
      mem_13379 = opencl_alloc(self, bytes_13321, "mem_13379")
      if ((1 * (num_groups_12933 * group_sizze_12930)) != 0):
        self.map_kernel_12935_var.set_args(np.int32(x_11921), np.int32(y_12878),
                                           mem_13361, mem_13376, mem_13379)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_12935_var,
                                   (np.long((num_groups_12933 * group_sizze_12930)),),
                                   (np.long(group_sizze_12930),))
        if synchronous:
          self.queue.finish()
      mem_13361 = None
      mem_13376 = None
      x_12943 = (y_12942 + convop_x_13352)
      w_div_group_sizze_12944 = squot32(x_12943, group_sizze_12939)
      num_groups_maybe_zzero_12945 = smin32(max_num_groups_12941,
                                            w_div_group_sizze_12944)
      num_groups_12946 = smax32(np.int32(1), num_groups_maybe_zzero_12945)
      num_threads_12947 = (group_sizze_12939 * num_groups_12946)
      mem_13382 = opencl_alloc(self, bytes_13351, "mem_13382")
      mem_13385 = opencl_alloc(self, bytes_13351, "mem_13385")
      y_12979 = (num_threads_12947 - np.int32(1))
      x_12980 = (y_12979 + convop_x_13352)
      num_iterations_12981 = squot32(x_12980, num_threads_12947)
      y_12984 = (group_sizze_12939 * num_iterations_12981)
      mem_13389 = opencl_alloc(self, bytes_13351, "mem_13389")
      self.futhark_map_transpose_opencl_i32(mem_13389, np.int32(0), mem_13358,
                                            np.int32(0), np.int32(1), x_11921,
                                            res_11895, (x_11921 * res_11895),
                                            (x_11921 * res_11895))
      mem_13358 = None
      binop_x_13400 = sext_i32_i64(num_groups_12946)
      bytes_13399 = (np.int64(4) * binop_x_13400)
      mem_13401 = opencl_alloc(self, bytes_13399, "mem_13401")
      if ((1 * (num_groups_12946 * group_sizze_12939)) != 0):
        self.scan1_kernel_12976_var.set_args(cl.LocalMemory(np.long(bytes_13392)),
                                             np.int32(res_11895),
                                             np.int32(num_iterations_12981),
                                             np.int32(y_12984),
                                             np.int32(convop_x_13352),
                                             mem_13382, mem_13385, mem_13389,
                                             mem_13401)
        cl.enqueue_nd_range_kernel(self.queue, self.scan1_kernel_12976_var,
                                   (np.long((num_groups_12946 * group_sizze_12939)),),
                                   (np.long(group_sizze_12939),))
        if synchronous:
          self.queue.finish()
      mem_13407 = opencl_alloc(self, bytes_13399, "mem_13407")
      if ((1 * num_groups_12946) != 0):
        self.scan2_kernel_13015_var.set_args(cl.LocalMemory(np.long(bytes_13399)),
                                             np.int32(num_groups_12946),
                                             mem_13401, mem_13407)
        cl.enqueue_nd_range_kernel(self.queue, self.scan2_kernel_13015_var,
                                   (np.long(num_groups_12946),),
                                   (np.long(num_groups_12946),))
        if synchronous:
          self.queue.finish()
      mem_13401 = None
      mem_13404 = None
      x_13041 = (y_13040 + convop_x_13352)
      num_groups_13042 = squot32(x_13041, group_sizze_13039)
      num_threads_13043 = (group_sizze_13039 * num_groups_13042)
      mem_13410 = opencl_alloc(self, bytes_13351, "mem_13410")
      if ((1 * (num_groups_13042 * group_sizze_13039)) != 0):
        self.map_kernel_13044_var.set_args(np.int32(y_12984),
                                           np.int32(convop_x_13352), mem_13382,
                                           mem_13407, mem_13410)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_13044_var,
                                   (np.long((num_groups_13042 * group_sizze_13039)),),
                                   (np.long(group_sizze_13039),))
        if synchronous:
          self.queue.finish()
      mem_13382 = None
      mem_13407 = None
      last_index_12382 = (convop_x_13352 - np.int32(1))
      is_empty_12383 = (convop_x_13352 == np.int32(0))
      if is_empty_12383:
        partition_sizze_12384 = np.int32(0)
      else:
        read_res_13660 = np.empty(1, dtype=ct.c_int32)
        cl.enqueue_copy(self.queue, read_res_13660, mem_13410,
                        device_offset=np.long((last_index_12382 * np.int32(4))),
                        is_blocking=True)
        last_offset_12385 = read_res_13660[0]
        partition_sizze_12384 = last_offset_12385
      binop_x_13412 = sext_i32_i64(partition_sizze_12384)
      bytes_13411 = (np.int64(4) * binop_x_13412)
      mem_13413 = opencl_alloc(self, bytes_13411, "mem_13413")
      mem_13416 = opencl_alloc(self, bytes_13411, "mem_13416")
      x_13051 = (y_13050 + convop_x_13352)
      num_groups_13052 = squot32(x_13051, group_sizze_13049)
      num_threads_13053 = (group_sizze_13049 * num_groups_13052)
      mem_13420 = opencl_alloc(self, bytes_13351, "mem_13420")
      self.futhark_map_transpose_opencl_f32(mem_13420, np.int32(0), mem_13354,
                                            np.int32(0), np.int32(1), x_11921,
                                            res_11895, (x_11921 * res_11895),
                                            (x_11921 * res_11895))
      mem_13354 = None
      if ((1 * (num_groups_13052 * group_sizze_13049)) != 0):
        self.map_kernel_13054_var.set_args(np.int32(res_11895),
                                           np.int32(partition_sizze_12384),
                                           np.int32(convop_x_13352), mem_13385,
                                           mem_13389, mem_13410, mem_13413,
                                           mem_13416, mem_13420)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_13054_var,
                                   (np.long((num_groups_13052 * group_sizze_13049)),),
                                   (np.long(group_sizze_13049),))
        if synchronous:
          self.queue.finish()
      mem_13385 = None
      mem_13389 = None
      mem_13410 = None
      mem_13420 = None
      x_12397 = abs(partition_sizze_12384)
      empty_slice_12398 = (x_12397 == np.int32(0))
      m_12399 = (x_12397 - np.int32(1))
      zzero_leq_i_p_m_t_s_12400 = sle32(np.int32(0), m_12399)
      i_p_m_t_s_leq_w_12401 = slt32(m_12399, partition_sizze_12384)
      y_12402 = (zzero_leq_i_p_m_t_s_12400 and i_p_m_t_s_leq_w_12401)
      ok_or_empty_12403 = (empty_slice_12398 or y_12402)
      index_certs_12404 = True
      assert ok_or_empty_12403, ("Error at forwardprojection_semiflat.fut:32:1-36:64 -> forwardprojection_semiflat.fut:36:11-36:64 -> projection_lib.fut:310:37-310:71 -> /futlib/soacs.fut:136:6-136:16: %s%s%s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                                             "",
                                                                                                                                                                                                                             ":",
                                                                                                                                                                                                                             partition_sizze_12384,
                                                                                                                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                                                                                                                             partition_sizze_12384,
                                                                                                                                                                                                                             "]."))
      x_12407 = sle32(np.int32(0), m_11923)
      index_certs_12408 = True
      assert x_12407, ("Error at forwardprojection_semiflat.fut:32:1-36:64 -> forwardprojection_semiflat.fut:36:11-36:64 -> projection_lib.fut:311:41-311:74 -> projection_lib.fut:61:26-61:44: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                               m_11923,
                                                                                                                                                                                                               "] out of bounds for array of shape [",
                                                                                                                                                                                                               x_11921,
                                                                                                                                                                                                               "]."))
      read_res_13661 = np.empty(1, dtype=ct.c_int32)
      cl.enqueue_copy(self.queue, read_res_13661, mem_13379,
                      device_offset=np.long((m_11923 * np.int32(4))),
                      is_blocking=True)
      res_12409 = read_res_13661[0]
      bounds_invalid_upwards_12411 = slt32(res_12409, np.int32(0))
      eq_x_zz_12412 = (np.int32(0) == res_12409)
      not_p_12413 = not(bounds_invalid_upwards_12411)
      p_and_eq_x_y_12414 = (eq_x_zz_12412 and not_p_12413)
      dim_zzero_12415 = (bounds_invalid_upwards_12411 or p_and_eq_x_y_12414)
      both_empty_12416 = (eq_x_zz_12412 and dim_zzero_12415)
      empty_or_match_12417 = (not_p_12413 or both_empty_12416)
      empty_or_match_cert_12418 = True
      assert empty_or_match_12417, ("Error at forwardprojection_semiflat.fut:32:1-36:64 -> forwardprojection_semiflat.fut:36:11-36:64 -> projection_lib.fut:311:41-311:74 -> projection_lib.fut:66:38-66:52 -> /futlib/array.fut:66:1-67:19: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                                                                                              "*",
                                                                                                                                                                                                                                                              "[",
                                                                                                                                                                                                                                                              res_12409,
                                                                                                                                                                                                                                                              "]",
                                                                                                                                                                                                                                                              "t"))
      binop_x_13422 = sext_i32_i64(res_12409)
      bytes_13421 = (np.int64(4) * binop_x_13422)
      mem_13423 = opencl_alloc(self, bytes_13421, "mem_13423")
      group_sizze_13608 = self.sizes["group_size_13608"]
      num_groups_13609 = squot32(((res_12409 + sext_i32_i32(group_sizze_13608)) - np.int32(1)),
                                 sext_i32_i32(group_sizze_13608))
      if ((1 * (num_groups_13609 * group_sizze_13608)) != 0):
        self.kernel_replicate_12419_var.set_args(np.int32(res_12409), mem_13423)
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_replicate_12419_var,
                                   (np.long((num_groups_13609 * group_sizze_13608)),),
                                   (np.long(group_sizze_13608),))
        if synchronous:
          self.queue.finish()
      x_13061 = (x_11921 + y_13060)
      num_groups_13062 = squot32(x_13061, group_sizze_13059)
      num_threads_13063 = (group_sizze_13059 * num_groups_13062)
      if ((1 * (num_groups_13062 * group_sizze_13059)) != 0):
        self.map_kernel_13064_var.set_args(np.int32(x_11921),
                                           np.int32(res_12409), mem_13379,
                                           mem_13423)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_13064_var,
                                   (np.long((num_groups_13062 * group_sizze_13059)),),
                                   (np.long(group_sizze_13059),))
        if synchronous:
          self.queue.finish()
      dim_zzero_12426 = (np.int32(0) == x_12397)
      both_empty_12427 = (eq_x_zz_12412 and dim_zzero_12426)
      dim_match_12428 = (res_12409 == x_12397)
      empty_or_match_12429 = (both_empty_12427 or dim_match_12428)
      empty_or_match_cert_12430 = True
      assert empty_or_match_12429, ("Error at forwardprojection_semiflat.fut:32:1-36:64 -> forwardprojection_semiflat.fut:36:11-36:64 -> projection_lib.fut:311:41-311:74 -> projection_lib.fut:69:27-69:47: %s" % ("function arguments of wrong shape",))
      x_13072 = (res_12409 + y_13071)
      w_div_group_sizze_13073 = squot32(x_13072, group_sizze_13068)
      num_groups_maybe_zzero_13074 = smin32(max_num_groups_13070,
                                            w_div_group_sizze_13073)
      num_groups_13075 = smax32(np.int32(1), num_groups_maybe_zzero_13074)
      num_threads_13076 = (group_sizze_13068 * num_groups_13075)
      mem_13426 = opencl_alloc(self, bytes_13421, "mem_13426")
      mem_13429 = opencl_alloc(self, bytes_13421, "mem_13429")
      y_13120 = (num_threads_13076 - np.int32(1))
      x_13121 = (res_12409 + y_13120)
      num_iterations_13122 = squot32(x_13121, num_threads_13076)
      y_13126 = (group_sizze_13068 * num_iterations_13122)
      binop_x_13443 = sext_i32_i64(num_groups_13075)
      bytes_13442 = (np.int64(4) * binop_x_13443)
      mem_13444 = opencl_alloc(self, bytes_13442, "mem_13444")
      mem_13447 = opencl_alloc(self, bytes_13442, "mem_13447")
      if ((1 * (num_groups_13075 * group_sizze_13068)) != 0):
        self.scan1_kernel_13117_var.set_args(cl.LocalMemory(np.long(bytes_13430)),
                                             cl.LocalMemory(np.long(bytes_13430)),
                                             np.int32(res_12409),
                                             np.int32(num_iterations_13122),
                                             np.int32(y_13126),
                                             voxels_mem_13291, mem_13413,
                                             mem_13416, mem_13423, mem_13426,
                                             mem_13429, mem_13444, mem_13447)
        cl.enqueue_nd_range_kernel(self.queue, self.scan1_kernel_13117_var,
                                   (np.long((num_groups_13075 * group_sizze_13068)),),
                                   (np.long(group_sizze_13068),))
        if synchronous:
          self.queue.finish()
      mem_13413 = None
      mem_13416 = None
      mem_13423 = None
      mem_13456 = opencl_alloc(self, bytes_13442, "mem_13456")
      mem_13459 = opencl_alloc(self, bytes_13442, "mem_13459")
      if ((1 * num_groups_13075) != 0):
        self.scan2_kernel_13172_var.set_args(cl.LocalMemory(np.long(bytes_13442)),
                                             cl.LocalMemory(np.long(bytes_13442)),
                                             np.int32(num_groups_13075),
                                             mem_13444, mem_13447, mem_13456,
                                             mem_13459)
        cl.enqueue_nd_range_kernel(self.queue, self.scan2_kernel_13172_var,
                                   (np.long(num_groups_13075),),
                                   (np.long(num_groups_13075),))
        if synchronous:
          self.queue.finish()
      mem_13444 = None
      mem_13447 = None
      mem_13450 = None
      mem_13453 = None
      x_13211 = (res_12409 + y_13210)
      num_groups_13212 = squot32(x_13211, group_sizze_13209)
      num_threads_13213 = (group_sizze_13209 * num_groups_13212)
      mem_13462 = opencl_alloc(self, bytes_13421, "mem_13462")
      mem_13465 = opencl_alloc(self, bytes_13421, "mem_13465")
      if ((1 * (num_groups_13212 * group_sizze_13209)) != 0):
        self.map_kernel_13214_var.set_args(np.int32(res_12409),
                                           np.int32(y_13126), mem_13426,
                                           mem_13429, mem_13456, mem_13459,
                                           mem_13462, mem_13465)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_13214_var,
                                   (np.long((num_groups_13212 * group_sizze_13209)),),
                                   (np.long(group_sizze_13209),))
        if synchronous:
          self.queue.finish()
      mem_13426 = None
      mem_13429 = None
      mem_13456 = None
      mem_13459 = None
      mem_13462 = None
      x_13221 = (x_11921 + y_13220)
      num_groups_13222 = squot32(x_13221, group_sizze_13219)
      num_threads_13223 = (group_sizze_13219 * num_groups_13222)
      mem_13468 = opencl_alloc(self, bytes_13321, "mem_13468")
      if ((1 * (num_groups_13222 * group_sizze_13219)) != 0):
        self.map_kernel_13224_var.set_args(np.int32(x_11921), mem_13379,
                                           mem_13465, mem_13468)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_13224_var,
                                   (np.long((num_groups_13222 * group_sizze_13219)),),
                                   (np.long(group_sizze_13219),))
        if synchronous:
          self.queue.finish()
      mem_13379 = None
      mem_13465 = None
      conc_tmp_12452 = (sizze_11909 + x_11921)
      binop_x_13470 = sext_i32_i64(conc_tmp_12452)
      bytes_13469 = (np.int64(4) * binop_x_13470)
      mem_13471 = opencl_alloc(self, bytes_13469, "mem_13471")
      tmp_offs_13659 = np.int32(0)
      if ((sizze_11909 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_13471, output_mem_13316,
                        dest_offset=np.long((tmp_offs_13659 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sizze_11909 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_13659 = (tmp_offs_13659 + sizze_11909)
      if ((x_11921 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_13471, mem_13468,
                        dest_offset=np.long((tmp_offs_13659 * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((x_11921 * np.int32(4))))
      if synchronous:
        self.queue.finish()
      tmp_offs_13659 = (tmp_offs_13659 + x_11921)
      mem_13468 = None
      loop_cond_12454 = slt32(x_11913, res_11890)
      sizze_tmp_13517 = conc_tmp_12452
      output_mem_sizze_tmp_13518 = bytes_13469
      output_mem_tmp_13519 = mem_13471
      loop_while_tmp_13520 = loop_cond_12454
      run_tmp_13522 = x_11913
      sizze_11909 = sizze_tmp_13517
      output_mem_sizze_13315 = output_mem_sizze_tmp_13518
      output_mem_13316 = output_mem_tmp_13519
      loop_while_11910 = loop_while_tmp_13520
      run_11912 = run_tmp_13522
    sizze_11905 = sizze_11909
    res_mem_sizze_13472 = output_mem_sizze_13315
    res_mem_13473 = output_mem_13316
    res_11906 = loop_while_11910
    res_11908 = run_11912
    mem_13301 = None
    mem_13305 = None
    mem_13308 = None
    mem_13311 = None
    mem_13314 = None
    mem_13364 = None
    mem_13394 = None
    mem_13432 = None
    mem_13435 = None
    j_m_i_12455 = (sizze_11905 - np.int32(1))
    x_12456 = abs(j_m_i_12455)
    empty_slice_12457 = (x_12456 == np.int32(0))
    m_12458 = (x_12456 - np.int32(1))
    i_p_m_t_s_12459 = (np.int32(1) + m_12458)
    zzero_leq_i_p_m_t_s_12460 = sle32(np.int32(0), i_p_m_t_s_12459)
    i_p_m_t_s_leq_w_12461 = slt32(i_p_m_t_s_12459, sizze_11905)
    i_lte_j_12462 = sle32(np.int32(1), sizze_11905)
    y_12463 = (zzero_leq_i_p_m_t_s_12460 and i_p_m_t_s_leq_w_12461)
    y_12464 = (i_lte_j_12462 and y_12463)
    ok_or_empty_12465 = (empty_slice_12457 or y_12464)
    index_certs_12466 = True
    assert ok_or_empty_12465, ("Error at forwardprojection_semiflat.fut:32:1-36:64 -> forwardprojection_semiflat.fut:36:11-36:64 -> projection_lib.fut:313:20-313:31 -> /futlib/array.fut:21:29-21:33: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                                      np.int32(1),
                                                                                                                                                                                                                      "] out of bounds for array of shape [",
                                                                                                                                                                                                                      sizze_11905,
                                                                                                                                                                                                                      "]."))
    binop_x_13475 = sext_i32_i64(x_12456)
    bytes_13474 = (np.int64(4) * binop_x_13475)
    mem_13476 = opencl_alloc(self, bytes_13474, "mem_13476")
    if ((x_12456 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_13476, res_mem_13473,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(4)),
                      byte_count=np.long((x_12456 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    res_mem_13473 = None
    out_arrsizze_13500 = x_12456
    out_memsizze_13499 = bytes_13474
    out_mem_13498 = mem_13476
    return (out_memsizze_13499, out_mem_13498, out_arrsizze_13500)
  def main(self, angles_mem_13287_ext, rays_mem_13289_ext, voxels_mem_13291_ext,
           stepsizze_11843_ext):
    try:
      assert ((type(angles_mem_13287_ext) in [np.ndarray,
                                              cl.array.Array]) and (angles_mem_13287_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_11837 = np.int32(angles_mem_13287_ext.shape[0])
      angles_mem_sizze_13286 = np.int64(angles_mem_13287_ext.nbytes)
      if (type(angles_mem_13287_ext) == cl.array.Array):
        angles_mem_13287 = angles_mem_13287_ext.data
      else:
        angles_mem_13287 = opencl_alloc(self, angles_mem_sizze_13286,
                                        "angles_mem_13287")
        if (angles_mem_sizze_13286 != 0):
          cl.enqueue_copy(self.queue, angles_mem_13287,
                          normaliseArray(angles_mem_13287_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(angles_mem_13287_ext),
                                                                                                                            angles_mem_13287_ext))
    try:
      assert ((type(rays_mem_13289_ext) in [np.ndarray,
                                            cl.array.Array]) and (rays_mem_13289_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_11838 = np.int32(rays_mem_13289_ext.shape[0])
      rays_mem_sizze_13288 = np.int64(rays_mem_13289_ext.nbytes)
      if (type(rays_mem_13289_ext) == cl.array.Array):
        rays_mem_13289 = rays_mem_13289_ext.data
      else:
        rays_mem_13289 = opencl_alloc(self, rays_mem_sizze_13288,
                                      "rays_mem_13289")
        if (rays_mem_sizze_13288 != 0):
          cl.enqueue_copy(self.queue, rays_mem_13289,
                          normaliseArray(rays_mem_13289_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(rays_mem_13289_ext),
                                                                                                                            rays_mem_13289_ext))
    try:
      assert ((type(voxels_mem_13291_ext) in [np.ndarray,
                                              cl.array.Array]) and (voxels_mem_13291_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_11839 = np.int32(voxels_mem_13291_ext.shape[0])
      voxels_mem_sizze_13290 = np.int64(voxels_mem_13291_ext.nbytes)
      if (type(voxels_mem_13291_ext) == cl.array.Array):
        voxels_mem_13291 = voxels_mem_13291_ext.data
      else:
        voxels_mem_13291 = opencl_alloc(self, voxels_mem_sizze_13290,
                                        "voxels_mem_13291")
        if (voxels_mem_sizze_13290 != 0):
          cl.enqueue_copy(self.queue, voxels_mem_13291,
                          normaliseArray(voxels_mem_13291_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(voxels_mem_13291_ext),
                                                                                                                            voxels_mem_13291_ext))
    try:
      stepsizze_11843 = np.int32(ct.c_int32(stepsizze_11843_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(stepsizze_11843_ext),
                                                                                                                            stepsizze_11843_ext))
    (out_memsizze_13499, out_mem_13498,
     out_arrsizze_13500) = self.futhark_main(angles_mem_sizze_13286,
                                             angles_mem_13287,
                                             rays_mem_sizze_13288,
                                             rays_mem_13289,
                                             voxels_mem_sizze_13290,
                                             voxels_mem_13291, sizze_11837,
                                             sizze_11838, sizze_11839,
                                             stepsizze_11843)
    return cl.array.Array(self.queue, (out_arrsizze_13500,), ct.c_float,
                          data=out_mem_13498)