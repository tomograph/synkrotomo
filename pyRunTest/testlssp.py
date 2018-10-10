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
#define group_sizze_4849 (group_size_4848)
#define max_num_groups_4851 (max_num_groups_4850)
__kernel void chunked_reduce_kernel_4870(__local volatile
                                         int64_t *mem_aligned_0,
                                         __local volatile
                                         int64_t *mem_aligned_1,
                                         __local volatile
                                         int64_t *mem_aligned_2,
                                         __local volatile
                                         int64_t *mem_aligned_3,
                                         __local volatile
                                         int64_t *mem_aligned_4,
                                         __local volatile
                                         int64_t *mem_aligned_5,
                                         int32_t sizze_4756,
                                         unsigned char cond_4759,
                                         unsigned char x_4760,
                                         unsigned char cond_4762,
                                         unsigned char cond_4763,
                                         unsigned char x_4764,
                                         unsigned char x_4765,
                                         int32_t num_threads_4857,
                                         int32_t per_thread_elements_4860,
                                         int32_t per_chunk_5109, __global
                                         unsigned char *mem_5137, __global
                                         unsigned char *mem_5158, __global
                                         unsigned char *mem_5161, __global
                                         unsigned char *mem_5164, __global
                                         unsigned char *mem_5167, __global
                                         unsigned char *mem_5170, __global
                                         unsigned char *mem_5173)
{
    __local volatile char *restrict mem_5140 = mem_aligned_0;
    __local volatile char *restrict mem_5143 = mem_aligned_1;
    __local volatile char *restrict mem_5146 = mem_aligned_2;
    __local volatile char *restrict mem_5149 = mem_aligned_3;
    __local volatile char *restrict mem_5152 = mem_aligned_4;
    __local volatile char *restrict mem_5155 = mem_aligned_5;
    int32_t wave_sizze_5214;
    int32_t group_sizze_5215;
    bool thread_active_5216;
    int32_t global_tid_4870;
    int32_t local_tid_4871;
    int32_t group_id_4872;
    
    global_tid_4870 = get_global_id(0);
    local_tid_4871 = get_local_id(0);
    group_sizze_5215 = get_local_size(0);
    wave_sizze_5214 = LOCKSTEP_WIDTH;
    group_id_4872 = get_group_id(0);
    thread_active_5216 = 1;
    
    int32_t chunk_sizze_4898;
    int32_t starting_point_5217 = global_tid_4870 * per_thread_elements_4860;
    int32_t remaining_elements_5218 = sizze_4756 - starting_point_5217;
    
    if (sle32(remaining_elements_5218, 0) || sle32(sizze_4756,
                                                   starting_point_5217)) {
        chunk_sizze_4898 = 0;
    } else {
        if (slt32(sizze_4756, (global_tid_4870 + 1) *
                  per_thread_elements_4860)) {
            chunk_sizze_4898 = sizze_4756 - global_tid_4870 *
                per_thread_elements_4860;
        } else {
            chunk_sizze_4898 = per_thread_elements_4860;
        }
    }
    
    int32_t slice_offset_4899;
    int32_t res_4908;
    int32_t res_4909;
    int32_t res_4910;
    int32_t res_4911;
    int32_t res_4912;
    int32_t res_4913;
    
    if (thread_active_5216) {
        slice_offset_4899 = per_thread_elements_4860 * global_tid_4870;
        
        int32_t acc_4916;
        int32_t acc_4917;
        int32_t acc_4918;
        int32_t acc_4919;
        int32_t acc_4920;
        int32_t acc_4921;
        
        acc_4916 = 0;
        acc_4917 = 0;
        acc_4918 = 0;
        acc_4919 = 0;
        acc_4920 = 0;
        acc_4921 = 0;
        for (int32_t i_4915 = 0; i_4915 < chunk_sizze_4898; i_4915++) {
            int32_t j_p_i_t_s_5121 = slice_offset_4899 + i_4915;
            int32_t new_index_5122 = squot32(j_p_i_t_s_5121, per_chunk_5109);
            int32_t binop_y_5124 = per_chunk_5109 * new_index_5122;
            int32_t new_index_5125 = j_p_i_t_s_5121 - binop_y_5124;
            int32_t x_4924 = *(__global int32_t *) &mem_5137[(new_index_5125 *
                                                              num_threads_4857 +
                                                              new_index_5122) *
                                                             4];
            bool res_4932 = x_4924 == 0;
            bool x_4933 = cond_4759 && res_4932;
            bool res_4934 = x_4760 || x_4933;
            int32_t res_4935;
            
            if (res_4934) {
                res_4935 = 1;
            } else {
                res_4935 = 0;
            }
            
            bool cond_4942 = acc_4919 == 0;
            bool cond_4947 = acc_4921 == 0;
            bool x_4949 = res_4932 && cond_4947;
            bool res_4950 = sle32(acc_4921, x_4924);
            bool res_4951 = acc_4921 == x_4924;
            bool x_4952 = cond_4763 && res_4951;
            bool res_4953 = x_4764 || x_4952;
            bool x_4954 = cond_4762 && res_4950;
            bool y_4955 = x_4765 && res_4953;
            bool res_4956 = x_4954 || y_4955;
            bool x_4957 = cond_4759 && x_4949;
            bool y_4958 = x_4760 && res_4956;
            bool res_4959 = x_4957 || y_4958;
            bool x_4960 = !cond_4942;
            bool y_4961 = res_4959 && x_4960;
            bool res_4962 = cond_4942 || y_4961;
            int32_t res_4963;
            
            if (res_4962) {
                int32_t arg_4964;
                int32_t res_4965;
                int32_t res_4966;
                
                arg_4964 = acc_4918 + res_4935;
                res_4965 = smax32(acc_4916, arg_4964);
                res_4966 = smax32(res_4935, res_4965);
                res_4963 = res_4966;
            } else {
                int32_t res_4967 = smax32(acc_4916, res_4935);
                
                res_4963 = res_4967;
            }
            
            int32_t res_4968;
            
            if (cond_4942) {
                res_4968 = res_4935;
            } else {
                bool cond_4969;
                bool x_4970;
                int32_t res_4971;
                
                cond_4969 = acc_4919 == acc_4917;
                x_4970 = res_4962 && cond_4969;
                if (x_4970) {
                    int32_t res_4972 = acc_4917 + res_4935;
                    
                    res_4971 = res_4972;
                } else {
                    res_4971 = acc_4917;
                }
                res_4968 = res_4971;
            }
            
            bool x_4975 = res_4934 && res_4962;
            int32_t res_4976;
            
            if (x_4975) {
                int32_t res_4977 = acc_4918 + res_4935;
                
                res_4976 = res_4977;
            } else {
                res_4976 = res_4935;
            }
            
            int32_t res_4978 = 1 + acc_4919;
            int32_t res_4979;
            
            if (cond_4942) {
                res_4979 = x_4924;
            } else {
                res_4979 = acc_4920;
            }
            
            int32_t acc_tmp_5219 = res_4963;
            int32_t acc_tmp_5220 = res_4968;
            int32_t acc_tmp_5221 = res_4976;
            int32_t acc_tmp_5222 = res_4978;
            int32_t acc_tmp_5223 = res_4979;
            int32_t acc_tmp_5224;
            
            acc_tmp_5224 = x_4924;
            acc_4916 = acc_tmp_5219;
            acc_4917 = acc_tmp_5220;
            acc_4918 = acc_tmp_5221;
            acc_4919 = acc_tmp_5222;
            acc_4920 = acc_tmp_5223;
            acc_4921 = acc_tmp_5224;
        }
        res_4908 = acc_4916;
        res_4909 = acc_4917;
        res_4910 = acc_4918;
        res_4911 = acc_4919;
        res_4912 = acc_4920;
        res_4913 = acc_4921;
    }
    
    int32_t final_result_4993;
    int32_t final_result_4994;
    int32_t final_result_4995;
    int32_t final_result_4996;
    int32_t final_result_4997;
    int32_t final_result_4998;
    
    for (int32_t comb_iter_5225 = 0; comb_iter_5225 < squot32(group_sizze_4849 +
                                                              group_sizze_4849 -
                                                              1,
                                                              group_sizze_4849);
         comb_iter_5225++) {
        int32_t combine_id_4886;
        int32_t flat_comb_id_5226 = comb_iter_5225 * group_sizze_4849 +
                local_tid_4871;
        
        combine_id_4886 = flat_comb_id_5226;
        if (slt32(combine_id_4886, group_sizze_4849) && 1) {
            *(__local int32_t *) &mem_5140[combine_id_4886 * 4] = res_4908;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t comb_iter_5227 = 0; comb_iter_5227 < squot32(group_sizze_4849 +
                                                              group_sizze_4849 -
                                                              1,
                                                              group_sizze_4849);
         comb_iter_5227++) {
        int32_t combine_id_4887;
        int32_t flat_comb_id_5228 = comb_iter_5227 * group_sizze_4849 +
                local_tid_4871;
        
        combine_id_4887 = flat_comb_id_5228;
        if (slt32(combine_id_4887, group_sizze_4849) && 1) {
            *(__local int32_t *) &mem_5143[combine_id_4887 * 4] = res_4909;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t comb_iter_5229 = 0; comb_iter_5229 < squot32(group_sizze_4849 +
                                                              group_sizze_4849 -
                                                              1,
                                                              group_sizze_4849);
         comb_iter_5229++) {
        int32_t combine_id_4888;
        int32_t flat_comb_id_5230 = comb_iter_5229 * group_sizze_4849 +
                local_tid_4871;
        
        combine_id_4888 = flat_comb_id_5230;
        if (slt32(combine_id_4888, group_sizze_4849) && 1) {
            *(__local int32_t *) &mem_5146[combine_id_4888 * 4] = res_4910;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t comb_iter_5231 = 0; comb_iter_5231 < squot32(group_sizze_4849 +
                                                              group_sizze_4849 -
                                                              1,
                                                              group_sizze_4849);
         comb_iter_5231++) {
        int32_t combine_id_4889;
        int32_t flat_comb_id_5232 = comb_iter_5231 * group_sizze_4849 +
                local_tid_4871;
        
        combine_id_4889 = flat_comb_id_5232;
        if (slt32(combine_id_4889, group_sizze_4849) && 1) {
            *(__local int32_t *) &mem_5149[combine_id_4889 * 4] = res_4911;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t comb_iter_5233 = 0; comb_iter_5233 < squot32(group_sizze_4849 +
                                                              group_sizze_4849 -
                                                              1,
                                                              group_sizze_4849);
         comb_iter_5233++) {
        int32_t combine_id_4890;
        int32_t flat_comb_id_5234 = comb_iter_5233 * group_sizze_4849 +
                local_tid_4871;
        
        combine_id_4890 = flat_comb_id_5234;
        if (slt32(combine_id_4890, group_sizze_4849) && 1) {
            *(__local int32_t *) &mem_5152[combine_id_4890 * 4] = res_4912;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t comb_iter_5235 = 0; comb_iter_5235 < squot32(group_sizze_4849 +
                                                              group_sizze_4849 -
                                                              1,
                                                              group_sizze_4849);
         comb_iter_5235++) {
        int32_t combine_id_4891;
        int32_t flat_comb_id_5236 = comb_iter_5235 * group_sizze_4849 +
                local_tid_4871;
        
        combine_id_4891 = flat_comb_id_5236;
        if (slt32(combine_id_4891, group_sizze_4849) && 1) {
            *(__local int32_t *) &mem_5155[combine_id_4891 * 4] = res_4913;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_5238;
    int32_t skip_waves_5237;
    int32_t my_index_4999;
    int32_t other_index_5000;
    int32_t x_5001;
    int32_t x_5002;
    int32_t x_5003;
    int32_t x_5004;
    int32_t x_5005;
    int32_t x_5006;
    int32_t x_5007;
    int32_t x_5008;
    int32_t x_5009;
    int32_t x_5010;
    int32_t x_5011;
    int32_t x_5012;
    
    my_index_4999 = local_tid_4871;
    offset_5238 = 0;
    other_index_5000 = local_tid_4871 + offset_5238;
    if (slt32(local_tid_4871, group_sizze_4849)) {
        x_5001 = *(__local int32_t *) &mem_5140[(local_tid_4871 + offset_5238) *
                                                4];
        x_5002 = *(__local int32_t *) &mem_5143[(local_tid_4871 + offset_5238) *
                                                4];
        x_5003 = *(__local int32_t *) &mem_5146[(local_tid_4871 + offset_5238) *
                                                4];
        x_5004 = *(__local int32_t *) &mem_5149[(local_tid_4871 + offset_5238) *
                                                4];
        x_5005 = *(__local int32_t *) &mem_5152[(local_tid_4871 + offset_5238) *
                                                4];
        x_5006 = *(__local int32_t *) &mem_5155[(local_tid_4871 + offset_5238) *
                                                4];
    }
    offset_5238 = 1;
    other_index_5000 = local_tid_4871 + offset_5238;
    while (slt32(offset_5238, wave_sizze_5214)) {
        if (slt32(other_index_5000, group_sizze_4849) && ((local_tid_4871 -
                                                           squot32(local_tid_4871,
                                                                   wave_sizze_5214) *
                                                           wave_sizze_5214) &
                                                          (2 * offset_5238 -
                                                           1)) == 0) {
            // read array element
            {
                x_5007 = *(volatile __local
                           int32_t *) &mem_5140[(local_tid_4871 + offset_5238) *
                                                4];
                x_5008 = *(volatile __local
                           int32_t *) &mem_5143[(local_tid_4871 + offset_5238) *
                                                4];
                x_5009 = *(volatile __local
                           int32_t *) &mem_5146[(local_tid_4871 + offset_5238) *
                                                4];
                x_5010 = *(volatile __local
                           int32_t *) &mem_5149[(local_tid_4871 + offset_5238) *
                                                4];
                x_5011 = *(volatile __local
                           int32_t *) &mem_5152[(local_tid_4871 + offset_5238) *
                                                4];
                x_5012 = *(volatile __local
                           int32_t *) &mem_5155[(local_tid_4871 + offset_5238) *
                                                4];
            }
            
            bool cond_5013;
            bool res_5014;
            bool x_5015;
            bool y_5016;
            bool cond_5017;
            bool cond_5018;
            bool res_5019;
            bool x_5020;
            bool res_5021;
            bool res_5022;
            bool x_5023;
            bool res_5024;
            bool x_5025;
            bool y_5026;
            bool res_5027;
            bool x_5028;
            bool y_5029;
            bool res_5030;
            bool x_5031;
            bool y_5032;
            bool res_5033;
            int32_t res_5034;
            int32_t res_5039;
            int32_t res_5044;
            int32_t res_5049;
            int32_t res_5050;
            int32_t res_5051;
            
            if (thread_active_5216) {
                cond_5013 = x_5004 == 0;
                res_5014 = x_5010 == 0;
                x_5015 = !cond_5013;
                y_5016 = res_5014 && x_5015;
                cond_5017 = cond_5013 || y_5016;
                cond_5018 = x_5006 == 0;
                res_5019 = x_5011 == 0;
                x_5020 = cond_5018 && res_5019;
                res_5021 = sle32(x_5006, x_5011);
                res_5022 = x_5006 == x_5011;
                x_5023 = cond_4763 && res_5022;
                res_5024 = x_4764 || x_5023;
                x_5025 = cond_4762 && res_5021;
                y_5026 = x_4765 && res_5024;
                res_5027 = x_5025 || y_5026;
                x_5028 = cond_4759 && x_5020;
                y_5029 = x_4760 && res_5027;
                res_5030 = x_5028 || y_5029;
                x_5031 = !cond_5017;
                y_5032 = res_5030 && x_5031;
                res_5033 = cond_5017 || y_5032;
                if (res_5033) {
                    int32_t arg_5035;
                    int32_t res_5036;
                    int32_t res_5037;
                    
                    arg_5035 = x_5003 + x_5008;
                    res_5036 = smax32(x_5001, arg_5035);
                    res_5037 = smax32(x_5007, res_5036);
                    res_5034 = res_5037;
                } else {
                    int32_t res_5038 = smax32(x_5001, x_5007);
                    
                    res_5034 = res_5038;
                }
                if (cond_5013) {
                    res_5039 = x_5008;
                } else {
                    bool cond_5040;
                    bool x_5041;
                    int32_t res_5042;
                    
                    cond_5040 = x_5004 == x_5002;
                    x_5041 = res_5033 && cond_5040;
                    if (x_5041) {
                        int32_t res_5043 = x_5002 + x_5008;
                        
                        res_5042 = res_5043;
                    } else {
                        res_5042 = x_5002;
                    }
                    res_5039 = res_5042;
                }
                if (res_5014) {
                    res_5044 = x_5003;
                } else {
                    bool cond_5045;
                    bool x_5046;
                    int32_t res_5047;
                    
                    cond_5045 = x_5010 == x_5009;
                    x_5046 = res_5033 && cond_5045;
                    if (x_5046) {
                        int32_t res_5048 = x_5003 + x_5009;
                        
                        res_5047 = res_5048;
                    } else {
                        res_5047 = x_5009;
                    }
                    res_5044 = res_5047;
                }
                res_5049 = x_5004 + x_5010;
                if (cond_5013) {
                    res_5050 = x_5011;
                } else {
                    res_5050 = x_5005;
                }
                if (res_5014) {
                    res_5051 = x_5006;
                } else {
                    res_5051 = x_5012;
                }
            }
            x_5001 = res_5034;
            x_5002 = res_5039;
            x_5003 = res_5044;
            x_5004 = res_5049;
            x_5005 = res_5050;
            x_5006 = res_5051;
            *(volatile __local int32_t *) &mem_5140[local_tid_4871 * 4] =
                x_5001;
            *(volatile __local int32_t *) &mem_5143[local_tid_4871 * 4] =
                x_5002;
            *(volatile __local int32_t *) &mem_5146[local_tid_4871 * 4] =
                x_5003;
            *(volatile __local int32_t *) &mem_5149[local_tid_4871 * 4] =
                x_5004;
            *(volatile __local int32_t *) &mem_5152[local_tid_4871 * 4] =
                x_5005;
            *(volatile __local int32_t *) &mem_5155[local_tid_4871 * 4] =
                x_5006;
        }
        offset_5238 *= 2;
        other_index_5000 = local_tid_4871 + offset_5238;
    }
    skip_waves_5237 = 1;
    while (slt32(skip_waves_5237, squot32(group_sizze_4849 + wave_sizze_5214 -
                                          1, wave_sizze_5214))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_5238 = skip_waves_5237 * wave_sizze_5214;
        other_index_5000 = local_tid_4871 + offset_5238;
        if (slt32(other_index_5000, group_sizze_4849) && ((local_tid_4871 -
                                                           squot32(local_tid_4871,
                                                                   wave_sizze_5214) *
                                                           wave_sizze_5214) ==
                                                          0 &&
                                                          (squot32(local_tid_4871,
                                                                   wave_sizze_5214) &
                                                           (2 *
                                                            skip_waves_5237 -
                                                            1)) == 0)) {
            // read array element
            {
                x_5007 = *(__local int32_t *) &mem_5140[(local_tid_4871 +
                                                         offset_5238) * 4];
                x_5008 = *(__local int32_t *) &mem_5143[(local_tid_4871 +
                                                         offset_5238) * 4];
                x_5009 = *(__local int32_t *) &mem_5146[(local_tid_4871 +
                                                         offset_5238) * 4];
                x_5010 = *(__local int32_t *) &mem_5149[(local_tid_4871 +
                                                         offset_5238) * 4];
                x_5011 = *(__local int32_t *) &mem_5152[(local_tid_4871 +
                                                         offset_5238) * 4];
                x_5012 = *(__local int32_t *) &mem_5155[(local_tid_4871 +
                                                         offset_5238) * 4];
            }
            
            bool cond_5013;
            bool res_5014;
            bool x_5015;
            bool y_5016;
            bool cond_5017;
            bool cond_5018;
            bool res_5019;
            bool x_5020;
            bool res_5021;
            bool res_5022;
            bool x_5023;
            bool res_5024;
            bool x_5025;
            bool y_5026;
            bool res_5027;
            bool x_5028;
            bool y_5029;
            bool res_5030;
            bool x_5031;
            bool y_5032;
            bool res_5033;
            int32_t res_5034;
            int32_t res_5039;
            int32_t res_5044;
            int32_t res_5049;
            int32_t res_5050;
            int32_t res_5051;
            
            if (thread_active_5216) {
                cond_5013 = x_5004 == 0;
                res_5014 = x_5010 == 0;
                x_5015 = !cond_5013;
                y_5016 = res_5014 && x_5015;
                cond_5017 = cond_5013 || y_5016;
                cond_5018 = x_5006 == 0;
                res_5019 = x_5011 == 0;
                x_5020 = cond_5018 && res_5019;
                res_5021 = sle32(x_5006, x_5011);
                res_5022 = x_5006 == x_5011;
                x_5023 = cond_4763 && res_5022;
                res_5024 = x_4764 || x_5023;
                x_5025 = cond_4762 && res_5021;
                y_5026 = x_4765 && res_5024;
                res_5027 = x_5025 || y_5026;
                x_5028 = cond_4759 && x_5020;
                y_5029 = x_4760 && res_5027;
                res_5030 = x_5028 || y_5029;
                x_5031 = !cond_5017;
                y_5032 = res_5030 && x_5031;
                res_5033 = cond_5017 || y_5032;
                if (res_5033) {
                    int32_t arg_5035;
                    int32_t res_5036;
                    int32_t res_5037;
                    
                    arg_5035 = x_5003 + x_5008;
                    res_5036 = smax32(x_5001, arg_5035);
                    res_5037 = smax32(x_5007, res_5036);
                    res_5034 = res_5037;
                } else {
                    int32_t res_5038 = smax32(x_5001, x_5007);
                    
                    res_5034 = res_5038;
                }
                if (cond_5013) {
                    res_5039 = x_5008;
                } else {
                    bool cond_5040;
                    bool x_5041;
                    int32_t res_5042;
                    
                    cond_5040 = x_5004 == x_5002;
                    x_5041 = res_5033 && cond_5040;
                    if (x_5041) {
                        int32_t res_5043 = x_5002 + x_5008;
                        
                        res_5042 = res_5043;
                    } else {
                        res_5042 = x_5002;
                    }
                    res_5039 = res_5042;
                }
                if (res_5014) {
                    res_5044 = x_5003;
                } else {
                    bool cond_5045;
                    bool x_5046;
                    int32_t res_5047;
                    
                    cond_5045 = x_5010 == x_5009;
                    x_5046 = res_5033 && cond_5045;
                    if (x_5046) {
                        int32_t res_5048 = x_5003 + x_5009;
                        
                        res_5047 = res_5048;
                    } else {
                        res_5047 = x_5009;
                    }
                    res_5044 = res_5047;
                }
                res_5049 = x_5004 + x_5010;
                if (cond_5013) {
                    res_5050 = x_5011;
                } else {
                    res_5050 = x_5005;
                }
                if (res_5014) {
                    res_5051 = x_5006;
                } else {
                    res_5051 = x_5012;
                }
            }
            x_5001 = res_5034;
            x_5002 = res_5039;
            x_5003 = res_5044;
            x_5004 = res_5049;
            x_5005 = res_5050;
            x_5006 = res_5051;
            *(__local int32_t *) &mem_5140[local_tid_4871 * 4] = x_5001;
            *(__local int32_t *) &mem_5143[local_tid_4871 * 4] = x_5002;
            *(__local int32_t *) &mem_5146[local_tid_4871 * 4] = x_5003;
            *(__local int32_t *) &mem_5149[local_tid_4871 * 4] = x_5004;
            *(__local int32_t *) &mem_5152[local_tid_4871 * 4] = x_5005;
            *(__local int32_t *) &mem_5155[local_tid_4871 * 4] = x_5006;
        }
        skip_waves_5237 *= 2;
    }
    final_result_4993 = x_5001;
    final_result_4994 = x_5002;
    final_result_4995 = x_5003;
    final_result_4996 = x_5004;
    final_result_4997 = x_5005;
    final_result_4998 = x_5006;
    if (local_tid_4871 == 0) {
        *(__global int32_t *) &mem_5158[group_id_4872 * 4] = final_result_4993;
    }
    if (local_tid_4871 == 0) {
        *(__global int32_t *) &mem_5161[group_id_4872 * 4] = final_result_4994;
    }
    if (local_tid_4871 == 0) {
        *(__global int32_t *) &mem_5164[group_id_4872 * 4] = final_result_4995;
    }
    if (local_tid_4871 == 0) {
        *(__global int32_t *) &mem_5167[group_id_4872 * 4] = final_result_4996;
    }
    if (local_tid_4871 == 0) {
        *(__global int32_t *) &mem_5170[group_id_4872 * 4] = final_result_4997;
    }
    if (local_tid_4871 == 0) {
        *(__global int32_t *) &mem_5173[group_id_4872 * 4] = final_result_4998;
    }
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
__kernel void reduce_kernel_5058(__local volatile int64_t *mem_aligned_0,
                                 __local volatile int64_t *mem_aligned_1,
                                 __local volatile int64_t *mem_aligned_2,
                                 __local volatile int64_t *mem_aligned_3,
                                 __local volatile int64_t *mem_aligned_4,
                                 __local volatile int64_t *mem_aligned_5,
                                 unsigned char cond_4759, unsigned char x_4760,
                                 unsigned char cond_4762,
                                 unsigned char cond_4763, unsigned char x_4764,
                                 unsigned char x_4765, int32_t num_groups_4856,
                                 __global unsigned char *mem_5158, __global
                                 unsigned char *mem_5161, __global
                                 unsigned char *mem_5164, __global
                                 unsigned char *mem_5167, __global
                                 unsigned char *mem_5170, __global
                                 unsigned char *mem_5173, __global
                                 unsigned char *mem_5194, __global
                                 unsigned char *mem_5197, __global
                                 unsigned char *mem_5200, __global
                                 unsigned char *mem_5203, __global
                                 unsigned char *mem_5206, __global
                                 unsigned char *mem_5209)
{
    __local volatile char *restrict mem_5176 = mem_aligned_0;
    __local volatile char *restrict mem_5179 = mem_aligned_1;
    __local volatile char *restrict mem_5182 = mem_aligned_2;
    __local volatile char *restrict mem_5185 = mem_aligned_3;
    __local volatile char *restrict mem_5188 = mem_aligned_4;
    __local volatile char *restrict mem_5191 = mem_aligned_5;
    int32_t wave_sizze_5245;
    int32_t group_sizze_5246;
    bool thread_active_5247;
    int32_t global_tid_5058;
    int32_t local_tid_5059;
    int32_t group_id_5060;
    
    global_tid_5058 = get_global_id(0);
    local_tid_5059 = get_local_id(0);
    group_sizze_5246 = get_local_size(0);
    wave_sizze_5245 = LOCKSTEP_WIDTH;
    group_id_5060 = get_group_id(0);
    thread_active_5247 = 1;
    
    bool in_bounds_5061;
    int32_t x_5092;
    int32_t x_5094;
    int32_t x_5096;
    int32_t x_5098;
    int32_t x_5100;
    int32_t x_5102;
    
    if (thread_active_5247) {
        in_bounds_5061 = slt32(local_tid_5059, num_groups_4856);
        if (in_bounds_5061) {
            int32_t x_5062 = *(__global int32_t *) &mem_5158[global_tid_5058 *
                                                             4];
            
            x_5092 = x_5062;
        } else {
            x_5092 = 0;
        }
        if (in_bounds_5061) {
            int32_t x_5064 = *(__global int32_t *) &mem_5161[global_tid_5058 *
                                                             4];
            
            x_5094 = x_5064;
        } else {
            x_5094 = 0;
        }
        if (in_bounds_5061) {
            int32_t x_5066 = *(__global int32_t *) &mem_5164[global_tid_5058 *
                                                             4];
            
            x_5096 = x_5066;
        } else {
            x_5096 = 0;
        }
        if (in_bounds_5061) {
            int32_t x_5068 = *(__global int32_t *) &mem_5167[global_tid_5058 *
                                                             4];
            
            x_5098 = x_5068;
        } else {
            x_5098 = 0;
        }
        if (in_bounds_5061) {
            int32_t x_5070 = *(__global int32_t *) &mem_5170[global_tid_5058 *
                                                             4];
            
            x_5100 = x_5070;
        } else {
            x_5100 = 0;
        }
        if (in_bounds_5061) {
            int32_t x_5072 = *(__global int32_t *) &mem_5173[global_tid_5058 *
                                                             4];
            
            x_5102 = x_5072;
        } else {
            x_5102 = 0;
        }
    }
    
    int32_t final_result_5081;
    int32_t final_result_5082;
    int32_t final_result_5083;
    int32_t final_result_5084;
    int32_t final_result_5085;
    int32_t final_result_5086;
    
    for (int32_t comb_iter_5248 = 0; comb_iter_5248 <
         squot32(max_num_groups_4851 + max_num_groups_4851 - 1,
                 max_num_groups_4851); comb_iter_5248++) {
        int32_t combine_id_5080;
        int32_t flat_comb_id_5249 = comb_iter_5248 * max_num_groups_4851 +
                local_tid_5059;
        
        combine_id_5080 = flat_comb_id_5249;
        if (slt32(combine_id_5080, max_num_groups_4851) && 1) {
            *(__local int32_t *) &mem_5176[combine_id_5080 * 4] = x_5092;
            *(__local int32_t *) &mem_5179[combine_id_5080 * 4] = x_5094;
            *(__local int32_t *) &mem_5182[combine_id_5080 * 4] = x_5096;
            *(__local int32_t *) &mem_5185[combine_id_5080 * 4] = x_5098;
            *(__local int32_t *) &mem_5188[combine_id_5080 * 4] = x_5100;
            *(__local int32_t *) &mem_5191[combine_id_5080 * 4] = x_5102;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_5251;
    int32_t skip_waves_5250;
    int32_t x_4772;
    int32_t x_4773;
    int32_t x_4774;
    int32_t x_4775;
    int32_t x_4776;
    int32_t x_4777;
    int32_t x_4778;
    int32_t x_4779;
    int32_t x_4780;
    int32_t x_4781;
    int32_t x_4782;
    int32_t x_4783;
    int32_t my_index_4868;
    int32_t other_index_4869;
    
    my_index_4868 = local_tid_5059;
    offset_5251 = 0;
    other_index_4869 = local_tid_5059 + offset_5251;
    if (slt32(local_tid_5059, max_num_groups_4851)) {
        x_4772 = *(__local int32_t *) &mem_5176[(local_tid_5059 + offset_5251) *
                                                4];
        x_4773 = *(__local int32_t *) &mem_5179[(local_tid_5059 + offset_5251) *
                                                4];
        x_4774 = *(__local int32_t *) &mem_5182[(local_tid_5059 + offset_5251) *
                                                4];
        x_4775 = *(__local int32_t *) &mem_5185[(local_tid_5059 + offset_5251) *
                                                4];
        x_4776 = *(__local int32_t *) &mem_5188[(local_tid_5059 + offset_5251) *
                                                4];
        x_4777 = *(__local int32_t *) &mem_5191[(local_tid_5059 + offset_5251) *
                                                4];
    }
    offset_5251 = 1;
    other_index_4869 = local_tid_5059 + offset_5251;
    while (slt32(offset_5251, wave_sizze_5245)) {
        if (slt32(other_index_4869, max_num_groups_4851) && ((local_tid_5059 -
                                                              squot32(local_tid_5059,
                                                                      wave_sizze_5245) *
                                                              wave_sizze_5245) &
                                                             (2 * offset_5251 -
                                                              1)) == 0) {
            // read array element
            {
                x_4778 = *(volatile __local
                           int32_t *) &mem_5176[(local_tid_5059 + offset_5251) *
                                                4];
                x_4779 = *(volatile __local
                           int32_t *) &mem_5179[(local_tid_5059 + offset_5251) *
                                                4];
                x_4780 = *(volatile __local
                           int32_t *) &mem_5182[(local_tid_5059 + offset_5251) *
                                                4];
                x_4781 = *(volatile __local
                           int32_t *) &mem_5185[(local_tid_5059 + offset_5251) *
                                                4];
                x_4782 = *(volatile __local
                           int32_t *) &mem_5188[(local_tid_5059 + offset_5251) *
                                                4];
                x_4783 = *(volatile __local
                           int32_t *) &mem_5191[(local_tid_5059 + offset_5251) *
                                                4];
            }
            
            bool cond_4784;
            bool res_4785;
            bool x_4786;
            bool y_4787;
            bool cond_4788;
            bool cond_4789;
            bool res_4790;
            bool x_4791;
            bool res_4792;
            bool res_4793;
            bool x_4794;
            bool res_4795;
            bool x_4796;
            bool y_4797;
            bool res_4798;
            bool x_4799;
            bool y_4800;
            bool res_4801;
            bool x_4802;
            bool y_4803;
            bool res_4804;
            int32_t res_4805;
            int32_t res_4810;
            int32_t res_4815;
            int32_t res_4820;
            int32_t res_4821;
            int32_t res_4822;
            
            if (thread_active_5247) {
                cond_4784 = x_4775 == 0;
                res_4785 = x_4781 == 0;
                x_4786 = !cond_4784;
                y_4787 = res_4785 && x_4786;
                cond_4788 = cond_4784 || y_4787;
                cond_4789 = x_4777 == 0;
                res_4790 = x_4782 == 0;
                x_4791 = cond_4789 && res_4790;
                res_4792 = sle32(x_4777, x_4782);
                res_4793 = x_4777 == x_4782;
                x_4794 = cond_4763 && res_4793;
                res_4795 = x_4764 || x_4794;
                x_4796 = cond_4762 && res_4792;
                y_4797 = x_4765 && res_4795;
                res_4798 = x_4796 || y_4797;
                x_4799 = cond_4759 && x_4791;
                y_4800 = x_4760 && res_4798;
                res_4801 = x_4799 || y_4800;
                x_4802 = !cond_4788;
                y_4803 = res_4801 && x_4802;
                res_4804 = cond_4788 || y_4803;
                if (res_4804) {
                    int32_t arg_4806;
                    int32_t res_4807;
                    int32_t res_4808;
                    
                    arg_4806 = x_4774 + x_4779;
                    res_4807 = smax32(x_4772, arg_4806);
                    res_4808 = smax32(x_4778, res_4807);
                    res_4805 = res_4808;
                } else {
                    int32_t res_4809 = smax32(x_4772, x_4778);
                    
                    res_4805 = res_4809;
                }
                if (cond_4784) {
                    res_4810 = x_4779;
                } else {
                    bool cond_4811;
                    bool x_4812;
                    int32_t res_4813;
                    
                    cond_4811 = x_4775 == x_4773;
                    x_4812 = res_4804 && cond_4811;
                    if (x_4812) {
                        int32_t res_4814 = x_4773 + x_4779;
                        
                        res_4813 = res_4814;
                    } else {
                        res_4813 = x_4773;
                    }
                    res_4810 = res_4813;
                }
                if (res_4785) {
                    res_4815 = x_4774;
                } else {
                    bool cond_4816;
                    bool x_4817;
                    int32_t res_4818;
                    
                    cond_4816 = x_4781 == x_4780;
                    x_4817 = res_4804 && cond_4816;
                    if (x_4817) {
                        int32_t res_4819 = x_4774 + x_4780;
                        
                        res_4818 = res_4819;
                    } else {
                        res_4818 = x_4780;
                    }
                    res_4815 = res_4818;
                }
                res_4820 = x_4775 + x_4781;
                if (cond_4784) {
                    res_4821 = x_4782;
                } else {
                    res_4821 = x_4776;
                }
                if (res_4785) {
                    res_4822 = x_4777;
                } else {
                    res_4822 = x_4783;
                }
            }
            x_4772 = res_4805;
            x_4773 = res_4810;
            x_4774 = res_4815;
            x_4775 = res_4820;
            x_4776 = res_4821;
            x_4777 = res_4822;
            *(volatile __local int32_t *) &mem_5176[local_tid_5059 * 4] =
                x_4772;
            *(volatile __local int32_t *) &mem_5179[local_tid_5059 * 4] =
                x_4773;
            *(volatile __local int32_t *) &mem_5182[local_tid_5059 * 4] =
                x_4774;
            *(volatile __local int32_t *) &mem_5185[local_tid_5059 * 4] =
                x_4775;
            *(volatile __local int32_t *) &mem_5188[local_tid_5059 * 4] =
                x_4776;
            *(volatile __local int32_t *) &mem_5191[local_tid_5059 * 4] =
                x_4777;
        }
        offset_5251 *= 2;
        other_index_4869 = local_tid_5059 + offset_5251;
    }
    skip_waves_5250 = 1;
    while (slt32(skip_waves_5250, squot32(max_num_groups_4851 +
                                          wave_sizze_5245 - 1,
                                          wave_sizze_5245))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_5251 = skip_waves_5250 * wave_sizze_5245;
        other_index_4869 = local_tid_5059 + offset_5251;
        if (slt32(other_index_4869, max_num_groups_4851) && ((local_tid_5059 -
                                                              squot32(local_tid_5059,
                                                                      wave_sizze_5245) *
                                                              wave_sizze_5245) ==
                                                             0 &&
                                                             (squot32(local_tid_5059,
                                                                      wave_sizze_5245) &
                                                              (2 *
                                                               skip_waves_5250 -
                                                               1)) == 0)) {
            // read array element
            {
                x_4778 = *(__local int32_t *) &mem_5176[(local_tid_5059 +
                                                         offset_5251) * 4];
                x_4779 = *(__local int32_t *) &mem_5179[(local_tid_5059 +
                                                         offset_5251) * 4];
                x_4780 = *(__local int32_t *) &mem_5182[(local_tid_5059 +
                                                         offset_5251) * 4];
                x_4781 = *(__local int32_t *) &mem_5185[(local_tid_5059 +
                                                         offset_5251) * 4];
                x_4782 = *(__local int32_t *) &mem_5188[(local_tid_5059 +
                                                         offset_5251) * 4];
                x_4783 = *(__local int32_t *) &mem_5191[(local_tid_5059 +
                                                         offset_5251) * 4];
            }
            
            bool cond_4784;
            bool res_4785;
            bool x_4786;
            bool y_4787;
            bool cond_4788;
            bool cond_4789;
            bool res_4790;
            bool x_4791;
            bool res_4792;
            bool res_4793;
            bool x_4794;
            bool res_4795;
            bool x_4796;
            bool y_4797;
            bool res_4798;
            bool x_4799;
            bool y_4800;
            bool res_4801;
            bool x_4802;
            bool y_4803;
            bool res_4804;
            int32_t res_4805;
            int32_t res_4810;
            int32_t res_4815;
            int32_t res_4820;
            int32_t res_4821;
            int32_t res_4822;
            
            if (thread_active_5247) {
                cond_4784 = x_4775 == 0;
                res_4785 = x_4781 == 0;
                x_4786 = !cond_4784;
                y_4787 = res_4785 && x_4786;
                cond_4788 = cond_4784 || y_4787;
                cond_4789 = x_4777 == 0;
                res_4790 = x_4782 == 0;
                x_4791 = cond_4789 && res_4790;
                res_4792 = sle32(x_4777, x_4782);
                res_4793 = x_4777 == x_4782;
                x_4794 = cond_4763 && res_4793;
                res_4795 = x_4764 || x_4794;
                x_4796 = cond_4762 && res_4792;
                y_4797 = x_4765 && res_4795;
                res_4798 = x_4796 || y_4797;
                x_4799 = cond_4759 && x_4791;
                y_4800 = x_4760 && res_4798;
                res_4801 = x_4799 || y_4800;
                x_4802 = !cond_4788;
                y_4803 = res_4801 && x_4802;
                res_4804 = cond_4788 || y_4803;
                if (res_4804) {
                    int32_t arg_4806;
                    int32_t res_4807;
                    int32_t res_4808;
                    
                    arg_4806 = x_4774 + x_4779;
                    res_4807 = smax32(x_4772, arg_4806);
                    res_4808 = smax32(x_4778, res_4807);
                    res_4805 = res_4808;
                } else {
                    int32_t res_4809 = smax32(x_4772, x_4778);
                    
                    res_4805 = res_4809;
                }
                if (cond_4784) {
                    res_4810 = x_4779;
                } else {
                    bool cond_4811;
                    bool x_4812;
                    int32_t res_4813;
                    
                    cond_4811 = x_4775 == x_4773;
                    x_4812 = res_4804 && cond_4811;
                    if (x_4812) {
                        int32_t res_4814 = x_4773 + x_4779;
                        
                        res_4813 = res_4814;
                    } else {
                        res_4813 = x_4773;
                    }
                    res_4810 = res_4813;
                }
                if (res_4785) {
                    res_4815 = x_4774;
                } else {
                    bool cond_4816;
                    bool x_4817;
                    int32_t res_4818;
                    
                    cond_4816 = x_4781 == x_4780;
                    x_4817 = res_4804 && cond_4816;
                    if (x_4817) {
                        int32_t res_4819 = x_4774 + x_4780;
                        
                        res_4818 = res_4819;
                    } else {
                        res_4818 = x_4780;
                    }
                    res_4815 = res_4818;
                }
                res_4820 = x_4775 + x_4781;
                if (cond_4784) {
                    res_4821 = x_4782;
                } else {
                    res_4821 = x_4776;
                }
                if (res_4785) {
                    res_4822 = x_4777;
                } else {
                    res_4822 = x_4783;
                }
            }
            x_4772 = res_4805;
            x_4773 = res_4810;
            x_4774 = res_4815;
            x_4775 = res_4820;
            x_4776 = res_4821;
            x_4777 = res_4822;
            *(__local int32_t *) &mem_5176[local_tid_5059 * 4] = x_4772;
            *(__local int32_t *) &mem_5179[local_tid_5059 * 4] = x_4773;
            *(__local int32_t *) &mem_5182[local_tid_5059 * 4] = x_4774;
            *(__local int32_t *) &mem_5185[local_tid_5059 * 4] = x_4775;
            *(__local int32_t *) &mem_5188[local_tid_5059 * 4] = x_4776;
            *(__local int32_t *) &mem_5191[local_tid_5059 * 4] = x_4777;
        }
        skip_waves_5250 *= 2;
    }
    final_result_5081 = x_4772;
    final_result_5082 = x_4773;
    final_result_5083 = x_4774;
    final_result_5084 = x_4775;
    final_result_5085 = x_4776;
    final_result_5086 = x_4777;
    if (local_tid_5059 == 0) {
        *(__global int32_t *) &mem_5194[group_id_5060 * 4] = final_result_5081;
    }
    if (local_tid_5059 == 0) {
        *(__global int32_t *) &mem_5197[group_id_5060 * 4] = final_result_5082;
    }
    if (local_tid_5059 == 0) {
        *(__global int32_t *) &mem_5200[group_id_5060 * 4] = final_result_5083;
    }
    if (local_tid_5059 == 0) {
        *(__global int32_t *) &mem_5203[group_id_5060 * 4] = final_result_5084;
    }
    if (local_tid_5059 == 0) {
        *(__global int32_t *) &mem_5206[group_id_5060 * 4] = final_result_5085;
    }
    if (local_tid_5059 == 0) {
        *(__global int32_t *) &mem_5209[group_id_5060 * 4] = final_result_5086;
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
class testlssp:
  entry_points = {"main": (["i32", "[]i32"], ["i32"])}
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
                                       required_types=["i32", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"group_size_4848": {"class": "group_size", "value": None},
                                        "max_num_groups_4850": {"class": "num_groups", "value": None}})
    self.chunked_reduce_kernel_4870_var = program.chunked_reduce_kernel_4870
    self.fut_kernel_map_transpose_i32_var = program.fut_kernel_map_transpose_i32
    self.fut_kernel_map_transpose_lowheight_i32_var = program.fut_kernel_map_transpose_lowheight_i32
    self.fut_kernel_map_transpose_lowwidth_i32_var = program.fut_kernel_map_transpose_lowwidth_i32
    self.fut_kernel_map_transpose_small_i32_var = program.fut_kernel_map_transpose_small_i32
    self.reduce_kernel_5058_var = program.reduce_kernel_5058
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
  def futhark_main(self, xs_mem_sizze_5126, xs_mem_5127, sizze_4756, pind_4757):
    cond_4759 = (pind_4757 == np.int32(1))
    x_4760 = not(cond_4759)
    cond_4762 = (pind_4757 == np.int32(2))
    cond_4763 = (pind_4757 == np.int32(3))
    x_4764 = not(cond_4763)
    x_4765 = not(cond_4762)
    group_sizze_4849 = self.sizes["group_size_4848"]
    max_num_groups_4851 = self.sizes["max_num_groups_4850"]
    y_4852 = (group_sizze_4849 - np.int32(1))
    x_4853 = (sizze_4756 + y_4852)
    w_div_group_sizze_4854 = squot32(x_4853, group_sizze_4849)
    num_groups_maybe_zzero_4855 = smin32(max_num_groups_4851,
                                         w_div_group_sizze_4854)
    num_groups_4856 = smax32(np.int32(1), num_groups_maybe_zzero_4855)
    num_threads_4857 = (group_sizze_4849 * num_groups_4856)
    y_4858 = (num_threads_4857 - np.int32(1))
    x_4859 = (sizze_4756 + y_4858)
    per_thread_elements_4860 = squot32(x_4859, num_threads_4857)
    y_5104 = smod32(sizze_4756, num_threads_4857)
    x_5105 = (num_threads_4857 - y_5104)
    y_5106 = smod32(x_5105, num_threads_4857)
    padded_sizze_5107 = (sizze_4756 + y_5106)
    per_chunk_5109 = squot32(padded_sizze_5107, num_threads_4857)
    binop_x_5129 = sext_i32_i64(y_5106)
    bytes_5128 = (np.int64(4) * binop_x_5129)
    mem_5130 = opencl_alloc(self, bytes_5128, "mem_5130")
    binop_x_5132 = sext_i32_i64(padded_sizze_5107)
    bytes_5131 = (np.int64(4) * binop_x_5132)
    mem_5133 = opencl_alloc(self, bytes_5131, "mem_5133")
    tmp_offs_5213 = np.int32(0)
    if ((sizze_4756 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_5133, xs_mem_5127,
                      dest_offset=np.long((tmp_offs_5213 * np.int32(4))),
                      src_offset=np.long(np.int32(0)),
                      byte_count=np.long((sizze_4756 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    tmp_offs_5213 = (tmp_offs_5213 + sizze_4756)
    if ((y_5106 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_5133, mem_5130,
                      dest_offset=np.long((tmp_offs_5213 * np.int32(4))),
                      src_offset=np.long(np.int32(0)),
                      byte_count=np.long((y_5106 * np.int32(4))))
    if synchronous:
      self.queue.finish()
    tmp_offs_5213 = (tmp_offs_5213 + y_5106)
    mem_5130 = None
    convop_x_5135 = (num_threads_4857 * per_chunk_5109)
    binop_x_5136 = sext_i32_i64(convop_x_5135)
    bytes_5134 = (np.int64(4) * binop_x_5136)
    mem_5137 = opencl_alloc(self, bytes_5134, "mem_5137")
    self.futhark_map_transpose_opencl_i32(mem_5137, np.int32(0), mem_5133,
                                          np.int32(0), np.int32(1),
                                          per_chunk_5109, num_threads_4857,
                                          (num_threads_4857 * per_chunk_5109),
                                          (num_threads_4857 * per_chunk_5109))
    mem_5133 = None
    binop_x_5157 = sext_i32_i64(num_groups_4856)
    bytes_5156 = (np.int64(4) * binop_x_5157)
    mem_5158 = opencl_alloc(self, bytes_5156, "mem_5158")
    mem_5161 = opencl_alloc(self, bytes_5156, "mem_5161")
    mem_5164 = opencl_alloc(self, bytes_5156, "mem_5164")
    mem_5167 = opencl_alloc(self, bytes_5156, "mem_5167")
    mem_5170 = opencl_alloc(self, bytes_5156, "mem_5170")
    mem_5173 = opencl_alloc(self, bytes_5156, "mem_5173")
    binop_x_5139 = sext_i32_i64(group_sizze_4849)
    bytes_5138 = (np.int64(4) * binop_x_5139)
    if ((1 * (num_groups_4856 * group_sizze_4849)) != 0):
      self.chunked_reduce_kernel_4870_var.set_args(cl.LocalMemory(np.long(bytes_5138)),
                                                   cl.LocalMemory(np.long(bytes_5138)),
                                                   cl.LocalMemory(np.long(bytes_5138)),
                                                   cl.LocalMemory(np.long(bytes_5138)),
                                                   cl.LocalMemory(np.long(bytes_5138)),
                                                   cl.LocalMemory(np.long(bytes_5138)),
                                                   np.int32(sizze_4756),
                                                   np.byte(cond_4759),
                                                   np.byte(x_4760),
                                                   np.byte(cond_4762),
                                                   np.byte(cond_4763),
                                                   np.byte(x_4764),
                                                   np.byte(x_4765),
                                                   np.int32(num_threads_4857),
                                                   np.int32(per_thread_elements_4860),
                                                   np.int32(per_chunk_5109),
                                                   mem_5137, mem_5158, mem_5161,
                                                   mem_5164, mem_5167, mem_5170,
                                                   mem_5173)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.chunked_reduce_kernel_4870_var,
                                 (np.long((num_groups_4856 * group_sizze_4849)),),
                                 (np.long(group_sizze_4849),))
      if synchronous:
        self.queue.finish()
    mem_5137 = None
    mem_5140 = None
    mem_5143 = None
    mem_5146 = None
    mem_5149 = None
    mem_5152 = None
    mem_5155 = None
    mem_5194 = opencl_alloc(self, np.int64(4), "mem_5194")
    mem_5197 = opencl_alloc(self, np.int64(4), "mem_5197")
    mem_5200 = opencl_alloc(self, np.int64(4), "mem_5200")
    mem_5203 = opencl_alloc(self, np.int64(4), "mem_5203")
    mem_5206 = opencl_alloc(self, np.int64(4), "mem_5206")
    mem_5209 = opencl_alloc(self, np.int64(4), "mem_5209")
    binop_x_5175 = sext_i32_i64(max_num_groups_4851)
    bytes_5174 = (np.int64(4) * binop_x_5175)
    if ((1 * max_num_groups_4851) != 0):
      self.reduce_kernel_5058_var.set_args(cl.LocalMemory(np.long(bytes_5174)),
                                           cl.LocalMemory(np.long(bytes_5174)),
                                           cl.LocalMemory(np.long(bytes_5174)),
                                           cl.LocalMemory(np.long(bytes_5174)),
                                           cl.LocalMemory(np.long(bytes_5174)),
                                           cl.LocalMemory(np.long(bytes_5174)),
                                           np.byte(cond_4759), np.byte(x_4760),
                                           np.byte(cond_4762),
                                           np.byte(cond_4763), np.byte(x_4764),
                                           np.byte(x_4765),
                                           np.int32(num_groups_4856), mem_5158,
                                           mem_5161, mem_5164, mem_5167,
                                           mem_5170, mem_5173, mem_5194,
                                           mem_5197, mem_5200, mem_5203,
                                           mem_5206, mem_5209)
      cl.enqueue_nd_range_kernel(self.queue, self.reduce_kernel_5058_var,
                                 (np.long(max_num_groups_4851),),
                                 (np.long(max_num_groups_4851),))
      if synchronous:
        self.queue.finish()
    mem_5158 = None
    mem_5161 = None
    mem_5164 = None
    mem_5167 = None
    mem_5170 = None
    mem_5173 = None
    mem_5176 = None
    mem_5179 = None
    mem_5182 = None
    mem_5185 = None
    mem_5188 = None
    mem_5191 = None
    mem_5197 = None
    mem_5200 = None
    mem_5203 = None
    mem_5206 = None
    mem_5209 = None
    read_res_5258 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_5258, mem_5194,
                    device_offset=np.long(np.int32(0)), is_blocking=True)
    res_4766 = read_res_5258[0]
    mem_5194 = None
    scalar_out_5212 = res_4766
    return scalar_out_5212
  def main(self, pind_4757_ext, xs_mem_5127_ext):
    try:
      pind_4757 = np.int32(ct.c_int32(pind_4757_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(pind_4757_ext),
                                                                                                                            pind_4757_ext))
    try:
      assert ((type(xs_mem_5127_ext) in [np.ndarray,
                                         cl.array.Array]) and (xs_mem_5127_ext.dtype == np.int32)), "Parameter has unexpected type"
      sizze_4756 = np.int32(xs_mem_5127_ext.shape[0])
      xs_mem_sizze_5126 = np.int64(xs_mem_5127_ext.nbytes)
      if (type(xs_mem_5127_ext) == cl.array.Array):
        xs_mem_5127 = xs_mem_5127_ext.data
      else:
        xs_mem_5127 = opencl_alloc(self, xs_mem_sizze_5126, "xs_mem_5127")
        if (xs_mem_sizze_5126 != 0):
          cl.enqueue_copy(self.queue, xs_mem_5127,
                          normaliseArray(xs_mem_5127_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i32",
                                                                                                                            type(xs_mem_5127_ext),
                                                                                                                            xs_mem_5127_ext))
    scalar_out_5212 = self.futhark_main(xs_mem_sizze_5126, xs_mem_5127,
                                        sizze_4756, pind_4757)
    return np.int32(scalar_out_5212)