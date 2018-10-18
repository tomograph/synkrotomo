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
#define group_sizze_10209 (group_size_10208)
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
__kernel void map_kernel_10214(float delta_9897, float scan_start_9899,
                               float scan_step_9901, int32_t res_9904,
                               int32_t range_start_9915, int32_t num_elems_9919,
                               float res_9920, int32_t res_9923, float res_9938,
                               __global unsigned char *mem_10523, __global
                               unsigned char *mem_10526, __global
                               unsigned char *mem_10529, __global
                               unsigned char *mem_10531, __global
                               unsigned char *mem_10534, __global
                               unsigned char *mem_10536, __global
                               unsigned char *mem_10539, __global
                               unsigned char *mem_10542, __global
                               unsigned char *mem_10565, __global
                               unsigned char *mem_10570)
{
    int32_t wave_sizze_10600;
    int32_t group_sizze_10601;
    bool thread_active_10602;
    int32_t gtid_10205;
    int32_t gtid_10206;
    int32_t global_tid_10214;
    int32_t local_tid_10215;
    int32_t group_id_10216;
    
    global_tid_10214 = get_global_id(0);
    local_tid_10215 = get_local_id(0);
    group_sizze_10601 = get_local_size(0);
    wave_sizze_10600 = LOCKSTEP_WIDTH;
    group_id_10216 = get_group_id(0);
    gtid_10205 = squot32(global_tid_10214, num_elems_9919);
    gtid_10206 = global_tid_10214 - squot32(global_tid_10214, num_elems_9919) *
        num_elems_9919;
    thread_active_10602 = slt32(gtid_10205, res_9904) && slt32(gtid_10206,
                                                               num_elems_9919);
    
    float binop_y_10513;
    float binop_y_10514;
    float cmpop_x_10515;
    bool index_primexp_10516;
    bool index_primexp_10512;
    float res_10219;
    float res_10220;
    float res_10221;
    bool cond_10222;
    float res_10223;
    bool x_10224;
    int32_t convop_x_10507;
    float index_primexp_10508;
    float res_10228;
    float res_10229;
    int32_t res_10246;
    float res_10247;
    bool res_10248;
    float res_10249;
    float res_10256;
    float res_10257;
    bool res_10280;
    bool x_10281;
    bool cond_10282;
    bool res_10283;
    bool x_10284;
    bool cond_10285;
    bool res_10286;
    bool x_10287;
    bool x_10288;
    bool y_10289;
    bool res_10290;
    bool x_10291;
    float y_10292;
    float arg_10293;
    bool res_10294;
    float res_10297;
    float res_10298;
    float res_10299;
    float res_10300;
    int32_t res_10301;
    
    if (thread_active_10602) {
        binop_y_10513 = sitofp_i32_f32(gtid_10205);
        binop_y_10514 = scan_step_9901 * binop_y_10513;
        cmpop_x_10515 = scan_start_9899 + binop_y_10514;
        index_primexp_10516 = cmpop_x_10515 == 90.0F;
        index_primexp_10512 = cmpop_x_10515 == 0.0F;
        res_10219 = *(__global float *) &mem_10523[gtid_10205 * 4];
        res_10220 = *(__global float *) &mem_10526[gtid_10205 * 4];
        res_10221 = *(__global float *) &mem_10529[gtid_10205 * 4];
        cond_10222 = *(__global bool *) &mem_10531[gtid_10205];
        res_10223 = *(__global float *) &mem_10534[gtid_10205 * 4];
        x_10224 = *(__global bool *) &mem_10536[gtid_10205];
        convop_x_10507 = range_start_9915 + gtid_10206;
        index_primexp_10508 = sitofp_i32_f32(convop_x_10507);
        for (int32_t i_10603 = 0; i_10603 < res_9923; i_10603++) {
            *(__global float *) &mem_10539[(group_id_10216 * (res_9923 *
                                                              group_sizze_10209) +
                                            i_10603 * group_sizze_10209 +
                                            local_tid_10215) * 4] = -1.0F;
        }
        for (int32_t i_10604 = 0; i_10604 < res_9923; i_10604++) {
            *(__global int32_t *) &mem_10542[(group_id_10216 * (res_9923 *
                                                                group_sizze_10209) +
                                              i_10604 * group_sizze_10209 +
                                              local_tid_10215) * 4] = -1;
        }
        if (index_primexp_10512) {
            float y_10230;
            float res_10231;
            
            y_10230 = delta_9897 * index_primexp_10508;
            res_10231 = res_9938 + y_10230;
            res_10228 = res_10231;
            res_10229 = res_9920;
        } else {
            float x_10232;
            float arg_10233;
            bool cond_10234;
            bool res_10235;
            bool x_10236;
            float res_10237;
            float res_10238;
            
            x_10232 = res_10220 * index_primexp_10508;
            arg_10233 = res_10221 + x_10232;
            cond_10234 = 0.0F <= arg_10233;
            res_10235 = arg_10233 <= res_9920;
            x_10236 = cond_10234 && res_10235;
            if (x_10236) {
                res_10237 = 0.0F;
                res_10238 = arg_10233;
            } else {
                float x_10239;
                float res_10240;
                float x_10241;
                float res_10242;
                bool cond_10243;
                float res_10244;
                float res_10245;
                
                x_10239 = res_9920 - arg_10233;
                res_10240 = x_10239 / res_10219;
                x_10241 = 0.0F - arg_10233;
                res_10242 = x_10241 / res_10219;
                cond_10243 = res_10240 < res_10242;
                if (cond_10243) {
                    res_10244 = res_10240;
                } else {
                    res_10244 = res_10242;
                }
                if (cond_10243) {
                    res_10245 = res_9920;
                } else {
                    res_10245 = 0.0F;
                }
                res_10237 = res_10244;
                res_10238 = res_10245;
            }
            res_10228 = res_10237;
            res_10229 = res_10238;
        }
        res_10246 = fptosi_f32_i32(res_10228);
        res_10247 = sitofp_i32_f32(res_10246);
        res_10248 = 0.0F <= res_10228;
        if (res_10248) {
            bool res_10250;
            float res_10251;
            
            res_10250 = res_10247 < res_10228;
            if (res_10250) {
                res_10251 = res_10247;
            } else {
                res_10251 = res_10228;
            }
            res_10249 = res_10251;
        } else {
            bool res_10252;
            float res_10253;
            
            res_10252 = res_10228 < res_10247;
            if (res_10252) {
                int32_t res_10254;
                float res_10255;
                
                res_10254 = res_10246 - 1;
                res_10255 = sitofp_i32_f32(res_10254);
                res_10253 = res_10255;
            } else {
                res_10253 = res_10228;
            }
            res_10249 = res_10253;
        }
        res_10256 = 1.0F + res_10249;
        if (cond_10222) {
            int32_t res_10258;
            float res_10259;
            bool res_10260;
            float res_10261;
            float res_10268;
            
            res_10258 = fptosi_f32_i32(res_10229);
            res_10259 = sitofp_i32_f32(res_10258);
            res_10260 = 0.0F <= res_10229;
            if (res_10260) {
                bool res_10262;
                float res_10263;
                
                res_10262 = res_10259 < res_10229;
                if (res_10262) {
                    int32_t res_10264;
                    float res_10265;
                    
                    res_10264 = 1 + res_10258;
                    res_10265 = sitofp_i32_f32(res_10264);
                    res_10263 = res_10265;
                } else {
                    res_10263 = res_10229;
                }
                res_10261 = res_10263;
            } else {
                bool res_10266;
                float res_10267;
                
                res_10266 = res_10229 < res_10259;
                if (res_10266) {
                    res_10267 = res_10259;
                } else {
                    res_10267 = res_10229;
                }
                res_10261 = res_10267;
            }
            res_10268 = res_10261 - 1.0F;
            res_10257 = res_10268;
        } else {
            int32_t res_10269;
            float res_10270;
            bool res_10271;
            float res_10272;
            float res_10279;
            
            res_10269 = fptosi_f32_i32(res_10229);
            res_10270 = sitofp_i32_f32(res_10269);
            res_10271 = 0.0F <= res_10229;
            if (res_10271) {
                bool res_10273;
                float res_10274;
                
                res_10273 = res_10270 < res_10229;
                if (res_10273) {
                    res_10274 = res_10270;
                } else {
                    res_10274 = res_10229;
                }
                res_10272 = res_10274;
            } else {
                bool res_10275;
                float res_10276;
                
                res_10275 = res_10229 < res_10270;
                if (res_10275) {
                    int32_t res_10277;
                    float res_10278;
                    
                    res_10277 = res_10269 - 1;
                    res_10278 = sitofp_i32_f32(res_10277);
                    res_10276 = res_10278;
                } else {
                    res_10276 = res_10229;
                }
                res_10272 = res_10276;
            }
            res_10279 = 1.0F + res_10272;
            res_10257 = res_10279;
        }
        res_10280 = res_10228 < res_9920;
        x_10281 = res_10248 && res_10280;
        cond_10282 = 0.0F < res_10229;
        res_10283 = res_10229 <= res_9920;
        x_10284 = cond_10282 && res_10283;
        cond_10285 = 0.0F <= res_10229;
        res_10286 = res_10229 < res_9920;
        x_10287 = cond_10285 && res_10286;
        x_10288 = cond_10222 && x_10284;
        y_10289 = x_10224 && x_10287;
        res_10290 = x_10288 || y_10289;
        x_10291 = x_10281 && res_10290;
        y_10292 = res_10220 * index_primexp_10508;
        arg_10293 = res_10221 + y_10292;
        
        bool loop_while_10302;
        float focusPoint_10305;
        float focusPoint_10306;
        float anchorX_10307;
        float anchorY_10308;
        int32_t write_index_10309;
        
        loop_while_10302 = x_10291;
        focusPoint_10305 = res_10228;
        focusPoint_10306 = res_10229;
        anchorX_10307 = res_10256;
        anchorY_10308 = res_10257;
        write_index_10309 = 0;
        while (loop_while_10302) {
            float x_10310 = res_10219 * anchorX_10307;
            float res_10311 = arg_10293 + x_10310;
            float res_10312;
            
            if (index_primexp_10512) {
                res_10312 = res_10228;
            } else {
                float x_10313;
                float res_10314;
                
                x_10313 = anchorY_10308 - arg_10293;
                res_10314 = x_10313 / res_10219;
                res_10312 = res_10314;
            }
            
            float x_10315 = anchorX_10307 - focusPoint_10305;
            float x_10316 = fpow32(x_10315, 2.0F);
            float x_10317 = res_10311 - focusPoint_10306;
            float y_10318 = fpow32(x_10317, 2.0F);
            float arg_10319 = x_10316 + y_10318;
            float res_10320;
            
            res_10320 = futrts_sqrt32(arg_10319);
            
            float x_10321 = res_10312 - focusPoint_10305;
            float x_10322 = fpow32(x_10321, 2.0F);
            float x_10323 = anchorY_10308 - focusPoint_10306;
            float y_10324 = fpow32(x_10323, 2.0F);
            float arg_10325 = x_10322 + y_10324;
            float res_10326;
            
            res_10326 = futrts_sqrt32(arg_10325);
            
            int32_t res_10327 = fptosi_f32_i32(focusPoint_10306);
            float res_10328 = sitofp_i32_f32(res_10327);
            bool res_10329 = 0.0F <= focusPoint_10306;
            float res_10330;
            
            if (res_10329) {
                bool res_10331;
                float res_10332;
                
                res_10331 = res_10328 < focusPoint_10306;
                if (res_10331) {
                    res_10332 = res_10328;
                } else {
                    res_10332 = focusPoint_10306;
                }
                res_10330 = res_10332;
            } else {
                bool res_10333;
                float res_10334;
                
                res_10333 = focusPoint_10306 < res_10328;
                if (res_10333) {
                    int32_t res_10335;
                    float res_10336;
                    
                    res_10335 = res_10327 - 1;
                    res_10336 = sitofp_i32_f32(res_10335);
                    res_10334 = res_10336;
                } else {
                    res_10334 = focusPoint_10306;
                }
                res_10330 = res_10334;
            }
            
            float x_10337 = focusPoint_10306 - res_10330;
            bool res_10338 = x_10337 == 0.0F;
            bool x_10339 = cond_10222 && res_10338;
            float res_10340;
            
            if (x_10339) {
                float res_10341 = focusPoint_10306 - 1.0F;
                
                res_10340 = res_10341;
            } else {
                res_10340 = res_10330;
            }
            
            int32_t res_10342 = fptosi_f32_i32(focusPoint_10305);
            float res_10343 = sitofp_i32_f32(res_10342);
            bool res_10344 = 0.0F <= focusPoint_10305;
            float res_10345;
            
            if (res_10344) {
                bool res_10346;
                float res_10347;
                
                res_10346 = res_10343 < focusPoint_10305;
                if (res_10346) {
                    res_10347 = res_10343;
                } else {
                    res_10347 = focusPoint_10305;
                }
                res_10345 = res_10347;
            } else {
                bool res_10348;
                float res_10349;
                
                res_10348 = focusPoint_10305 < res_10343;
                if (res_10348) {
                    int32_t res_10350;
                    float res_10351;
                    
                    res_10350 = res_10342 - 1;
                    res_10351 = sitofp_i32_f32(res_10350);
                    res_10349 = res_10351;
                } else {
                    res_10349 = focusPoint_10305;
                }
                res_10345 = res_10349;
            }
            
            float y_10352 = res_9920 * res_10340;
            float arg_10353 = res_10345 + y_10352;
            int32_t res_10354 = fptosi_f32_i32(arg_10353);
            float res_10357;
            float res_10358;
            float res_10359;
            float res_10360;
            int32_t res_10361;
            
            if (index_primexp_10516) {
                float res_10364;
                int32_t res_10365;
                
                *(__global float *) &mem_10539[(group_id_10216 * (res_9923 *
                                                                  group_sizze_10209) +
                                                write_index_10309 *
                                                group_sizze_10209 +
                                                local_tid_10215) * 4] =
                    res_10320;
                *(__global int32_t *) &mem_10542[(group_id_10216 * (res_9923 *
                                                                    group_sizze_10209) +
                                                  write_index_10309 *
                                                  group_sizze_10209 +
                                                  local_tid_10215) * 4] =
                    res_10354;
                res_10364 = 1.0F + anchorX_10307;
                res_10365 = 1 + write_index_10309;
                res_10357 = anchorX_10307;
                res_10358 = res_10311;
                res_10359 = res_10364;
                res_10360 = anchorY_10308;
                res_10361 = res_10365;
            } else {
                float res_10368;
                float res_10369;
                float res_10370;
                float res_10371;
                int32_t res_10372;
                
                if (index_primexp_10512) {
                    float res_10375;
                    int32_t res_10376;
                    
                    *(__global float *) &mem_10539[(group_id_10216 * (res_9923 *
                                                                      group_sizze_10209) +
                                                    write_index_10309 *
                                                    group_sizze_10209 +
                                                    local_tid_10215) * 4] =
                        res_10326;
                    *(__global int32_t *) &mem_10542[(group_id_10216 *
                                                      (res_9923 *
                                                       group_sizze_10209) +
                                                      write_index_10309 *
                                                      group_sizze_10209 +
                                                      local_tid_10215) * 4] =
                        res_10354;
                    res_10375 = res_10223 + anchorY_10308;
                    res_10376 = 1 + write_index_10309;
                    res_10368 = res_10312;
                    res_10369 = anchorY_10308;
                    res_10370 = anchorX_10307;
                    res_10371 = res_10375;
                    res_10372 = res_10376;
                } else {
                    float arg_10377;
                    float res_10378;
                    bool cond_10379;
                    float res_10382;
                    float res_10383;
                    float res_10384;
                    float res_10385;
                    int32_t res_10386;
                    
                    arg_10377 = res_10320 - res_10326;
                    res_10378 = (float) fabs(arg_10377);
                    cond_10379 = 1.0e-9F < res_10378;
                    if (cond_10379) {
                        bool cond_10387;
                        float res_10388;
                        float res_10389;
                        float res_10392;
                        float res_10393;
                        int32_t res_10394;
                        
                        cond_10387 = res_10320 < res_10326;
                        if (cond_10387) {
                            res_10388 = anchorX_10307;
                        } else {
                            res_10388 = res_10312;
                        }
                        if (cond_10387) {
                            res_10389 = res_10311;
                        } else {
                            res_10389 = anchorY_10308;
                        }
                        if (cond_10387) {
                            float res_10397;
                            int32_t res_10398;
                            
                            *(__global float *) &mem_10539[(group_id_10216 *
                                                            (res_9923 *
                                                             group_sizze_10209) +
                                                            write_index_10309 *
                                                            group_sizze_10209 +
                                                            local_tid_10215) *
                                                           4] = res_10320;
                            *(__global int32_t *) &mem_10542[(group_id_10216 *
                                                              (res_9923 *
                                                               group_sizze_10209) +
                                                              write_index_10309 *
                                                              group_sizze_10209 +
                                                              local_tid_10215) *
                                                             4] = res_10354;
                            res_10397 = 1.0F + anchorX_10307;
                            res_10398 = 1 + write_index_10309;
                            res_10392 = res_10397;
                            res_10393 = anchorY_10308;
                            res_10394 = res_10398;
                        } else {
                            float res_10401;
                            int32_t res_10402;
                            
                            *(__global float *) &mem_10539[(group_id_10216 *
                                                            (res_9923 *
                                                             group_sizze_10209) +
                                                            write_index_10309 *
                                                            group_sizze_10209 +
                                                            local_tid_10215) *
                                                           4] = res_10326;
                            *(__global int32_t *) &mem_10542[(group_id_10216 *
                                                              (res_9923 *
                                                               group_sizze_10209) +
                                                              write_index_10309 *
                                                              group_sizze_10209 +
                                                              local_tid_10215) *
                                                             4] = res_10354;
                            res_10401 = res_10223 + anchorY_10308;
                            res_10402 = 1 + write_index_10309;
                            res_10392 = anchorX_10307;
                            res_10393 = res_10401;
                            res_10394 = res_10402;
                        }
                        res_10382 = res_10388;
                        res_10383 = res_10389;
                        res_10384 = res_10392;
                        res_10385 = res_10393;
                        res_10386 = res_10394;
                    } else {
                        float res_10405;
                        float res_10406;
                        int32_t res_10407;
                        
                        *(__global float *) &mem_10539[(group_id_10216 *
                                                        (res_9923 *
                                                         group_sizze_10209) +
                                                        write_index_10309 *
                                                        group_sizze_10209 +
                                                        local_tid_10215) * 4] =
                            res_10320;
                        *(__global int32_t *) &mem_10542[(group_id_10216 *
                                                          (res_9923 *
                                                           group_sizze_10209) +
                                                          write_index_10309 *
                                                          group_sizze_10209 +
                                                          local_tid_10215) *
                                                         4] = res_10354;
                        res_10405 = 1.0F + anchorX_10307;
                        res_10406 = res_10223 + anchorY_10308;
                        res_10407 = 1 + write_index_10309;
                        res_10382 = anchorX_10307;
                        res_10383 = res_10311;
                        res_10384 = res_10405;
                        res_10385 = res_10406;
                        res_10386 = res_10407;
                    }
                    res_10368 = res_10382;
                    res_10369 = res_10383;
                    res_10370 = res_10384;
                    res_10371 = res_10385;
                    res_10372 = res_10386;
                }
                res_10357 = res_10368;
                res_10358 = res_10369;
                res_10359 = res_10370;
                res_10360 = res_10371;
                res_10361 = res_10372;
            }
            
            bool cond_10408 = 0.0F <= res_10357;
            bool res_10409 = res_10357 < res_9920;
            bool x_10410 = cond_10408 && res_10409;
            bool cond_10411 = 0.0F < res_10358;
            bool res_10412 = res_10358 <= res_9920;
            bool x_10413 = cond_10411 && res_10412;
            bool cond_10414 = 0.0F <= res_10358;
            bool res_10415 = res_10358 < res_9920;
            bool x_10416 = cond_10414 && res_10415;
            bool x_10417 = cond_10222 && x_10413;
            bool y_10418 = x_10224 && x_10416;
            bool res_10419 = x_10417 || y_10418;
            bool x_10420 = x_10410 && res_10419;
            bool loop_while_tmp_10605 = x_10420;
            float focusPoint_tmp_10608 = res_10357;
            float focusPoint_tmp_10609 = res_10358;
            float anchorX_tmp_10610 = res_10359;
            float anchorY_tmp_10611 = res_10360;
            int32_t write_index_tmp_10612;
            
            write_index_tmp_10612 = res_10361;
            loop_while_10302 = loop_while_tmp_10605;
            focusPoint_10305 = focusPoint_tmp_10608;
            focusPoint_10306 = focusPoint_tmp_10609;
            anchorX_10307 = anchorX_tmp_10610;
            anchorY_10308 = anchorY_tmp_10611;
            write_index_10309 = write_index_tmp_10612;
        }
        res_10294 = loop_while_10302;
        res_10297 = focusPoint_10305;
        res_10298 = focusPoint_10306;
        res_10299 = anchorX_10307;
        res_10300 = anchorY_10308;
        res_10301 = write_index_10309;
    }
    if (thread_active_10602) {
        for (int32_t i_10613 = 0; i_10613 < res_9923; i_10613++) {
            *(__global float *) &mem_10565[(i_10613 * (res_9904 *
                                                       num_elems_9919) +
                                            gtid_10205 * num_elems_9919 +
                                            gtid_10206) * 4] = *(__global
                                                                 float *) &mem_10539[(group_id_10216 *
                                                                                      (res_9923 *
                                                                                       group_sizze_10209) +
                                                                                      i_10613 *
                                                                                      group_sizze_10209 +
                                                                                      local_tid_10215) *
                                                                                     4];
        }
    }
    if (thread_active_10602) {
        for (int32_t i_10614 = 0; i_10614 < res_9923; i_10614++) {
            *(__global int32_t *) &mem_10570[(i_10614 * (res_9904 *
                                                         num_elems_9919) +
                                              gtid_10205 * num_elems_9919 +
                                              gtid_10206) * 4] = *(__global
                                                                   int32_t *) &mem_10542[(group_id_10216 *
                                                                                          (res_9923 *
                                                                                           group_sizze_10209) +
                                                                                          i_10614 *
                                                                                          group_sizze_10209 +
                                                                                          local_tid_10215) *
                                                                                         4];
        }
    }
}
__kernel void map_kernel_10473(float delta_9897, float scan_start_9899,
                               float scan_step_9901, int32_t res_9904,
                               float res_9938, __global
                               unsigned char *mem_10518, __global
                               unsigned char *mem_10520, __global
                               unsigned char *mem_10523, __global
                               unsigned char *mem_10526, __global
                               unsigned char *mem_10529, __global
                               unsigned char *mem_10531, __global
                               unsigned char *mem_10534, __global
                               unsigned char *mem_10536)
{
    int32_t wave_sizze_10597;
    int32_t group_sizze_10598;
    bool thread_active_10599;
    int32_t gtid_10466;
    int32_t global_tid_10473;
    int32_t local_tid_10474;
    int32_t group_id_10475;
    
    global_tid_10473 = get_global_id(0);
    local_tid_10474 = get_local_id(0);
    group_sizze_10598 = get_local_size(0);
    wave_sizze_10597 = LOCKSTEP_WIDTH;
    group_id_10475 = get_group_id(0);
    gtid_10466 = global_tid_10473;
    thread_active_10599 = slt32(gtid_10466, res_9904);
    
    float res_10477;
    float y_10478;
    float res_10479;
    bool res_10480;
    bool res_10481;
    float arg_10482;
    float res_10483;
    float res_10484;
    float res_10485;
    float res_10486;
    float arg_10487;
    float res_10488;
    float res_10489;
    float res_10490;
    float res_10491;
    bool cond_10495;
    float res_10496;
    bool x_10497;
    
    if (thread_active_10599) {
        res_10477 = sitofp_i32_f32(gtid_10466);
        y_10478 = scan_step_9901 * res_10477;
        res_10479 = scan_start_9899 + y_10478;
        res_10480 = res_10479 == 90.0F;
        res_10481 = res_10479 == 0.0F;
        arg_10482 = 90.0F + res_10479;
        res_10483 = 1.7453292e-2F * arg_10482;
        res_10484 = futrts_sin32(res_10483);
        res_10485 = futrts_cos32(res_10483);
        res_10486 = res_10484 / res_10485;
        arg_10487 = 90.0F - res_10479;
        res_10488 = 1.7453292e-2F * arg_10487;
        res_10489 = futrts_cos32(res_10488);
        res_10490 = delta_9897 / res_10489;
        if (res_10480) {
            float res_10492 = res_9938 - 1.0F;
            
            res_10491 = res_10492;
        } else {
            float y_10493;
            float res_10494;
            
            y_10493 = res_9938 * res_10486;
            res_10494 = res_9938 - y_10493;
            res_10491 = res_10494;
        }
        cond_10495 = res_10486 < 0.0F;
        if (cond_10495) {
            res_10496 = -1.0F;
        } else {
            res_10496 = 1.0F;
        }
        x_10497 = !cond_10495;
    }
    if (thread_active_10599) {
        *(__global bool *) &mem_10518[gtid_10466] = res_10480;
    }
    if (thread_active_10599) {
        *(__global bool *) &mem_10520[gtid_10466] = res_10481;
    }
    if (thread_active_10599) {
        *(__global float *) &mem_10523[gtid_10466 * 4] = res_10486;
    }
    if (thread_active_10599) {
        *(__global float *) &mem_10526[gtid_10466 * 4] = res_10490;
    }
    if (thread_active_10599) {
        *(__global float *) &mem_10529[gtid_10466 * 4] = res_10491;
    }
    if (thread_active_10599) {
        *(__global bool *) &mem_10531[gtid_10466] = cond_10495;
    }
    if (thread_active_10599) {
        *(__global float *) &mem_10534[gtid_10466 * 4] = res_10496;
    }
    if (thread_active_10599) {
        *(__global bool *) &mem_10536[gtid_10466] = x_10497;
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
class intersections:
  entry_points = {"main": (["i32", "f32", "i32", "f32", "f32", "f32"],
                           ["[][][]f32", "[][][]i32"])}
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
                                       all_sizes={"group_size_10208": {"class": "group_size", "value": None},
                                        "group_size_10467": {"class": "group_size", "value": None}})
    self.fut_kernel_map_transpose_f32_var = program.fut_kernel_map_transpose_f32
    self.fut_kernel_map_transpose_i32_var = program.fut_kernel_map_transpose_i32
    self.fut_kernel_map_transpose_lowheight_f32_var = program.fut_kernel_map_transpose_lowheight_f32
    self.fut_kernel_map_transpose_lowheight_i32_var = program.fut_kernel_map_transpose_lowheight_i32
    self.fut_kernel_map_transpose_lowwidth_f32_var = program.fut_kernel_map_transpose_lowwidth_f32
    self.fut_kernel_map_transpose_lowwidth_i32_var = program.fut_kernel_map_transpose_lowwidth_i32
    self.fut_kernel_map_transpose_small_f32_var = program.fut_kernel_map_transpose_small_f32
    self.fut_kernel_map_transpose_small_i32_var = program.fut_kernel_map_transpose_small_i32
    self.map_kernel_10214_var = program.map_kernel_10214
    self.map_kernel_10473_var = program.map_kernel_10473
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
  def futhark_main(self, grid_sizze_9896, delta_9897, line_count_9898,
                   scan_start_9899, scan_end_9900, scan_step_9901):
    x_9902 = (scan_end_9900 - scan_start_9899)
    arg_9903 = (x_9902 / scan_step_9901)
    res_9904 = fptosi_f32_i32(arg_9903)
    bounds_invalid_upwards_9905 = slt32(res_9904, np.int32(0))
    eq_x_zz_9908 = (np.int32(0) == res_9904)
    not_p_9909 = not(bounds_invalid_upwards_9905)
    p_and_eq_x_y_9910 = (eq_x_zz_9908 and not_p_9909)
    dim_zzero_9911 = (bounds_invalid_upwards_9905 or p_and_eq_x_y_9910)
    both_empty_9912 = (eq_x_zz_9908 and dim_zzero_9911)
    empty_or_match_9913 = (not_p_9909 or both_empty_9912)
    empty_or_match_cert_9914 = True
    assert empty_or_match_9913, ("Error at intersections.fut:11:1-23:13 -> intersections.fut:20:61-20:76 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                           "*",
                                                                                                                                                           "[",
                                                                                                                                                           res_9904,
                                                                                                                                                           "]",
                                                                                                                                                           "intrinsics.i32"))
    range_start_9915 = (np.int32(0) - line_count_9898)
    bounds_invalid_upwards_9916 = slt32(line_count_9898, range_start_9915)
    distance_upwards_exclusive_9917 = (line_count_9898 - range_start_9915)
    distance_9918 = (np.int32(1) + distance_upwards_exclusive_9917)
    if bounds_invalid_upwards_9916:
      num_elems_9919 = np.int32(0)
    else:
      num_elems_9919 = distance_9918
    res_9920 = sitofp_i32_f32(grid_sizze_9896)
    x_9921 = (np.float32(2.0) * res_9920)
    arg_9922 = (x_9921 - np.float32(1.0))
    res_9923 = fptosi_f32_i32(arg_9922)
    res_9938 = (res_9920 / np.float32(2.0))
    group_sizze_10468 = self.sizes["group_size_10467"]
    y_10469 = (group_sizze_10468 - np.int32(1))
    x_10470 = (res_9904 + y_10469)
    num_groups_10471 = squot32(x_10470, group_sizze_10468)
    num_threads_10472 = (group_sizze_10468 * num_groups_10471)
    bytes_10517 = sext_i32_i64(res_9904)
    mem_10518 = opencl_alloc(self, bytes_10517, "mem_10518")
    mem_10520 = opencl_alloc(self, bytes_10517, "mem_10520")
    bytes_10521 = (np.int64(4) * bytes_10517)
    mem_10523 = opencl_alloc(self, bytes_10521, "mem_10523")
    mem_10526 = opencl_alloc(self, bytes_10521, "mem_10526")
    mem_10529 = opencl_alloc(self, bytes_10521, "mem_10529")
    mem_10531 = opencl_alloc(self, bytes_10517, "mem_10531")
    mem_10534 = opencl_alloc(self, bytes_10521, "mem_10534")
    mem_10536 = opencl_alloc(self, bytes_10517, "mem_10536")
    if ((1 * (num_groups_10471 * group_sizze_10468)) != 0):
      self.map_kernel_10473_var.set_args(np.float32(delta_9897),
                                         np.float32(scan_start_9899),
                                         np.float32(scan_step_9901),
                                         np.int32(res_9904),
                                         np.float32(res_9938), mem_10518,
                                         mem_10520, mem_10523, mem_10526,
                                         mem_10529, mem_10531, mem_10534,
                                         mem_10536)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10473_var,
                                 (np.long((num_groups_10471 * group_sizze_10468)),),
                                 (np.long(group_sizze_10468),))
      if synchronous:
        self.queue.finish()
    mem_10518 = None
    mem_10520 = None
    nesting_sizze_10207 = (res_9904 * num_elems_9919)
    group_sizze_10209 = self.sizes["group_size_10208"]
    y_10210 = (group_sizze_10209 - np.int32(1))
    x_10211 = (nesting_sizze_10207 + y_10210)
    num_groups_10212 = squot32(x_10211, group_sizze_10209)
    num_threads_10213 = (group_sizze_10209 * num_groups_10212)
    binop_x_10562 = (res_9904 * res_9923)
    convop_x_10563 = (num_elems_9919 * binop_x_10562)
    binop_x_10564 = sext_i32_i64(convop_x_10563)
    bytes_10561 = (np.int64(4) * binop_x_10564)
    mem_10565 = opencl_alloc(self, bytes_10561, "mem_10565")
    mem_10570 = opencl_alloc(self, bytes_10561, "mem_10570")
    binop_x_10538 = sext_i32_i64(res_9923)
    bytes_10537 = (np.int64(4) * binop_x_10538)
    num_threads64_10584 = sext_i32_i64(num_threads_10213)
    total_sizze_10585 = (bytes_10537 * num_threads64_10584)
    mem_10539 = opencl_alloc(self, total_sizze_10585, "mem_10539")
    total_sizze_10586 = (bytes_10537 * num_threads64_10584)
    mem_10542 = opencl_alloc(self, total_sizze_10586, "mem_10542")
    if ((1 * (num_groups_10212 * group_sizze_10209)) != 0):
      self.map_kernel_10214_var.set_args(np.float32(delta_9897),
                                         np.float32(scan_start_9899),
                                         np.float32(scan_step_9901),
                                         np.int32(res_9904),
                                         np.int32(range_start_9915),
                                         np.int32(num_elems_9919),
                                         np.float32(res_9920),
                                         np.int32(res_9923),
                                         np.float32(res_9938), mem_10523,
                                         mem_10526, mem_10529, mem_10531,
                                         mem_10534, mem_10536, mem_10539,
                                         mem_10542, mem_10565, mem_10570)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_10214_var,
                                 (np.long((num_groups_10212 * group_sizze_10209)),),
                                 (np.long(group_sizze_10209),))
      if synchronous:
        self.queue.finish()
    mem_10523 = None
    mem_10526 = None
    mem_10529 = None
    mem_10531 = None
    mem_10534 = None
    mem_10536 = None
    mem_10539 = None
    mem_10542 = None
    convop_x_10573 = (res_9923 * nesting_sizze_10207)
    binop_x_10574 = sext_i32_i64(convop_x_10573)
    bytes_10571 = (np.int64(4) * binop_x_10574)
    mem_10575 = opencl_alloc(self, bytes_10571, "mem_10575")
    self.futhark_map_transpose_opencl_f32(mem_10575, np.int32(0), mem_10565,
                                          np.int32(0), np.int32(1),
                                          (res_9904 * num_elems_9919), res_9923,
                                          ((res_9904 * num_elems_9919) * res_9923),
                                          ((res_9904 * num_elems_9919) * res_9923))
    mem_10565 = None
    mem_10581 = opencl_alloc(self, bytes_10571, "mem_10581")
    self.futhark_map_transpose_opencl_i32(mem_10581, np.int32(0), mem_10570,
                                          np.int32(0), np.int32(1),
                                          (res_9904 * num_elems_9919), res_9923,
                                          ((res_9904 * num_elems_9919) * res_9923),
                                          ((res_9904 * num_elems_9919) * res_9923))
    mem_10570 = None
    out_arrsizze_10589 = res_9904
    out_arrsizze_10590 = num_elems_9919
    out_arrsizze_10591 = res_9923
    out_arrsizze_10594 = res_9904
    out_arrsizze_10595 = num_elems_9919
    out_arrsizze_10596 = res_9923
    out_memsizze_10588 = bytes_10571
    out_mem_10587 = mem_10575
    out_memsizze_10593 = bytes_10571
    out_mem_10592 = mem_10581
    return (out_memsizze_10588, out_mem_10587, out_arrsizze_10589,
            out_arrsizze_10590, out_arrsizze_10591, out_memsizze_10593,
            out_mem_10592, out_arrsizze_10594, out_arrsizze_10595,
            out_arrsizze_10596)
  def main(self, grid_sizze_9896_ext, delta_9897_ext, line_count_9898_ext,
           scan_start_9899_ext, scan_end_9900_ext, scan_step_9901_ext):
    try:
      grid_sizze_9896 = np.int32(ct.c_int32(grid_sizze_9896_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(grid_sizze_9896_ext),
                                                                                                                            grid_sizze_9896_ext))
    try:
      delta_9897 = np.float32(ct.c_float(delta_9897_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(delta_9897_ext),
                                                                                                                            delta_9897_ext))
    try:
      line_count_9898 = np.int32(ct.c_int32(line_count_9898_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(line_count_9898_ext),
                                                                                                                            line_count_9898_ext))
    try:
      scan_start_9899 = np.float32(ct.c_float(scan_start_9899_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(scan_start_9899_ext),
                                                                                                                            scan_start_9899_ext))
    try:
      scan_end_9900 = np.float32(ct.c_float(scan_end_9900_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(scan_end_9900_ext),
                                                                                                                            scan_end_9900_ext))
    try:
      scan_step_9901 = np.float32(ct.c_float(scan_step_9901_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(scan_step_9901_ext),
                                                                                                                            scan_step_9901_ext))
    (out_memsizze_10588, out_mem_10587, out_arrsizze_10589, out_arrsizze_10590,
     out_arrsizze_10591, out_memsizze_10593, out_mem_10592, out_arrsizze_10594,
     out_arrsizze_10595,
     out_arrsizze_10596) = self.futhark_main(grid_sizze_9896, delta_9897,
                                             line_count_9898, scan_start_9899,
                                             scan_end_9900, scan_step_9901)
    return (cl.array.Array(self.queue, (out_arrsizze_10589, out_arrsizze_10590,
                                        out_arrsizze_10591), ct.c_float,
                           data=out_mem_10587), cl.array.Array(self.queue,
                                                               (out_arrsizze_10594,
                                                                out_arrsizze_10595,
                                                                out_arrsizze_10596),
                                                               ct.c_int32,
                                                               data=out_mem_10592))