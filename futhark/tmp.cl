#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
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
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
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
__kernel void map_4742(int32_t sizze_4697, int32_t numrhos_4699, __global
                       unsigned char *mem_4823, __global
                       unsigned char *mem_4827)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_4857;
    int32_t group_sizze_4858;
    int32_t gtid_4733;
    int32_t gtid_4734;
    int32_t global_tid_4742;
    int32_t local_tid_4743;
    int32_t group_id_4744;
    
    global_tid_4742 = get_global_id(0);
    local_tid_4743 = get_local_id(0);
    group_sizze_4858 = get_local_size(0);
    wave_sizze_4857 = LOCKSTEP_WIDTH;
    group_id_4744 = get_group_id(0);
    gtid_4733 = squot32(global_tid_4742, numrhos_4699);
    gtid_4734 = global_tid_4742 - squot32(global_tid_4742, numrhos_4699) *
        numrhos_4699;
    
    float res_4745;
    
    if (slt32(gtid_4733, sizze_4697) && slt32(gtid_4734, numrhos_4699)) {
        res_4745 = *(__global float *) &mem_4823[gtid_4733 * 4];
    }
    if (slt32(gtid_4733, sizze_4697) && slt32(gtid_4734, numrhos_4699)) {
        *(__global float *) &mem_4827[(gtid_4733 * numrhos_4699 + gtid_4734) *
                                      4] = res_4745;
    }
}
__kernel void map_4754(int32_t sizze_4697, __global
                       unsigned char *angles_mem_4820, __global
                       unsigned char *mem_4823)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_4855;
    int32_t group_sizze_4856;
    int32_t gtid_4747;
    int32_t global_tid_4754;
    int32_t local_tid_4755;
    int32_t group_id_4756;
    
    global_tid_4754 = get_global_id(0);
    local_tid_4755 = get_local_id(0);
    group_sizze_4856 = get_local_size(0);
    wave_sizze_4855 = LOCKSTEP_WIDTH;
    group_id_4756 = get_group_id(0);
    gtid_4747 = global_tid_4754;
    
    float x_4757;
    float res_4758;
    float res_4759;
    
    if (slt32(gtid_4747, sizze_4697)) {
        x_4757 = *(__global float *) &angles_mem_4820[gtid_4747 * 4];
        res_4758 = futrts_cos32(x_4757);
        res_4759 = (float) fabs(res_4758);
    }
    if (slt32(gtid_4747, sizze_4697)) {
        *(__global float *) &mem_4823[gtid_4747 * 4] = res_4759;
    }
}
__kernel void map_4771(int32_t sizze_4697, int32_t numrhos_4699, __global
                       unsigned char *mem_4830, __global
                       unsigned char *mem_4834)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_4861;
    int32_t group_sizze_4862;
    int32_t gtid_4762;
    int32_t gtid_4763;
    int32_t global_tid_4771;
    int32_t local_tid_4772;
    int32_t group_id_4773;
    
    global_tid_4771 = get_global_id(0);
    local_tid_4772 = get_local_id(0);
    group_sizze_4862 = get_local_size(0);
    wave_sizze_4861 = LOCKSTEP_WIDTH;
    group_id_4773 = get_group_id(0);
    gtid_4762 = squot32(global_tid_4771, numrhos_4699);
    gtid_4763 = global_tid_4771 - squot32(global_tid_4771, numrhos_4699) *
        numrhos_4699;
    
    float res_4774;
    
    if (slt32(gtid_4762, sizze_4697) && slt32(gtid_4763, numrhos_4699)) {
        res_4774 = *(__global float *) &mem_4830[gtid_4762 * 4];
    }
    if (slt32(gtid_4762, sizze_4697) && slt32(gtid_4763, numrhos_4699)) {
        *(__global float *) &mem_4834[(gtid_4762 * numrhos_4699 + gtid_4763) *
                                      4] = res_4774;
    }
}
__kernel void map_4783(int32_t sizze_4697, __global
                       unsigned char *angles_mem_4820, __global
                       unsigned char *mem_4830)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_4859;
    int32_t group_sizze_4860;
    int32_t gtid_4776;
    int32_t global_tid_4783;
    int32_t local_tid_4784;
    int32_t group_id_4785;
    
    global_tid_4783 = get_global_id(0);
    local_tid_4784 = get_local_id(0);
    group_sizze_4860 = get_local_size(0);
    wave_sizze_4859 = LOCKSTEP_WIDTH;
    group_id_4785 = get_group_id(0);
    gtid_4776 = global_tid_4783;
    
    float x_4786;
    float res_4787;
    float res_4788;
    
    if (slt32(gtid_4776, sizze_4697)) {
        x_4786 = *(__global float *) &angles_mem_4820[gtid_4776 * 4];
        res_4787 = futrts_sin32(x_4786);
        res_4788 = (float) fabs(res_4787);
    }
    if (slt32(gtid_4776, sizze_4697)) {
        *(__global float *) &mem_4830[gtid_4776 * 4] = res_4788;
    }
}
__kernel void map_4800(int32_t sizze_4697, int32_t numrhos_4699, __global
                       unsigned char *mem_4836, __global
                       unsigned char *mem_4839)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_4865;
    int32_t group_sizze_4866;
    int32_t gtid_4791;
    int32_t gtid_4792;
    int32_t global_tid_4800;
    int32_t local_tid_4801;
    int32_t group_id_4802;
    
    global_tid_4800 = get_global_id(0);
    local_tid_4801 = get_local_id(0);
    group_sizze_4866 = get_local_size(0);
    wave_sizze_4865 = LOCKSTEP_WIDTH;
    group_id_4802 = get_group_id(0);
    gtid_4791 = squot32(global_tid_4800, numrhos_4699);
    gtid_4792 = global_tid_4800 - squot32(global_tid_4800, numrhos_4699) *
        numrhos_4699;
    
    bool arg_4803;
    
    if (slt32(gtid_4791, sizze_4697) && slt32(gtid_4792, numrhos_4699)) {
        arg_4803 = *(__global bool *) &mem_4836[gtid_4791];
    }
    if (slt32(gtid_4791, sizze_4697) && slt32(gtid_4792, numrhos_4699)) {
        *(__global bool *) &mem_4839[gtid_4791 * numrhos_4699 + gtid_4792] =
            arg_4803;
    }
}
__kernel void map_4812(int32_t sizze_4697, __global
                       unsigned char *angles_mem_4820, __global
                       unsigned char *mem_4836)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_4863;
    int32_t group_sizze_4864;
    int32_t gtid_4805;
    int32_t global_tid_4812;
    int32_t local_tid_4813;
    int32_t group_id_4814;
    
    global_tid_4812 = get_global_id(0);
    local_tid_4813 = get_local_id(0);
    group_sizze_4864 = get_local_size(0);
    wave_sizze_4863 = LOCKSTEP_WIDTH;
    group_id_4814 = get_group_id(0);
    gtid_4805 = global_tid_4812;
    
    float x_4815;
    float res_4816;
    float res_4817;
    bool arg_4818;
    
    if (slt32(gtid_4805, sizze_4697)) {
        x_4815 = *(__global float *) &angles_mem_4820[gtid_4805 * 4];
        res_4816 = futrts_sin32(x_4815);
        res_4817 = futrts_cos32(x_4815);
        arg_4818 = res_4817 <= res_4816;
    }
    if (slt32(gtid_4805, sizze_4697)) {
        *(__global bool *) &mem_4836[gtid_4805] = arg_4818;
    }
}
