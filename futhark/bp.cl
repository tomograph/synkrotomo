#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
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
#define group_sizze_8336 (mainzigroup_sizze_8266)
#define group_sizze_8336 (mainzigroup_sizze_8266)
#define group_sizze_8336 (mainzigroup_sizze_8266)
__kernel void map_7630(int32_t sizze_7338, int32_t sizze_7343, float res_7345,
                       float res_7347, int32_t res_7359, int32_t arg_7360,
                       int32_t y_7373, int32_t arg_7398, __global
                       unsigned char *angles_mem_8594, __global
                       unsigned char *projections_mem_8598, __global
                       unsigned char *mem_8601)
{
    int32_t wave_sizze_8621;
    int32_t group_sizze_8622;
    int32_t gtid_7623;
    int32_t global_tid_7630;
    int32_t local_tid_7631;
    int32_t group_id_7632;
    
    global_tid_7630 = get_global_id(0);
    local_tid_7631 = get_local_id(0);
    group_sizze_8622 = get_local_size(0);
    wave_sizze_8621 = LOCKSTEP_WIDTH;
    group_id_7632 = get_group_id(0);
    gtid_7623 = global_tid_7630;
    
    int32_t res_7842;
    int32_t y_7843;
    int32_t res_7844;
    int32_t res_7845;
    int32_t res_7846;
    float res_7847;
    float res_7848;
    float x_7849;
    float x_7850;
    float res_7851;
    float res_7852;
    float res_7853;
    
    if (slt32(gtid_7623, arg_7360)) {
        res_7842 = sdiv32(gtid_7623, sizze_7343);
        y_7843 = sizze_7343 * res_7842;
        res_7844 = gtid_7623 - y_7843;
        res_7845 = res_7844 - y_7373;
        res_7846 = res_7842 - y_7373;
        res_7847 = sitofp_i32_f32(res_7845);
        res_7848 = sitofp_i32_f32(res_7846);
        x_7849 = 0.5F + res_7847;
        x_7850 = 0.5F + res_7848;
        res_7851 = 1.0F + res_7847;
        res_7852 = 1.0F + res_7848;
        
        float x_7856 = 0.0F;
        
        for (int32_t chunk_offset_7855 = 0; chunk_offset_7855 < sizze_7338;
             chunk_offset_7855++) {
            float res_7865;
            float res_7866;
            float res_7867;
            float y_7868;
            float res_7869;
            float y_7870;
            float res_7871;
            float x_7872;
            float y_7873;
            float res_7874;
            float x_7875;
            float arg_7876;
            int32_t res_7877;
            float res_7878;
            bool res_7879;
            float res_7880;
            float y_7887;
            float res_7888;
            float res_7889;
            float res_7890;
            bool res_7891;
            int32_t x_7892;
            float y_7893;
            float y_7894;
            float y_7895;
            float y_7896;
            float res_7897;
            float res_8042;
            
            res_7865 = *(__global float *) &angles_mem_8594[chunk_offset_7855 *
                                                            4];
            res_7866 = futrts_sin32(res_7865);
            res_7867 = futrts_cos32(res_7865);
            y_7868 = 0.70710677F * res_7867;
            res_7869 = x_7849 - y_7868;
            y_7870 = 0.70710677F * res_7866;
            res_7871 = x_7850 - y_7870;
            x_7872 = res_7867 * res_7869;
            y_7873 = res_7866 * res_7871;
            res_7874 = x_7872 + y_7873;
            x_7875 = res_7874 - res_7345;
            arg_7876 = x_7875 / res_7347;
            res_7877 = fptosi_f32_i32(arg_7876);
            res_7878 = sitofp_i32_f32(res_7877);
            res_7879 = 0.0F <= arg_7876;
            if (res_7879) {
                bool res_7881;
                float res_7882;
                
                res_7881 = res_7878 < arg_7876;
                if (res_7881) {
                    int32_t res_7883;
                    float res_7884;
                    
                    res_7883 = 1 + res_7877;
                    res_7884 = sitofp_i32_f32(res_7883);
                    res_7882 = res_7884;
                } else {
                    res_7882 = arg_7876;
                }
                res_7880 = res_7882;
            } else {
                bool res_7885;
                float res_7886;
                
                res_7885 = arg_7876 < res_7878;
                if (res_7885) {
                    res_7886 = res_7878;
                } else {
                    res_7886 = arg_7876;
                }
                res_7880 = res_7886;
            }
            y_7887 = res_7347 * res_7880;
            res_7888 = res_7345 + y_7887;
            res_7889 = (float) fabs(res_7866);
            res_7890 = (float) fabs(res_7867);
            res_7891 = res_7890 <= res_7889;
            x_7892 = arg_7398 * chunk_offset_7855;
            y_7893 = res_7847 * res_7867;
            y_7894 = res_7848 * res_7866;
            y_7895 = res_7851 * res_7867;
            y_7896 = res_7852 * res_7866;
            
            float x_7900 = 0.0F;
            
            for (int32_t chunk_offset_7899 = 0; chunk_offset_7899 < res_7359;
                 chunk_offset_7899++) {
                float res_7909;
                float y_7910;
                float res_7911;
                float res_7912;
                float res_7913;
                float res_7918;
                float res_7919;
                bool cond_7924;
                bool res_7925;
                bool x_7926;
                bool cond_7927;
                bool res_7928;
                bool x_7929;
                bool x_7930;
                bool y_7931;
                bool cond_7932;
                bool cond_7933;
                bool res_7934;
                bool x_7935;
                bool x_7936;
                bool y_7937;
                bool cond_7938;
                bool cond_7939;
                bool res_7940;
                bool x_7941;
                bool x_7942;
                bool y_7943;
                bool cond_7944;
                float res_7945;
                float x_8032;
                float arg_8033;
                float arg_8034;
                int32_t res_8035;
                int32_t res_8036;
                float y_8037;
                float res_8038;
                float res_8040;
                
                res_7909 = sitofp_i32_f32(chunk_offset_7899);
                y_7910 = res_7347 * res_7909;
                res_7911 = res_7888 + y_7910;
                if (res_7891) {
                    float x_7914;
                    float res_7915;
                    
                    x_7914 = res_7911 - y_7893;
                    res_7915 = x_7914 / res_7866;
                    res_7912 = res_7847;
                    res_7913 = res_7915;
                } else {
                    float x_7916;
                    float res_7917;
                    
                    x_7916 = res_7911 - y_7894;
                    res_7917 = x_7916 / res_7867;
                    res_7912 = res_7917;
                    res_7913 = res_7848;
                }
                if (res_7891) {
                    float x_7920;
                    float res_7921;
                    
                    x_7920 = res_7911 - y_7895;
                    res_7921 = x_7920 / res_7866;
                    res_7918 = res_7851;
                    res_7919 = res_7921;
                } else {
                    float x_7922;
                    float res_7923;
                    
                    x_7922 = res_7911 - y_7896;
                    res_7923 = x_7922 / res_7867;
                    res_7918 = res_7923;
                    res_7919 = res_7852;
                }
                cond_7924 = res_7912 < res_7847;
                res_7925 = res_7918 < res_7847;
                x_7926 = cond_7924 && res_7925;
                cond_7927 = res_7851 < res_7912;
                res_7928 = res_7851 < res_7918;
                x_7929 = cond_7927 && res_7928;
                x_7930 = !x_7926;
                y_7931 = x_7929 && x_7930;
                cond_7932 = x_7926 || y_7931;
                cond_7933 = res_7913 < res_7848;
                res_7934 = res_7919 < res_7848;
                x_7935 = cond_7933 && res_7934;
                x_7936 = !cond_7932;
                y_7937 = x_7935 && x_7936;
                cond_7938 = cond_7932 || y_7937;
                cond_7939 = res_7852 < res_7913;
                res_7940 = res_7852 < res_7919;
                x_7941 = cond_7939 && res_7940;
                x_7942 = !cond_7938;
                y_7943 = x_7941 && x_7942;
                cond_7944 = cond_7938 || y_7943;
                if (cond_7944) {
                    res_7945 = 0.0F;
                } else {
                    float res_7946;
                    
                    if (cond_7924) {
                        float x_7948;
                        float res_7949;
                        float x_7950;
                        float x_7951;
                        float x_7952;
                        float y_7953;
                        float arg_7954;
                        float res_7955;
                        
                        x_7948 = res_7911 - y_7893;
                        res_7949 = x_7948 / res_7866;
                        x_7950 = res_7918 - res_7847;
                        x_7951 = fpow32(x_7950, 2.0F);
                        x_7952 = res_7919 - res_7949;
                        y_7953 = fpow32(x_7952, 2.0F);
                        arg_7954 = x_7951 + y_7953;
                        res_7955 = futrts_sqrt32(arg_7954);
                        res_7946 = res_7955;
                    } else {
                        float res_7956;
                        
                        if (res_7925) {
                            float x_7958;
                            float res_7959;
                            float x_7960;
                            float x_7961;
                            float x_7962;
                            float y_7963;
                            float arg_7964;
                            float res_7965;
                            
                            x_7958 = res_7911 - y_7893;
                            res_7959 = x_7958 / res_7866;
                            x_7960 = res_7912 - res_7847;
                            x_7961 = fpow32(x_7960, 2.0F);
                            x_7962 = res_7913 - res_7959;
                            y_7963 = fpow32(x_7962, 2.0F);
                            arg_7964 = x_7961 + y_7963;
                            res_7965 = futrts_sqrt32(arg_7964);
                            res_7956 = res_7965;
                        } else {
                            float res_7966;
                            
                            if (cond_7927) {
                                float x_7968;
                                float res_7969;
                                float x_7970;
                                float x_7971;
                                float x_7972;
                                float y_7973;
                                float arg_7974;
                                float res_7975;
                                
                                x_7968 = res_7911 - y_7895;
                                res_7969 = x_7968 / res_7866;
                                x_7970 = res_7918 - res_7851;
                                x_7971 = fpow32(x_7970, 2.0F);
                                x_7972 = res_7919 - res_7969;
                                y_7973 = fpow32(x_7972, 2.0F);
                                arg_7974 = x_7971 + y_7973;
                                res_7975 = futrts_sqrt32(arg_7974);
                                res_7966 = res_7975;
                            } else {
                                float res_7976;
                                
                                if (res_7928) {
                                    float x_7978;
                                    float res_7979;
                                    float x_7980;
                                    float x_7981;
                                    float x_7982;
                                    float y_7983;
                                    float arg_7984;
                                    float res_7985;
                                    
                                    x_7978 = res_7911 - y_7895;
                                    res_7979 = x_7978 / res_7866;
                                    x_7980 = res_7912 - res_7851;
                                    x_7981 = fpow32(x_7980, 2.0F);
                                    x_7982 = res_7913 - res_7979;
                                    y_7983 = fpow32(x_7982, 2.0F);
                                    arg_7984 = x_7981 + y_7983;
                                    res_7985 = futrts_sqrt32(arg_7984);
                                    res_7976 = res_7985;
                                } else {
                                    float res_7986;
                                    
                                    if (cond_7933) {
                                        float x_7988;
                                        float res_7989;
                                        float x_7990;
                                        float x_7991;
                                        float x_7992;
                                        float y_7993;
                                        float arg_7994;
                                        float res_7995;
                                        
                                        x_7988 = res_7911 - y_7894;
                                        res_7989 = x_7988 / res_7867;
                                        x_7990 = res_7918 - res_7989;
                                        x_7991 = fpow32(x_7990, 2.0F);
                                        x_7992 = res_7919 - res_7848;
                                        y_7993 = fpow32(x_7992, 2.0F);
                                        arg_7994 = x_7991 + y_7993;
                                        res_7995 = futrts_sqrt32(arg_7994);
                                        res_7986 = res_7995;
                                    } else {
                                        float res_7996;
                                        
                                        if (res_7934) {
                                            float x_7998;
                                            float res_7999;
                                            float x_8000;
                                            float x_8001;
                                            float x_8002;
                                            float y_8003;
                                            float arg_8004;
                                            float res_8005;
                                            
                                            x_7998 = res_7911 - y_7894;
                                            res_7999 = x_7998 / res_7867;
                                            x_8000 = res_7912 - res_7999;
                                            x_8001 = fpow32(x_8000, 2.0F);
                                            x_8002 = res_7913 - res_7848;
                                            y_8003 = fpow32(x_8002, 2.0F);
                                            arg_8004 = x_8001 + y_8003;
                                            res_8005 = futrts_sqrt32(arg_8004);
                                            res_7996 = res_8005;
                                        } else {
                                            float res_8006;
                                            
                                            if (cond_7939) {
                                                float x_8008;
                                                float res_8009;
                                                float x_8010;
                                                float x_8011;
                                                float x_8012;
                                                float y_8013;
                                                float arg_8014;
                                                float res_8015;
                                                
                                                x_8008 = res_7911 - y_7896;
                                                res_8009 = x_8008 / res_7867;
                                                x_8010 = res_7918 - res_8009;
                                                x_8011 = fpow32(x_8010, 2.0F);
                                                x_8012 = res_7919 - res_7852;
                                                y_8013 = fpow32(x_8012, 2.0F);
                                                arg_8014 = x_8011 + y_8013;
                                                res_8015 =
                                                    futrts_sqrt32(arg_8014);
                                                res_8006 = res_8015;
                                            } else {
                                                float res_8016;
                                                
                                                if (res_7940) {
                                                    float x_8018;
                                                    float res_8019;
                                                    float x_8020;
                                                    float x_8021;
                                                    float x_8022;
                                                    float y_8023;
                                                    float arg_8024;
                                                    float res_8025;
                                                    
                                                    x_8018 = res_7911 - y_7896;
                                                    res_8019 = x_8018 /
                                                        res_7867;
                                                    x_8020 = res_7912 -
                                                        res_8019;
                                                    x_8021 = fpow32(x_8020,
                                                                    2.0F);
                                                    x_8022 = res_7913 -
                                                        res_7852;
                                                    y_8023 = fpow32(x_8022,
                                                                    2.0F);
                                                    arg_8024 = x_8021 + y_8023;
                                                    res_8025 =
                                                        futrts_sqrt32(arg_8024);
                                                    res_8016 = res_8025;
                                                } else {
                                                    float x_8026;
                                                    float x_8027;
                                                    float x_8028;
                                                    float y_8029;
                                                    float arg_8030;
                                                    float res_8031;
                                                    
                                                    x_8026 = res_7918 -
                                                        res_7912;
                                                    x_8027 = fpow32(x_8026,
                                                                    2.0F);
                                                    x_8028 = res_7919 -
                                                        res_7913;
                                                    y_8029 = fpow32(x_8028,
                                                                    2.0F);
                                                    arg_8030 = x_8027 + y_8029;
                                                    res_8031 =
                                                        futrts_sqrt32(arg_8030);
                                                    res_8016 = res_8031;
                                                }
                                                res_8006 = res_8016;
                                            }
                                            res_7996 = res_8006;
                                        }
                                        res_7986 = res_7996;
                                    }
                                    res_7976 = res_7986;
                                }
                                res_7966 = res_7976;
                            }
                            res_7956 = res_7966;
                        }
                        res_7946 = res_7956;
                    }
                    res_7945 = res_7946;
                }
                x_8032 = res_7911 - res_7345;
                arg_8033 = x_8032 / res_7347;
                arg_8034 = futrts_round32(arg_8033);
                res_8035 = fptosi_f32_i32(arg_8034);
                res_8036 = x_7892 + res_8035;
                y_8037 = *(__global float *) &projections_mem_8598[res_8036 *
                                                                   4];
                res_8038 = res_7945 * y_8037;
                res_8040 = x_7900 + res_8038;
                
                float x_tmp_8624 = res_8040;
                
                x_7900 = x_tmp_8624;
            }
            res_7897 = x_7900;
            res_8042 = x_7856 + res_7897;
            
            float x_tmp_8623 = res_8042;
            
            x_7856 = x_tmp_8623;
        }
        res_7853 = x_7856;
    }
    if (slt32(gtid_7623, arg_7360)) {
        *(__global float *) &mem_8601[gtid_7623 * 4] = res_7853;
    }
}
__kernel void map_intra_group_7601(__local volatile int64_t *mem_aligned_0,
                                   int32_t sizze_7338, int32_t sizze_7343,
                                   float res_7345, float res_7347,
                                   int32_t res_7359, int32_t arg_7360,
                                   int32_t y_7373, int32_t arg_7398, __global
                                   unsigned char *angles_mem_8594, __global
                                   unsigned char *projections_mem_8598, __global
                                   unsigned char *mem_8607)
{
    __local volatile char *restrict mem_8604 = mem_aligned_0;
    int32_t wave_sizze_8625;
    int32_t group_sizze_8626;
    int32_t gtid_7585;
    int32_t ltid_7586;
    int32_t global_tid_7601;
    int32_t local_tid_7602;
    int32_t group_id_7603;
    
    global_tid_7601 = get_global_id(0);
    local_tid_7602 = get_local_id(0);
    group_sizze_8626 = get_local_size(0);
    wave_sizze_8625 = LOCKSTEP_WIDTH;
    group_id_7603 = get_group_id(0);
    gtid_7585 = squot32(global_tid_7601, sizze_7338);
    ltid_7586 = global_tid_7601 - squot32(global_tid_7601, sizze_7338) *
        sizze_7338;
    
    int32_t res_8049;
    int32_t y_8050;
    int32_t res_8051;
    int32_t res_8052;
    int32_t res_8053;
    float res_8054;
    float res_8055;
    float x_8056;
    float x_8057;
    float res_8058;
    float res_8059;
    float x_8554;
    float res_8063;
    float res_8064;
    float y_8065;
    float res_8066;
    float y_8067;
    float res_8068;
    float x_8069;
    float y_8070;
    float res_8071;
    float x_8072;
    float arg_8073;
    int32_t res_8074;
    float res_8075;
    bool res_8076;
    float res_8077;
    float y_8084;
    float res_8085;
    float res_8086;
    float res_8087;
    bool res_8088;
    int32_t x_8089;
    float y_8090;
    float y_8091;
    float y_8092;
    float y_8093;
    float x_8556;
    
    if (slt32(gtid_7585, arg_7360) && slt32(ltid_7586, sizze_7338)) {
        res_8049 = sdiv32(gtid_7585, sizze_7343);
        y_8050 = sizze_7343 * res_8049;
        res_8051 = gtid_7585 - y_8050;
        res_8052 = res_8051 - y_7373;
        res_8053 = res_8049 - y_7373;
        res_8054 = sitofp_i32_f32(res_8052);
        res_8055 = sitofp_i32_f32(res_8053);
        x_8056 = 0.5F + res_8054;
        x_8057 = 0.5F + res_8055;
        res_8058 = 1.0F + res_8054;
        res_8059 = 1.0F + res_8055;
        x_8554 = *(__global float *) &angles_mem_8594[ltid_7586 * 4];
        res_8063 = futrts_sin32(x_8554);
        res_8064 = futrts_cos32(x_8554);
        y_8065 = 0.70710677F * res_8064;
        res_8066 = x_8056 - y_8065;
        y_8067 = 0.70710677F * res_8063;
        res_8068 = x_8057 - y_8067;
        x_8069 = res_8064 * res_8066;
        y_8070 = res_8063 * res_8068;
        res_8071 = x_8069 + y_8070;
        x_8072 = res_8071 - res_7345;
        arg_8073 = x_8072 / res_7347;
        res_8074 = fptosi_f32_i32(arg_8073);
        res_8075 = sitofp_i32_f32(res_8074);
        res_8076 = 0.0F <= arg_8073;
        if (res_8076) {
            bool res_8078;
            float res_8079;
            
            res_8078 = res_8075 < arg_8073;
            if (res_8078) {
                int32_t res_8080;
                float res_8081;
                
                res_8080 = 1 + res_8074;
                res_8081 = sitofp_i32_f32(res_8080);
                res_8079 = res_8081;
            } else {
                res_8079 = arg_8073;
            }
            res_8077 = res_8079;
        } else {
            bool res_8082;
            float res_8083;
            
            res_8082 = arg_8073 < res_8075;
            if (res_8082) {
                res_8083 = res_8075;
            } else {
                res_8083 = arg_8073;
            }
            res_8077 = res_8083;
        }
        y_8084 = res_7347 * res_8077;
        res_8085 = res_7345 + y_8084;
        res_8086 = (float) fabs(res_8063);
        res_8087 = (float) fabs(res_8064);
        res_8088 = res_8087 <= res_8086;
        x_8089 = arg_7398 * ltid_7586;
        y_8090 = res_8054 * res_8064;
        y_8091 = res_8055 * res_8063;
        y_8092 = res_8058 * res_8064;
        y_8093 = res_8059 * res_8063;
        
        float x_8097 = 0.0F;
        
        for (int32_t chunk_offset_8096 = 0; chunk_offset_8096 < res_7359;
             chunk_offset_8096++) {
            float res_8106;
            float y_8107;
            float res_8108;
            float res_8109;
            float res_8110;
            float res_8115;
            float res_8116;
            bool cond_8121;
            bool res_8122;
            bool x_8123;
            bool cond_8124;
            bool res_8125;
            bool x_8126;
            bool x_8127;
            bool y_8128;
            bool cond_8129;
            bool cond_8130;
            bool res_8131;
            bool x_8132;
            bool x_8133;
            bool y_8134;
            bool cond_8135;
            bool cond_8136;
            bool res_8137;
            bool x_8138;
            bool x_8139;
            bool y_8140;
            bool cond_8141;
            float res_8142;
            float x_8229;
            float arg_8230;
            float arg_8231;
            int32_t res_8232;
            int32_t res_8233;
            float y_8234;
            float res_8235;
            float res_8237;
            
            res_8106 = sitofp_i32_f32(chunk_offset_8096);
            y_8107 = res_7347 * res_8106;
            res_8108 = res_8085 + y_8107;
            if (res_8088) {
                float x_8111;
                float res_8112;
                
                x_8111 = res_8108 - y_8090;
                res_8112 = x_8111 / res_8063;
                res_8109 = res_8054;
                res_8110 = res_8112;
            } else {
                float x_8113;
                float res_8114;
                
                x_8113 = res_8108 - y_8091;
                res_8114 = x_8113 / res_8064;
                res_8109 = res_8114;
                res_8110 = res_8055;
            }
            if (res_8088) {
                float x_8117;
                float res_8118;
                
                x_8117 = res_8108 - y_8092;
                res_8118 = x_8117 / res_8063;
                res_8115 = res_8058;
                res_8116 = res_8118;
            } else {
                float x_8119;
                float res_8120;
                
                x_8119 = res_8108 - y_8093;
                res_8120 = x_8119 / res_8064;
                res_8115 = res_8120;
                res_8116 = res_8059;
            }
            cond_8121 = res_8109 < res_8054;
            res_8122 = res_8115 < res_8054;
            x_8123 = cond_8121 && res_8122;
            cond_8124 = res_8058 < res_8109;
            res_8125 = res_8058 < res_8115;
            x_8126 = cond_8124 && res_8125;
            x_8127 = !x_8123;
            y_8128 = x_8126 && x_8127;
            cond_8129 = x_8123 || y_8128;
            cond_8130 = res_8110 < res_8055;
            res_8131 = res_8116 < res_8055;
            x_8132 = cond_8130 && res_8131;
            x_8133 = !cond_8129;
            y_8134 = x_8132 && x_8133;
            cond_8135 = cond_8129 || y_8134;
            cond_8136 = res_8059 < res_8110;
            res_8137 = res_8059 < res_8116;
            x_8138 = cond_8136 && res_8137;
            x_8139 = !cond_8135;
            y_8140 = x_8138 && x_8139;
            cond_8141 = cond_8135 || y_8140;
            if (cond_8141) {
                res_8142 = 0.0F;
            } else {
                float res_8143;
                
                if (cond_8121) {
                    float x_8145;
                    float res_8146;
                    float x_8147;
                    float x_8148;
                    float x_8149;
                    float y_8150;
                    float arg_8151;
                    float res_8152;
                    
                    x_8145 = res_8108 - y_8090;
                    res_8146 = x_8145 / res_8063;
                    x_8147 = res_8115 - res_8054;
                    x_8148 = fpow32(x_8147, 2.0F);
                    x_8149 = res_8116 - res_8146;
                    y_8150 = fpow32(x_8149, 2.0F);
                    arg_8151 = x_8148 + y_8150;
                    res_8152 = futrts_sqrt32(arg_8151);
                    res_8143 = res_8152;
                } else {
                    float res_8153;
                    
                    if (res_8122) {
                        float x_8155;
                        float res_8156;
                        float x_8157;
                        float x_8158;
                        float x_8159;
                        float y_8160;
                        float arg_8161;
                        float res_8162;
                        
                        x_8155 = res_8108 - y_8090;
                        res_8156 = x_8155 / res_8063;
                        x_8157 = res_8109 - res_8054;
                        x_8158 = fpow32(x_8157, 2.0F);
                        x_8159 = res_8110 - res_8156;
                        y_8160 = fpow32(x_8159, 2.0F);
                        arg_8161 = x_8158 + y_8160;
                        res_8162 = futrts_sqrt32(arg_8161);
                        res_8153 = res_8162;
                    } else {
                        float res_8163;
                        
                        if (cond_8124) {
                            float x_8165;
                            float res_8166;
                            float x_8167;
                            float x_8168;
                            float x_8169;
                            float y_8170;
                            float arg_8171;
                            float res_8172;
                            
                            x_8165 = res_8108 - y_8092;
                            res_8166 = x_8165 / res_8063;
                            x_8167 = res_8115 - res_8058;
                            x_8168 = fpow32(x_8167, 2.0F);
                            x_8169 = res_8116 - res_8166;
                            y_8170 = fpow32(x_8169, 2.0F);
                            arg_8171 = x_8168 + y_8170;
                            res_8172 = futrts_sqrt32(arg_8171);
                            res_8163 = res_8172;
                        } else {
                            float res_8173;
                            
                            if (res_8125) {
                                float x_8175;
                                float res_8176;
                                float x_8177;
                                float x_8178;
                                float x_8179;
                                float y_8180;
                                float arg_8181;
                                float res_8182;
                                
                                x_8175 = res_8108 - y_8092;
                                res_8176 = x_8175 / res_8063;
                                x_8177 = res_8109 - res_8058;
                                x_8178 = fpow32(x_8177, 2.0F);
                                x_8179 = res_8110 - res_8176;
                                y_8180 = fpow32(x_8179, 2.0F);
                                arg_8181 = x_8178 + y_8180;
                                res_8182 = futrts_sqrt32(arg_8181);
                                res_8173 = res_8182;
                            } else {
                                float res_8183;
                                
                                if (cond_8130) {
                                    float x_8185;
                                    float res_8186;
                                    float x_8187;
                                    float x_8188;
                                    float x_8189;
                                    float y_8190;
                                    float arg_8191;
                                    float res_8192;
                                    
                                    x_8185 = res_8108 - y_8091;
                                    res_8186 = x_8185 / res_8064;
                                    x_8187 = res_8115 - res_8186;
                                    x_8188 = fpow32(x_8187, 2.0F);
                                    x_8189 = res_8116 - res_8055;
                                    y_8190 = fpow32(x_8189, 2.0F);
                                    arg_8191 = x_8188 + y_8190;
                                    res_8192 = futrts_sqrt32(arg_8191);
                                    res_8183 = res_8192;
                                } else {
                                    float res_8193;
                                    
                                    if (res_8131) {
                                        float x_8195;
                                        float res_8196;
                                        float x_8197;
                                        float x_8198;
                                        float x_8199;
                                        float y_8200;
                                        float arg_8201;
                                        float res_8202;
                                        
                                        x_8195 = res_8108 - y_8091;
                                        res_8196 = x_8195 / res_8064;
                                        x_8197 = res_8109 - res_8196;
                                        x_8198 = fpow32(x_8197, 2.0F);
                                        x_8199 = res_8110 - res_8055;
                                        y_8200 = fpow32(x_8199, 2.0F);
                                        arg_8201 = x_8198 + y_8200;
                                        res_8202 = futrts_sqrt32(arg_8201);
                                        res_8193 = res_8202;
                                    } else {
                                        float res_8203;
                                        
                                        if (cond_8136) {
                                            float x_8205;
                                            float res_8206;
                                            float x_8207;
                                            float x_8208;
                                            float x_8209;
                                            float y_8210;
                                            float arg_8211;
                                            float res_8212;
                                            
                                            x_8205 = res_8108 - y_8093;
                                            res_8206 = x_8205 / res_8064;
                                            x_8207 = res_8115 - res_8206;
                                            x_8208 = fpow32(x_8207, 2.0F);
                                            x_8209 = res_8116 - res_8059;
                                            y_8210 = fpow32(x_8209, 2.0F);
                                            arg_8211 = x_8208 + y_8210;
                                            res_8212 = futrts_sqrt32(arg_8211);
                                            res_8203 = res_8212;
                                        } else {
                                            float res_8213;
                                            
                                            if (res_8137) {
                                                float x_8215;
                                                float res_8216;
                                                float x_8217;
                                                float x_8218;
                                                float x_8219;
                                                float y_8220;
                                                float arg_8221;
                                                float res_8222;
                                                
                                                x_8215 = res_8108 - y_8093;
                                                res_8216 = x_8215 / res_8064;
                                                x_8217 = res_8109 - res_8216;
                                                x_8218 = fpow32(x_8217, 2.0F);
                                                x_8219 = res_8110 - res_8059;
                                                y_8220 = fpow32(x_8219, 2.0F);
                                                arg_8221 = x_8218 + y_8220;
                                                res_8222 =
                                                    futrts_sqrt32(arg_8221);
                                                res_8213 = res_8222;
                                            } else {
                                                float x_8223;
                                                float x_8224;
                                                float x_8225;
                                                float y_8226;
                                                float arg_8227;
                                                float res_8228;
                                                
                                                x_8223 = res_8115 - res_8109;
                                                x_8224 = fpow32(x_8223, 2.0F);
                                                x_8225 = res_8116 - res_8110;
                                                y_8226 = fpow32(x_8225, 2.0F);
                                                arg_8227 = x_8224 + y_8226;
                                                res_8228 =
                                                    futrts_sqrt32(arg_8227);
                                                res_8213 = res_8228;
                                            }
                                            res_8203 = res_8213;
                                        }
                                        res_8193 = res_8203;
                                    }
                                    res_8183 = res_8193;
                                }
                                res_8173 = res_8183;
                            }
                            res_8163 = res_8173;
                        }
                        res_8153 = res_8163;
                    }
                    res_8143 = res_8153;
                }
                res_8142 = res_8143;
            }
            x_8229 = res_8108 - res_7345;
            arg_8230 = x_8229 / res_7347;
            arg_8231 = futrts_round32(arg_8230);
            res_8232 = fptosi_f32_i32(arg_8231);
            res_8233 = x_8089 + res_8232;
            y_8234 = *(__global float *) &projections_mem_8598[res_8233 * 4];
            res_8235 = res_8142 * y_8234;
            res_8237 = x_8097 + res_8235;
            
            float x_tmp_8627 = res_8237;
            
            x_8097 = x_tmp_8627;
        }
        x_8556 = x_8097;
    }
    
    float res_8238;
    
    for (int32_t comb_iter_8628 = 0; comb_iter_8628 < 1; comb_iter_8628++) {
        int32_t ctid_7597;
        int32_t flat_comb_id_8629 = comb_iter_8628 * sizze_7338 +
                local_tid_7602;
        
        ctid_7597 = flat_comb_id_8629;
        if (slt32(ctid_7597, sizze_7338) && 1) {
            *(__local float *) &mem_8604[ctid_7597 * 4] = x_8556;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t my_index_8239;
    int32_t offset_8240;
    
    my_index_8239 = global_tid_7601;
    
    int32_t skip_waves_8630;
    float x_8241;
    float x_8242;
    
    offset_8240 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_7602, sizze_7338)) {
            x_8241 = *(__local float *) &mem_8604[(local_tid_7602 +
                                                   offset_8240) * 4];
        }
    }
    offset_8240 = 1;
    while (slt32(offset_8240, wave_sizze_8625)) {
        if (slt32(local_tid_7602 + offset_8240, sizze_7338) &&
            ((local_tid_7602 - squot32(local_tid_7602, wave_sizze_8625) *
              wave_sizze_8625) & (2 * offset_8240 - 1)) == 0) {
            // read array element
            {
                x_8242 = *(volatile __local float *) &mem_8604[(local_tid_7602 +
                                                                offset_8240) *
                                                               4];
            }
            // apply reduction operation
            {
                float res_8243;
                
                if (slt32(gtid_7585, arg_7360) && slt32(ltid_7586,
                                                        sizze_7338)) {
                    res_8243 = x_8241 + x_8242;
                }
                x_8241 = res_8243;
            }
            // write result of operation
            {
                *(volatile __local float *) &mem_8604[local_tid_7602 * 4] =
                    x_8241;
            }
        }
        offset_8240 *= 2;
    }
    skip_waves_8630 = 1;
    while (slt32(skip_waves_8630, squot32(sizze_7338 + wave_sizze_8625 - 1,
                                          wave_sizze_8625))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_8240 = skip_waves_8630 * wave_sizze_8625;
        if (slt32(local_tid_7602 + offset_8240, sizze_7338) &&
            ((local_tid_7602 - squot32(local_tid_7602, wave_sizze_8625) *
              wave_sizze_8625) == 0 && (squot32(local_tid_7602,
                                                wave_sizze_8625) & (2 *
                                                                    skip_waves_8630 -
                                                                    1)) == 0)) {
            // read array element
            {
                x_8242 = *(__local float *) &mem_8604[(local_tid_7602 +
                                                       offset_8240) * 4];
            }
            // apply reduction operation
            {
                float res_8243;
                
                if (slt32(gtid_7585, arg_7360) && slt32(ltid_7586,
                                                        sizze_7338)) {
                    res_8243 = x_8241 + x_8242;
                }
                x_8241 = res_8243;
            }
            // write result of operation
            {
                *(__local float *) &mem_8604[local_tid_7602 * 4] = x_8241;
            }
        }
        skip_waves_8630 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_8238 = *(__local float *) &mem_8604[0];
    if (local_tid_7602 == 0) {
        *(__global float *) &mem_8607[group_id_7603 * 4] = res_8238;
    }
}
__kernel void segred_large_8284(__local volatile int64_t *red_arr_mem_aligned_0,
                                __local volatile
                                int64_t *sync_arr_mem_aligned_1,
                                int32_t sizze_7338, int32_t sizze_7343,
                                float res_7345, float res_7347,
                                int32_t res_7359, int32_t arg_7360,
                                int32_t y_7373, int32_t arg_7398,
                                int32_t num_groups_8346, __global
                                unsigned char *angles_mem_8594, __global
                                unsigned char *projections_mem_8598, __global
                                unsigned char *mem_8610,
                                int32_t thread_per_segment_8658, __global
                                unsigned char *group_res_arr_mem_8662, __global
                                unsigned char *counter_mem_8665)
{
    __local volatile char *restrict red_arr_mem_8659 = red_arr_mem_aligned_0;
    __local volatile char *restrict sync_arr_mem_8667 = sync_arr_mem_aligned_1;
    int32_t wave_sizze_8654;
    int32_t group_sizze_8655;
    int32_t gtid_8260;
    int32_t gtid_8283;
    int32_t global_tid_8284;
    int32_t local_tid_8285;
    int32_t group_id_8286;
    
    global_tid_8284 = get_global_id(0);
    local_tid_8285 = get_local_id(0);
    group_sizze_8655 = get_local_size(0);
    wave_sizze_8654 = LOCKSTEP_WIDTH;
    group_id_8286 = get_group_id(0);
    gtid_8260 = squot32(group_id_8286, squot32(num_groups_8346 + arg_7360 - 1,
                                               arg_7360));
    
    int32_t chunk_sizze_8670 = smin32(squot32(sizze_7338 + group_sizze_8336 *
                                              squot32(num_groups_8346 +
                                                      arg_7360 - 1, arg_7360) -
                                              1, group_sizze_8336 *
                                              squot32(num_groups_8346 +
                                                      arg_7360 - 1, arg_7360)),
                                      squot32(sizze_7338 -
                                              srem32(global_tid_8284,
                                                     group_sizze_8336 *
                                                     squot32(num_groups_8346 +
                                                             arg_7360 - 1,
                                                             arg_7360)) +
                                              thread_per_segment_8658 - 1,
                                              thread_per_segment_8658));
    float x_8352;
    float x_8353;
    
    x_8352 = 0.0F;
    for (int32_t i_8674 = 0; i_8674 < chunk_sizze_8670; i_8674++) {
        gtid_8283 = srem32(global_tid_8284, group_sizze_8336 *
                           squot32(num_groups_8346 + arg_7360 - 1, arg_7360)) +
            thread_per_segment_8658 * i_8674;
        // apply map function
        {
            int32_t binop_y_8588;
            int32_t binop_y_8589;
            int32_t binop_x_8590;
            int32_t convop_x_8591;
            float index_primexp_8592;
            float res_8355;
            int32_t convop_x_8586;
            float index_primexp_8587;
            float res_8356;
            float index_primexp_8584;
            float index_primexp_8578;
            float index_primexp_8574;
            float res_8359;
            float index_primexp_8568;
            float res_8360;
            float res_8362;
            float res_8363;
            float res_8364;
            float y_8365;
            float res_8366;
            float y_8367;
            float res_8368;
            float x_8369;
            float y_8370;
            float res_8371;
            float x_8372;
            float arg_8373;
            int32_t res_8374;
            float res_8375;
            bool res_8376;
            float res_8377;
            float y_8384;
            float res_8385;
            float res_8386;
            float res_8387;
            bool res_8388;
            int32_t x_8389;
            float y_8390;
            float y_8391;
            float y_8392;
            float y_8393;
            
            binop_y_8588 = sdiv32(gtid_8260, sizze_7343);
            binop_y_8589 = sizze_7343 * binop_y_8588;
            binop_x_8590 = gtid_8260 - binop_y_8589;
            convop_x_8591 = binop_x_8590 - y_7373;
            index_primexp_8592 = sitofp_i32_f32(convop_x_8591);
            res_8355 = index_primexp_8592;
            convop_x_8586 = binop_y_8588 - y_7373;
            index_primexp_8587 = sitofp_i32_f32(convop_x_8586);
            res_8356 = index_primexp_8587;
            index_primexp_8584 = 0.5F + index_primexp_8592;
            index_primexp_8578 = 0.5F + index_primexp_8587;
            index_primexp_8574 = 1.0F + index_primexp_8592;
            res_8359 = index_primexp_8574;
            index_primexp_8568 = 1.0F + index_primexp_8587;
            res_8360 = index_primexp_8568;
            res_8362 = *(__global float *) &angles_mem_8594[gtid_8283 * 4];
            res_8363 = futrts_sin32(res_8362);
            res_8364 = futrts_cos32(res_8362);
            y_8365 = 0.70710677F * res_8364;
            res_8366 = index_primexp_8584 - y_8365;
            y_8367 = 0.70710677F * res_8363;
            res_8368 = index_primexp_8578 - y_8367;
            x_8369 = res_8364 * res_8366;
            y_8370 = res_8363 * res_8368;
            res_8371 = x_8369 + y_8370;
            x_8372 = res_8371 - res_7345;
            arg_8373 = x_8372 / res_7347;
            res_8374 = fptosi_f32_i32(arg_8373);
            res_8375 = sitofp_i32_f32(res_8374);
            res_8376 = 0.0F <= arg_8373;
            if (res_8376) {
                bool res_8378;
                float res_8379;
                
                res_8378 = res_8375 < arg_8373;
                if (res_8378) {
                    int32_t res_8380;
                    float res_8381;
                    
                    res_8380 = 1 + res_8374;
                    res_8381 = sitofp_i32_f32(res_8380);
                    res_8379 = res_8381;
                } else {
                    res_8379 = arg_8373;
                }
                res_8377 = res_8379;
            } else {
                bool res_8382;
                float res_8383;
                
                res_8382 = arg_8373 < res_8375;
                if (res_8382) {
                    res_8383 = res_8375;
                } else {
                    res_8383 = arg_8373;
                }
                res_8377 = res_8383;
            }
            y_8384 = res_7347 * res_8377;
            res_8385 = res_7345 + y_8384;
            res_8386 = (float) fabs(res_8363);
            res_8387 = (float) fabs(res_8364);
            res_8388 = res_8387 <= res_8386;
            x_8389 = arg_7398 * gtid_8283;
            y_8390 = res_8364 * index_primexp_8592;
            y_8391 = res_8363 * index_primexp_8587;
            y_8392 = res_8364 * index_primexp_8574;
            y_8393 = res_8363 * index_primexp_8568;
            
            float res_8394;
            float x_8397 = 0.0F;
            int32_t chunk_sizze_8395;
            int32_t chunk_offset_8396 = 0;
            
            chunk_sizze_8395 = res_7359;
            
            float res_8399;
            float acc_8402 = x_8397;
            int32_t groupstream_mapaccum_dummy_chunk_sizze_8400;
            
            groupstream_mapaccum_dummy_chunk_sizze_8400 = 1;
            if (chunk_sizze_8395 == res_7359) {
                for (int32_t i_8401 = 0; i_8401 < res_7359; i_8401++) {
                    int32_t convop_x_8560;
                    float res_8406;
                    float y_8407;
                    float res_8408;
                    float res_8409;
                    float res_8410;
                    float res_8415;
                    float res_8416;
                    bool cond_8421;
                    bool res_8422;
                    bool x_8423;
                    bool cond_8424;
                    bool res_8425;
                    bool x_8426;
                    bool x_8427;
                    bool y_8428;
                    bool cond_8429;
                    bool cond_8430;
                    bool res_8431;
                    bool x_8432;
                    bool x_8433;
                    bool y_8434;
                    bool cond_8435;
                    bool cond_8436;
                    bool res_8437;
                    bool x_8438;
                    bool x_8439;
                    bool y_8440;
                    bool cond_8441;
                    float res_8442;
                    float x_8529;
                    float arg_8530;
                    float arg_8531;
                    int32_t res_8532;
                    int32_t res_8533;
                    float y_8534;
                    float res_8535;
                    float res_8537;
                    
                    convop_x_8560 = chunk_offset_8396 + i_8401;
                    res_8406 = sitofp_i32_f32(convop_x_8560);
                    y_8407 = res_7347 * res_8406;
                    res_8408 = res_8385 + y_8407;
                    if (res_8388) {
                        float x_8411;
                        float res_8412;
                        
                        x_8411 = res_8408 - y_8390;
                        res_8412 = x_8411 / res_8363;
                        res_8409 = res_8355;
                        res_8410 = res_8412;
                    } else {
                        float x_8413;
                        float res_8414;
                        
                        x_8413 = res_8408 - y_8391;
                        res_8414 = x_8413 / res_8364;
                        res_8409 = res_8414;
                        res_8410 = res_8356;
                    }
                    if (res_8388) {
                        float x_8417;
                        float res_8418;
                        
                        x_8417 = res_8408 - y_8392;
                        res_8418 = x_8417 / res_8363;
                        res_8415 = res_8359;
                        res_8416 = res_8418;
                    } else {
                        float x_8419;
                        float res_8420;
                        
                        x_8419 = res_8408 - y_8393;
                        res_8420 = x_8419 / res_8364;
                        res_8415 = res_8420;
                        res_8416 = res_8360;
                    }
                    cond_8421 = res_8409 < index_primexp_8592;
                    res_8422 = res_8415 < index_primexp_8592;
                    x_8423 = cond_8421 && res_8422;
                    cond_8424 = index_primexp_8574 < res_8409;
                    res_8425 = index_primexp_8574 < res_8415;
                    x_8426 = cond_8424 && res_8425;
                    x_8427 = !x_8423;
                    y_8428 = x_8426 && x_8427;
                    cond_8429 = x_8423 || y_8428;
                    cond_8430 = res_8410 < index_primexp_8587;
                    res_8431 = res_8416 < index_primexp_8587;
                    x_8432 = cond_8430 && res_8431;
                    x_8433 = !cond_8429;
                    y_8434 = x_8432 && x_8433;
                    cond_8435 = cond_8429 || y_8434;
                    cond_8436 = index_primexp_8568 < res_8410;
                    res_8437 = index_primexp_8568 < res_8416;
                    x_8438 = cond_8436 && res_8437;
                    x_8439 = !cond_8435;
                    y_8440 = x_8438 && x_8439;
                    cond_8441 = cond_8435 || y_8440;
                    if (cond_8441) {
                        res_8442 = 0.0F;
                    } else {
                        float res_8443;
                        
                        if (cond_8421) {
                            float y_8444;
                            float x_8445;
                            float res_8446;
                            float x_8447;
                            float x_8448;
                            float x_8449;
                            float y_8450;
                            float arg_8451;
                            float res_8452;
                            
                            y_8444 = res_8364 * index_primexp_8592;
                            x_8445 = res_8408 - y_8444;
                            res_8446 = x_8445 / res_8363;
                            x_8447 = res_8415 - index_primexp_8592;
                            x_8448 = fpow32(x_8447, 2.0F);
                            x_8449 = res_8416 - res_8446;
                            y_8450 = fpow32(x_8449, 2.0F);
                            arg_8451 = x_8448 + y_8450;
                            res_8452 = futrts_sqrt32(arg_8451);
                            res_8443 = res_8452;
                        } else {
                            float res_8453;
                            
                            if (res_8422) {
                                float y_8454;
                                float x_8455;
                                float res_8456;
                                float x_8457;
                                float x_8458;
                                float x_8459;
                                float y_8460;
                                float arg_8461;
                                float res_8462;
                                
                                y_8454 = res_8364 * index_primexp_8592;
                                x_8455 = res_8408 - y_8454;
                                res_8456 = x_8455 / res_8363;
                                x_8457 = res_8409 - index_primexp_8592;
                                x_8458 = fpow32(x_8457, 2.0F);
                                x_8459 = res_8410 - res_8456;
                                y_8460 = fpow32(x_8459, 2.0F);
                                arg_8461 = x_8458 + y_8460;
                                res_8462 = futrts_sqrt32(arg_8461);
                                res_8453 = res_8462;
                            } else {
                                float res_8463;
                                
                                if (cond_8424) {
                                    float y_8464;
                                    float x_8465;
                                    float res_8466;
                                    float x_8467;
                                    float x_8468;
                                    float x_8469;
                                    float y_8470;
                                    float arg_8471;
                                    float res_8472;
                                    
                                    y_8464 = res_8364 * index_primexp_8574;
                                    x_8465 = res_8408 - y_8464;
                                    res_8466 = x_8465 / res_8363;
                                    x_8467 = res_8415 - index_primexp_8574;
                                    x_8468 = fpow32(x_8467, 2.0F);
                                    x_8469 = res_8416 - res_8466;
                                    y_8470 = fpow32(x_8469, 2.0F);
                                    arg_8471 = x_8468 + y_8470;
                                    res_8472 = futrts_sqrt32(arg_8471);
                                    res_8463 = res_8472;
                                } else {
                                    float res_8473;
                                    
                                    if (res_8425) {
                                        float y_8474;
                                        float x_8475;
                                        float res_8476;
                                        float x_8477;
                                        float x_8478;
                                        float x_8479;
                                        float y_8480;
                                        float arg_8481;
                                        float res_8482;
                                        
                                        y_8474 = res_8364 * index_primexp_8574;
                                        x_8475 = res_8408 - y_8474;
                                        res_8476 = x_8475 / res_8363;
                                        x_8477 = res_8409 - index_primexp_8574;
                                        x_8478 = fpow32(x_8477, 2.0F);
                                        x_8479 = res_8410 - res_8476;
                                        y_8480 = fpow32(x_8479, 2.0F);
                                        arg_8481 = x_8478 + y_8480;
                                        res_8482 = futrts_sqrt32(arg_8481);
                                        res_8473 = res_8482;
                                    } else {
                                        float res_8483;
                                        
                                        if (cond_8430) {
                                            float y_8484;
                                            float x_8485;
                                            float res_8486;
                                            float x_8487;
                                            float x_8488;
                                            float x_8489;
                                            float y_8490;
                                            float arg_8491;
                                            float res_8492;
                                            
                                            y_8484 = res_8363 *
                                                index_primexp_8587;
                                            x_8485 = res_8408 - y_8484;
                                            res_8486 = x_8485 / res_8364;
                                            x_8487 = res_8415 - res_8486;
                                            x_8488 = fpow32(x_8487, 2.0F);
                                            x_8489 = res_8416 -
                                                index_primexp_8587;
                                            y_8490 = fpow32(x_8489, 2.0F);
                                            arg_8491 = x_8488 + y_8490;
                                            res_8492 = futrts_sqrt32(arg_8491);
                                            res_8483 = res_8492;
                                        } else {
                                            float res_8493;
                                            
                                            if (res_8431) {
                                                float y_8494;
                                                float x_8495;
                                                float res_8496;
                                                float x_8497;
                                                float x_8498;
                                                float x_8499;
                                                float y_8500;
                                                float arg_8501;
                                                float res_8502;
                                                
                                                y_8494 = res_8363 *
                                                    index_primexp_8587;
                                                x_8495 = res_8408 - y_8494;
                                                res_8496 = x_8495 / res_8364;
                                                x_8497 = res_8409 - res_8496;
                                                x_8498 = fpow32(x_8497, 2.0F);
                                                x_8499 = res_8410 -
                                                    index_primexp_8587;
                                                y_8500 = fpow32(x_8499, 2.0F);
                                                arg_8501 = x_8498 + y_8500;
                                                res_8502 =
                                                    futrts_sqrt32(arg_8501);
                                                res_8493 = res_8502;
                                            } else {
                                                float res_8503;
                                                
                                                if (cond_8436) {
                                                    float y_8504;
                                                    float x_8505;
                                                    float res_8506;
                                                    float x_8507;
                                                    float x_8508;
                                                    float x_8509;
                                                    float y_8510;
                                                    float arg_8511;
                                                    float res_8512;
                                                    
                                                    y_8504 = res_8363 *
                                                        index_primexp_8568;
                                                    x_8505 = res_8408 - y_8504;
                                                    res_8506 = x_8505 /
                                                        res_8364;
                                                    x_8507 = res_8415 -
                                                        res_8506;
                                                    x_8508 = fpow32(x_8507,
                                                                    2.0F);
                                                    x_8509 = res_8416 -
                                                        index_primexp_8568;
                                                    y_8510 = fpow32(x_8509,
                                                                    2.0F);
                                                    arg_8511 = x_8508 + y_8510;
                                                    res_8512 =
                                                        futrts_sqrt32(arg_8511);
                                                    res_8503 = res_8512;
                                                } else {
                                                    float res_8513;
                                                    
                                                    if (res_8437) {
                                                        float y_8514;
                                                        float x_8515;
                                                        float res_8516;
                                                        float x_8517;
                                                        float x_8518;
                                                        float x_8519;
                                                        float y_8520;
                                                        float arg_8521;
                                                        float res_8522;
                                                        
                                                        y_8514 = res_8363 *
                                                            index_primexp_8568;
                                                        x_8515 = res_8408 -
                                                            y_8514;
                                                        res_8516 = x_8515 /
                                                            res_8364;
                                                        x_8517 = res_8409 -
                                                            res_8516;
                                                        x_8518 = fpow32(x_8517,
                                                                        2.0F);
                                                        x_8519 = res_8410 -
                                                            index_primexp_8568;
                                                        y_8520 = fpow32(x_8519,
                                                                        2.0F);
                                                        arg_8521 = x_8518 +
                                                            y_8520;
                                                        res_8522 =
                                                            futrts_sqrt32(arg_8521);
                                                        res_8513 = res_8522;
                                                    } else {
                                                        float x_8523;
                                                        float x_8524;
                                                        float x_8525;
                                                        float y_8526;
                                                        float arg_8527;
                                                        float res_8528;
                                                        
                                                        x_8523 = res_8415 -
                                                            res_8409;
                                                        x_8524 = fpow32(x_8523,
                                                                        2.0F);
                                                        x_8525 = res_8416 -
                                                            res_8410;
                                                        y_8526 = fpow32(x_8525,
                                                                        2.0F);
                                                        arg_8527 = x_8524 +
                                                            y_8526;
                                                        res_8528 =
                                                            futrts_sqrt32(arg_8527);
                                                        res_8513 = res_8528;
                                                    }
                                                    res_8503 = res_8513;
                                                }
                                                res_8493 = res_8503;
                                            }
                                            res_8483 = res_8493;
                                        }
                                        res_8473 = res_8483;
                                    }
                                    res_8463 = res_8473;
                                }
                                res_8453 = res_8463;
                            }
                            res_8443 = res_8453;
                        }
                        res_8442 = res_8443;
                    }
                    x_8529 = res_8408 - res_7345;
                    arg_8530 = x_8529 / res_7347;
                    arg_8531 = futrts_round32(arg_8530);
                    res_8532 = fptosi_f32_i32(arg_8531);
                    res_8533 = x_8389 + res_8532;
                    y_8534 = *(__global
                               float *) &projections_mem_8598[res_8533 * 4];
                    res_8535 = res_8442 * y_8534;
                    res_8537 = acc_8402 + res_8535;
                    
                    float acc_tmp_8675 = res_8537;
                    
                    acc_8402 = acc_tmp_8675;
                }
            } else {
                for (int32_t i_8401 = 0; i_8401 < chunk_sizze_8395; i_8401++) {
                    int32_t convop_x_8560;
                    float res_8406;
                    float y_8407;
                    float res_8408;
                    float res_8409;
                    float res_8410;
                    float res_8415;
                    float res_8416;
                    bool cond_8421;
                    bool res_8422;
                    bool x_8423;
                    bool cond_8424;
                    bool res_8425;
                    bool x_8426;
                    bool x_8427;
                    bool y_8428;
                    bool cond_8429;
                    bool cond_8430;
                    bool res_8431;
                    bool x_8432;
                    bool x_8433;
                    bool y_8434;
                    bool cond_8435;
                    bool cond_8436;
                    bool res_8437;
                    bool x_8438;
                    bool x_8439;
                    bool y_8440;
                    bool cond_8441;
                    float res_8442;
                    float x_8529;
                    float arg_8530;
                    float arg_8531;
                    int32_t res_8532;
                    int32_t res_8533;
                    float y_8534;
                    float res_8535;
                    float res_8537;
                    
                    convop_x_8560 = chunk_offset_8396 + i_8401;
                    res_8406 = sitofp_i32_f32(convop_x_8560);
                    y_8407 = res_7347 * res_8406;
                    res_8408 = res_8385 + y_8407;
                    if (res_8388) {
                        float x_8411;
                        float res_8412;
                        
                        x_8411 = res_8408 - y_8390;
                        res_8412 = x_8411 / res_8363;
                        res_8409 = res_8355;
                        res_8410 = res_8412;
                    } else {
                        float x_8413;
                        float res_8414;
                        
                        x_8413 = res_8408 - y_8391;
                        res_8414 = x_8413 / res_8364;
                        res_8409 = res_8414;
                        res_8410 = res_8356;
                    }
                    if (res_8388) {
                        float x_8417;
                        float res_8418;
                        
                        x_8417 = res_8408 - y_8392;
                        res_8418 = x_8417 / res_8363;
                        res_8415 = res_8359;
                        res_8416 = res_8418;
                    } else {
                        float x_8419;
                        float res_8420;
                        
                        x_8419 = res_8408 - y_8393;
                        res_8420 = x_8419 / res_8364;
                        res_8415 = res_8420;
                        res_8416 = res_8360;
                    }
                    cond_8421 = res_8409 < index_primexp_8592;
                    res_8422 = res_8415 < index_primexp_8592;
                    x_8423 = cond_8421 && res_8422;
                    cond_8424 = index_primexp_8574 < res_8409;
                    res_8425 = index_primexp_8574 < res_8415;
                    x_8426 = cond_8424 && res_8425;
                    x_8427 = !x_8423;
                    y_8428 = x_8426 && x_8427;
                    cond_8429 = x_8423 || y_8428;
                    cond_8430 = res_8410 < index_primexp_8587;
                    res_8431 = res_8416 < index_primexp_8587;
                    x_8432 = cond_8430 && res_8431;
                    x_8433 = !cond_8429;
                    y_8434 = x_8432 && x_8433;
                    cond_8435 = cond_8429 || y_8434;
                    cond_8436 = index_primexp_8568 < res_8410;
                    res_8437 = index_primexp_8568 < res_8416;
                    x_8438 = cond_8436 && res_8437;
                    x_8439 = !cond_8435;
                    y_8440 = x_8438 && x_8439;
                    cond_8441 = cond_8435 || y_8440;
                    if (cond_8441) {
                        res_8442 = 0.0F;
                    } else {
                        float res_8443;
                        
                        if (cond_8421) {
                            float y_8444;
                            float x_8445;
                            float res_8446;
                            float x_8447;
                            float x_8448;
                            float x_8449;
                            float y_8450;
                            float arg_8451;
                            float res_8452;
                            
                            y_8444 = res_8364 * index_primexp_8592;
                            x_8445 = res_8408 - y_8444;
                            res_8446 = x_8445 / res_8363;
                            x_8447 = res_8415 - index_primexp_8592;
                            x_8448 = fpow32(x_8447, 2.0F);
                            x_8449 = res_8416 - res_8446;
                            y_8450 = fpow32(x_8449, 2.0F);
                            arg_8451 = x_8448 + y_8450;
                            res_8452 = futrts_sqrt32(arg_8451);
                            res_8443 = res_8452;
                        } else {
                            float res_8453;
                            
                            if (res_8422) {
                                float y_8454;
                                float x_8455;
                                float res_8456;
                                float x_8457;
                                float x_8458;
                                float x_8459;
                                float y_8460;
                                float arg_8461;
                                float res_8462;
                                
                                y_8454 = res_8364 * index_primexp_8592;
                                x_8455 = res_8408 - y_8454;
                                res_8456 = x_8455 / res_8363;
                                x_8457 = res_8409 - index_primexp_8592;
                                x_8458 = fpow32(x_8457, 2.0F);
                                x_8459 = res_8410 - res_8456;
                                y_8460 = fpow32(x_8459, 2.0F);
                                arg_8461 = x_8458 + y_8460;
                                res_8462 = futrts_sqrt32(arg_8461);
                                res_8453 = res_8462;
                            } else {
                                float res_8463;
                                
                                if (cond_8424) {
                                    float y_8464;
                                    float x_8465;
                                    float res_8466;
                                    float x_8467;
                                    float x_8468;
                                    float x_8469;
                                    float y_8470;
                                    float arg_8471;
                                    float res_8472;
                                    
                                    y_8464 = res_8364 * index_primexp_8574;
                                    x_8465 = res_8408 - y_8464;
                                    res_8466 = x_8465 / res_8363;
                                    x_8467 = res_8415 - index_primexp_8574;
                                    x_8468 = fpow32(x_8467, 2.0F);
                                    x_8469 = res_8416 - res_8466;
                                    y_8470 = fpow32(x_8469, 2.0F);
                                    arg_8471 = x_8468 + y_8470;
                                    res_8472 = futrts_sqrt32(arg_8471);
                                    res_8463 = res_8472;
                                } else {
                                    float res_8473;
                                    
                                    if (res_8425) {
                                        float y_8474;
                                        float x_8475;
                                        float res_8476;
                                        float x_8477;
                                        float x_8478;
                                        float x_8479;
                                        float y_8480;
                                        float arg_8481;
                                        float res_8482;
                                        
                                        y_8474 = res_8364 * index_primexp_8574;
                                        x_8475 = res_8408 - y_8474;
                                        res_8476 = x_8475 / res_8363;
                                        x_8477 = res_8409 - index_primexp_8574;
                                        x_8478 = fpow32(x_8477, 2.0F);
                                        x_8479 = res_8410 - res_8476;
                                        y_8480 = fpow32(x_8479, 2.0F);
                                        arg_8481 = x_8478 + y_8480;
                                        res_8482 = futrts_sqrt32(arg_8481);
                                        res_8473 = res_8482;
                                    } else {
                                        float res_8483;
                                        
                                        if (cond_8430) {
                                            float y_8484;
                                            float x_8485;
                                            float res_8486;
                                            float x_8487;
                                            float x_8488;
                                            float x_8489;
                                            float y_8490;
                                            float arg_8491;
                                            float res_8492;
                                            
                                            y_8484 = res_8363 *
                                                index_primexp_8587;
                                            x_8485 = res_8408 - y_8484;
                                            res_8486 = x_8485 / res_8364;
                                            x_8487 = res_8415 - res_8486;
                                            x_8488 = fpow32(x_8487, 2.0F);
                                            x_8489 = res_8416 -
                                                index_primexp_8587;
                                            y_8490 = fpow32(x_8489, 2.0F);
                                            arg_8491 = x_8488 + y_8490;
                                            res_8492 = futrts_sqrt32(arg_8491);
                                            res_8483 = res_8492;
                                        } else {
                                            float res_8493;
                                            
                                            if (res_8431) {
                                                float y_8494;
                                                float x_8495;
                                                float res_8496;
                                                float x_8497;
                                                float x_8498;
                                                float x_8499;
                                                float y_8500;
                                                float arg_8501;
                                                float res_8502;
                                                
                                                y_8494 = res_8363 *
                                                    index_primexp_8587;
                                                x_8495 = res_8408 - y_8494;
                                                res_8496 = x_8495 / res_8364;
                                                x_8497 = res_8409 - res_8496;
                                                x_8498 = fpow32(x_8497, 2.0F);
                                                x_8499 = res_8410 -
                                                    index_primexp_8587;
                                                y_8500 = fpow32(x_8499, 2.0F);
                                                arg_8501 = x_8498 + y_8500;
                                                res_8502 =
                                                    futrts_sqrt32(arg_8501);
                                                res_8493 = res_8502;
                                            } else {
                                                float res_8503;
                                                
                                                if (cond_8436) {
                                                    float y_8504;
                                                    float x_8505;
                                                    float res_8506;
                                                    float x_8507;
                                                    float x_8508;
                                                    float x_8509;
                                                    float y_8510;
                                                    float arg_8511;
                                                    float res_8512;
                                                    
                                                    y_8504 = res_8363 *
                                                        index_primexp_8568;
                                                    x_8505 = res_8408 - y_8504;
                                                    res_8506 = x_8505 /
                                                        res_8364;
                                                    x_8507 = res_8415 -
                                                        res_8506;
                                                    x_8508 = fpow32(x_8507,
                                                                    2.0F);
                                                    x_8509 = res_8416 -
                                                        index_primexp_8568;
                                                    y_8510 = fpow32(x_8509,
                                                                    2.0F);
                                                    arg_8511 = x_8508 + y_8510;
                                                    res_8512 =
                                                        futrts_sqrt32(arg_8511);
                                                    res_8503 = res_8512;
                                                } else {
                                                    float res_8513;
                                                    
                                                    if (res_8437) {
                                                        float y_8514;
                                                        float x_8515;
                                                        float res_8516;
                                                        float x_8517;
                                                        float x_8518;
                                                        float x_8519;
                                                        float y_8520;
                                                        float arg_8521;
                                                        float res_8522;
                                                        
                                                        y_8514 = res_8363 *
                                                            index_primexp_8568;
                                                        x_8515 = res_8408 -
                                                            y_8514;
                                                        res_8516 = x_8515 /
                                                            res_8364;
                                                        x_8517 = res_8409 -
                                                            res_8516;
                                                        x_8518 = fpow32(x_8517,
                                                                        2.0F);
                                                        x_8519 = res_8410 -
                                                            index_primexp_8568;
                                                        y_8520 = fpow32(x_8519,
                                                                        2.0F);
                                                        arg_8521 = x_8518 +
                                                            y_8520;
                                                        res_8522 =
                                                            futrts_sqrt32(arg_8521);
                                                        res_8513 = res_8522;
                                                    } else {
                                                        float x_8523;
                                                        float x_8524;
                                                        float x_8525;
                                                        float y_8526;
                                                        float arg_8527;
                                                        float res_8528;
                                                        
                                                        x_8523 = res_8415 -
                                                            res_8409;
                                                        x_8524 = fpow32(x_8523,
                                                                        2.0F);
                                                        x_8525 = res_8416 -
                                                            res_8410;
                                                        y_8526 = fpow32(x_8525,
                                                                        2.0F);
                                                        arg_8527 = x_8524 +
                                                            y_8526;
                                                        res_8528 =
                                                            futrts_sqrt32(arg_8527);
                                                        res_8513 = res_8528;
                                                    }
                                                    res_8503 = res_8513;
                                                }
                                                res_8493 = res_8503;
                                            }
                                            res_8483 = res_8493;
                                        }
                                        res_8473 = res_8483;
                                    }
                                    res_8463 = res_8473;
                                }
                                res_8453 = res_8463;
                            }
                            res_8443 = res_8453;
                        }
                        res_8442 = res_8443;
                    }
                    x_8529 = res_8408 - res_7345;
                    arg_8530 = x_8529 / res_7347;
                    arg_8531 = futrts_round32(arg_8530);
                    res_8532 = fptosi_f32_i32(arg_8531);
                    res_8533 = x_8389 + res_8532;
                    y_8534 = *(__global
                               float *) &projections_mem_8598[res_8533 * 4];
                    res_8535 = res_8442 * y_8534;
                    res_8537 = acc_8402 + res_8535;
                    
                    float acc_tmp_8676 = res_8537;
                    
                    acc_8402 = acc_tmp_8676;
                }
            }
            res_8399 = acc_8402;
            x_8397 = res_8399;
            res_8394 = x_8397;
            // save results to be reduced
            {
                x_8353 = res_8394;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                float res_8354 = x_8352 + x_8353;
                
                x_8352 = res_8354;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local float *) &red_arr_mem_8659[local_tid_8285 * 4] = x_8352;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_8677;
    int32_t skip_waves_8678;
    float x_8671;
    float x_8672;
    
    offset_8677 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_8285, group_sizze_8336)) {
            x_8671 = *(__local float *) &red_arr_mem_8659[(local_tid_8285 +
                                                           offset_8677) * 4];
        }
    }
    offset_8677 = 1;
    while (slt32(offset_8677, wave_sizze_8654)) {
        if (slt32(local_tid_8285 + offset_8677, group_sizze_8336) &&
            ((local_tid_8285 - squot32(local_tid_8285, wave_sizze_8654) *
              wave_sizze_8654) & (2 * offset_8677 - 1)) == 0) {
            // read array element
            {
                x_8672 = *(volatile __local
                           float *) &red_arr_mem_8659[(local_tid_8285 +
                                                       offset_8677) * 4];
            }
            // apply reduction operation
            {
                float res_8673 = x_8671 + x_8672;
                
                x_8671 = res_8673;
            }
            // write result of operation
            {
                *(volatile __local float *) &red_arr_mem_8659[local_tid_8285 *
                                                              4] = x_8671;
            }
        }
        offset_8677 *= 2;
    }
    skip_waves_8678 = 1;
    while (slt32(skip_waves_8678, squot32(group_sizze_8336 + wave_sizze_8654 -
                                          1, wave_sizze_8654))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_8677 = skip_waves_8678 * wave_sizze_8654;
        if (slt32(local_tid_8285 + offset_8677, group_sizze_8336) &&
            ((local_tid_8285 - squot32(local_tid_8285, wave_sizze_8654) *
              wave_sizze_8654) == 0 && (squot32(local_tid_8285,
                                                wave_sizze_8654) & (2 *
                                                                    skip_waves_8678 -
                                                                    1)) == 0)) {
            // read array element
            {
                x_8672 = *(__local float *) &red_arr_mem_8659[(local_tid_8285 +
                                                               offset_8677) *
                                                              4];
            }
            // apply reduction operation
            {
                float res_8673 = x_8671 + x_8672;
                
                x_8671 = res_8673;
            }
            // write result of operation
            {
                *(__local float *) &red_arr_mem_8659[local_tid_8285 * 4] =
                    x_8671;
            }
        }
        skip_waves_8678 *= 2;
    }
    if (squot32(num_groups_8346 + arg_7360 - 1, arg_7360) == 1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_8285 == 0) {
                *(__global float *) &mem_8610[gtid_8260 * 4] = x_8671;
            }
        }
    } else {
        int32_t old_counter_8679;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_8285 == 0) {
                *(__global float *) &group_res_arr_mem_8662[group_id_8286 * 4] =
                    x_8671;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                old_counter_8679 = atomic_add((volatile __global
                                               int *) &counter_mem_8665[srem32(squot32(group_id_8286,
                                                                                       squot32(num_groups_8346 +
                                                                                               arg_7360 -
                                                                                               1,
                                                                                               arg_7360)),
                                                                               1024) *
                                                                        4], 1);
                *(__local bool *) &sync_arr_mem_8667[0] = old_counter_8679 ==
                    squot32(num_groups_8346 + arg_7360 - 1, arg_7360) - 1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_8680 = *(__local bool *) &sync_arr_mem_8667[0];
        
        if (is_last_group_8680) {
            if (local_tid_8285 == 0) {
                old_counter_8679 = atomic_add((volatile __global
                                               int *) &counter_mem_8665[srem32(squot32(group_id_8286,
                                                                                       squot32(num_groups_8346 +
                                                                                               arg_7360 -
                                                                                               1,
                                                                                               arg_7360)),
                                                                               1024) *
                                                                        4], 0 -
                                              squot32(num_groups_8346 +
                                                      arg_7360 - 1, arg_7360));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_8285, squot32(num_groups_8346 + arg_7360 -
                                                  1, arg_7360))) {
                    x_8352 = *(__global
                               float *) &group_res_arr_mem_8662[(squot32(group_id_8286,
                                                                         squot32(num_groups_8346 +
                                                                                 arg_7360 -
                                                                                 1,
                                                                                 arg_7360)) *
                                                                 squot32(num_groups_8346 +
                                                                         arg_7360 -
                                                                         1,
                                                                         arg_7360) +
                                                                 local_tid_8285) *
                                                                4];
                } else {
                    x_8352 = 0.0F;
                }
                *(__local float *) &red_arr_mem_8659[local_tid_8285 * 4] =
                    x_8352;
            }
            // reduce the per-group results
            {
                int32_t offset_8681;
                int32_t skip_waves_8682;
                float x_8671;
                float x_8672;
                
                offset_8681 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_8285, group_sizze_8336)) {
                        x_8671 = *(__local
                                   float *) &red_arr_mem_8659[(local_tid_8285 +
                                                               offset_8681) *
                                                              4];
                    }
                }
                offset_8681 = 1;
                while (slt32(offset_8681, wave_sizze_8654)) {
                    if (slt32(local_tid_8285 + offset_8681, group_sizze_8336) &&
                        ((local_tid_8285 - squot32(local_tid_8285,
                                                   wave_sizze_8654) *
                          wave_sizze_8654) & (2 * offset_8681 - 1)) == 0) {
                        // read array element
                        {
                            x_8672 = *(volatile __local
                                       float *) &red_arr_mem_8659[(local_tid_8285 +
                                                                   offset_8681) *
                                                                  4];
                        }
                        // apply reduction operation
                        {
                            float res_8673 = x_8671 + x_8672;
                            
                            x_8671 = res_8673;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              float *) &red_arr_mem_8659[local_tid_8285 * 4] =
                                x_8671;
                        }
                    }
                    offset_8681 *= 2;
                }
                skip_waves_8682 = 1;
                while (slt32(skip_waves_8682, squot32(group_sizze_8336 +
                                                      wave_sizze_8654 - 1,
                                                      wave_sizze_8654))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_8681 = skip_waves_8682 * wave_sizze_8654;
                    if (slt32(local_tid_8285 + offset_8681, group_sizze_8336) &&
                        ((local_tid_8285 - squot32(local_tid_8285,
                                                   wave_sizze_8654) *
                          wave_sizze_8654) == 0 && (squot32(local_tid_8285,
                                                            wave_sizze_8654) &
                                                    (2 * skip_waves_8682 -
                                                     1)) == 0)) {
                        // read array element
                        {
                            x_8672 = *(__local
                                       float *) &red_arr_mem_8659[(local_tid_8285 +
                                                                   offset_8681) *
                                                                  4];
                        }
                        // apply reduction operation
                        {
                            float res_8673 = x_8671 + x_8672;
                            
                            x_8671 = res_8673;
                        }
                        // write result of operation
                        {
                            *(__local
                              float *) &red_arr_mem_8659[local_tid_8285 * 4] =
                                x_8671;
                        }
                    }
                    skip_waves_8682 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_8285 == 0) {
                        *(__global float *) &mem_8610[gtid_8260 * 4] = x_8671;
                    }
                }
            }
        }
    }
}
__kernel void segred_mapseg_8284(int32_t sizze_7338, int32_t sizze_7343,
                                 float res_7345, float res_7347,
                                 int32_t res_7359, int32_t arg_7360,
                                 int32_t y_7373, int32_t arg_7398,
                                 int32_t num_groups_8346, __global
                                 unsigned char *angles_mem_8594, __global
                                 unsigned char *projections_mem_8598, __global
                                 unsigned char *mem_8610)
{
    int32_t wave_sizze_8632;
    int32_t group_sizze_8633;
    int32_t gtid_8260;
    int32_t gtid_8283;
    int32_t global_tid_8284;
    int32_t local_tid_8285;
    int32_t group_id_8286;
    
    global_tid_8284 = get_global_id(0);
    local_tid_8285 = get_local_id(0);
    group_sizze_8633 = get_local_size(0);
    wave_sizze_8632 = LOCKSTEP_WIDTH;
    group_id_8286 = get_group_id(0);
    for (int32_t i_8634 = 0; i_8634 < squot32(squot32(arg_7360 +
                                                      group_sizze_8336 - 1,
                                                      group_sizze_8336) -
                                              group_id_8286 + num_groups_8346 -
                                              1, num_groups_8346); i_8634++) {
        gtid_8260 = squot32((group_id_8286 + i_8634 * num_groups_8346) *
                            group_sizze_8336 + local_tid_8285, sizze_7338);
        gtid_8283 = (group_id_8286 + i_8634 * num_groups_8346) *
            group_sizze_8336 + local_tid_8285 - squot32((group_id_8286 +
                                                         i_8634 *
                                                         num_groups_8346) *
                                                        group_sizze_8336 +
                                                        local_tid_8285,
                                                        sizze_7338) *
            sizze_7338;
        
        int32_t binop_y_8588;
        int32_t binop_y_8589;
        int32_t binop_x_8590;
        int32_t convop_x_8591;
        float index_primexp_8592;
        float res_8355;
        int32_t convop_x_8586;
        float index_primexp_8587;
        float res_8356;
        float index_primexp_8584;
        float index_primexp_8578;
        float index_primexp_8574;
        float res_8359;
        float index_primexp_8568;
        float res_8360;
        float res_8362;
        float res_8363;
        float res_8364;
        float y_8365;
        float res_8366;
        float y_8367;
        float res_8368;
        float x_8369;
        float y_8370;
        float res_8371;
        float x_8372;
        float arg_8373;
        int32_t res_8374;
        float res_8375;
        bool res_8376;
        float res_8377;
        float y_8384;
        float res_8385;
        float res_8386;
        float res_8387;
        bool res_8388;
        int32_t x_8389;
        float y_8390;
        float y_8391;
        float y_8392;
        float y_8393;
        
        if (slt32(gtid_8260, arg_7360) && slt32(gtid_8283, sizze_7338)) {
            binop_y_8588 = sdiv32(gtid_8260, sizze_7343);
            binop_y_8589 = sizze_7343 * binop_y_8588;
            binop_x_8590 = gtid_8260 - binop_y_8589;
            convop_x_8591 = binop_x_8590 - y_7373;
            index_primexp_8592 = sitofp_i32_f32(convop_x_8591);
            res_8355 = index_primexp_8592;
            convop_x_8586 = binop_y_8588 - y_7373;
            index_primexp_8587 = sitofp_i32_f32(convop_x_8586);
            res_8356 = index_primexp_8587;
            index_primexp_8584 = 0.5F + index_primexp_8592;
            index_primexp_8578 = 0.5F + index_primexp_8587;
            index_primexp_8574 = 1.0F + index_primexp_8592;
            res_8359 = index_primexp_8574;
            index_primexp_8568 = 1.0F + index_primexp_8587;
            res_8360 = index_primexp_8568;
            res_8362 = *(__global float *) &angles_mem_8594[gtid_8283 * 4];
            res_8363 = futrts_sin32(res_8362);
            res_8364 = futrts_cos32(res_8362);
            y_8365 = 0.70710677F * res_8364;
            res_8366 = index_primexp_8584 - y_8365;
            y_8367 = 0.70710677F * res_8363;
            res_8368 = index_primexp_8578 - y_8367;
            x_8369 = res_8364 * res_8366;
            y_8370 = res_8363 * res_8368;
            res_8371 = x_8369 + y_8370;
            x_8372 = res_8371 - res_7345;
            arg_8373 = x_8372 / res_7347;
            res_8374 = fptosi_f32_i32(arg_8373);
            res_8375 = sitofp_i32_f32(res_8374);
            res_8376 = 0.0F <= arg_8373;
            if (res_8376) {
                bool res_8378;
                float res_8379;
                
                res_8378 = res_8375 < arg_8373;
                if (res_8378) {
                    int32_t res_8380;
                    float res_8381;
                    
                    res_8380 = 1 + res_8374;
                    res_8381 = sitofp_i32_f32(res_8380);
                    res_8379 = res_8381;
                } else {
                    res_8379 = arg_8373;
                }
                res_8377 = res_8379;
            } else {
                bool res_8382;
                float res_8383;
                
                res_8382 = arg_8373 < res_8375;
                if (res_8382) {
                    res_8383 = res_8375;
                } else {
                    res_8383 = arg_8373;
                }
                res_8377 = res_8383;
            }
            y_8384 = res_7347 * res_8377;
            res_8385 = res_7345 + y_8384;
            res_8386 = (float) fabs(res_8363);
            res_8387 = (float) fabs(res_8364);
            res_8388 = res_8387 <= res_8386;
            x_8389 = arg_7398 * gtid_8283;
            y_8390 = res_8364 * index_primexp_8592;
            y_8391 = res_8363 * index_primexp_8587;
            y_8392 = res_8364 * index_primexp_8574;
            y_8393 = res_8363 * index_primexp_8568;
        }
        
        float res_8394;
        float x_8397 = 0.0F;
        int32_t chunk_sizze_8395;
        int32_t chunk_offset_8396 = 0;
        
        chunk_sizze_8395 = res_7359;
        
        float res_8399;
        float acc_8402 = x_8397;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_8400;
        
        groupstream_mapaccum_dummy_chunk_sizze_8400 = 1;
        if (slt32(gtid_8260, arg_7360) && slt32(gtid_8283, sizze_7338)) {
            if (chunk_sizze_8395 == res_7359) {
                for (int32_t i_8401 = 0; i_8401 < res_7359; i_8401++) {
                    int32_t convop_x_8560;
                    float res_8406;
                    float y_8407;
                    float res_8408;
                    float res_8409;
                    float res_8410;
                    float res_8415;
                    float res_8416;
                    bool cond_8421;
                    bool res_8422;
                    bool x_8423;
                    bool cond_8424;
                    bool res_8425;
                    bool x_8426;
                    bool x_8427;
                    bool y_8428;
                    bool cond_8429;
                    bool cond_8430;
                    bool res_8431;
                    bool x_8432;
                    bool x_8433;
                    bool y_8434;
                    bool cond_8435;
                    bool cond_8436;
                    bool res_8437;
                    bool x_8438;
                    bool x_8439;
                    bool y_8440;
                    bool cond_8441;
                    float res_8442;
                    float x_8529;
                    float arg_8530;
                    float arg_8531;
                    int32_t res_8532;
                    int32_t res_8533;
                    float y_8534;
                    float res_8535;
                    float res_8537;
                    
                    convop_x_8560 = chunk_offset_8396 + i_8401;
                    res_8406 = sitofp_i32_f32(convop_x_8560);
                    y_8407 = res_7347 * res_8406;
                    res_8408 = res_8385 + y_8407;
                    if (res_8388) {
                        float x_8411;
                        float res_8412;
                        
                        x_8411 = res_8408 - y_8390;
                        res_8412 = x_8411 / res_8363;
                        res_8409 = res_8355;
                        res_8410 = res_8412;
                    } else {
                        float x_8413;
                        float res_8414;
                        
                        x_8413 = res_8408 - y_8391;
                        res_8414 = x_8413 / res_8364;
                        res_8409 = res_8414;
                        res_8410 = res_8356;
                    }
                    if (res_8388) {
                        float x_8417;
                        float res_8418;
                        
                        x_8417 = res_8408 - y_8392;
                        res_8418 = x_8417 / res_8363;
                        res_8415 = res_8359;
                        res_8416 = res_8418;
                    } else {
                        float x_8419;
                        float res_8420;
                        
                        x_8419 = res_8408 - y_8393;
                        res_8420 = x_8419 / res_8364;
                        res_8415 = res_8420;
                        res_8416 = res_8360;
                    }
                    cond_8421 = res_8409 < index_primexp_8592;
                    res_8422 = res_8415 < index_primexp_8592;
                    x_8423 = cond_8421 && res_8422;
                    cond_8424 = index_primexp_8574 < res_8409;
                    res_8425 = index_primexp_8574 < res_8415;
                    x_8426 = cond_8424 && res_8425;
                    x_8427 = !x_8423;
                    y_8428 = x_8426 && x_8427;
                    cond_8429 = x_8423 || y_8428;
                    cond_8430 = res_8410 < index_primexp_8587;
                    res_8431 = res_8416 < index_primexp_8587;
                    x_8432 = cond_8430 && res_8431;
                    x_8433 = !cond_8429;
                    y_8434 = x_8432 && x_8433;
                    cond_8435 = cond_8429 || y_8434;
                    cond_8436 = index_primexp_8568 < res_8410;
                    res_8437 = index_primexp_8568 < res_8416;
                    x_8438 = cond_8436 && res_8437;
                    x_8439 = !cond_8435;
                    y_8440 = x_8438 && x_8439;
                    cond_8441 = cond_8435 || y_8440;
                    if (cond_8441) {
                        res_8442 = 0.0F;
                    } else {
                        float res_8443;
                        
                        if (cond_8421) {
                            float y_8444;
                            float x_8445;
                            float res_8446;
                            float x_8447;
                            float x_8448;
                            float x_8449;
                            float y_8450;
                            float arg_8451;
                            float res_8452;
                            
                            y_8444 = res_8364 * index_primexp_8592;
                            x_8445 = res_8408 - y_8444;
                            res_8446 = x_8445 / res_8363;
                            x_8447 = res_8415 - index_primexp_8592;
                            x_8448 = fpow32(x_8447, 2.0F);
                            x_8449 = res_8416 - res_8446;
                            y_8450 = fpow32(x_8449, 2.0F);
                            arg_8451 = x_8448 + y_8450;
                            res_8452 = futrts_sqrt32(arg_8451);
                            res_8443 = res_8452;
                        } else {
                            float res_8453;
                            
                            if (res_8422) {
                                float y_8454;
                                float x_8455;
                                float res_8456;
                                float x_8457;
                                float x_8458;
                                float x_8459;
                                float y_8460;
                                float arg_8461;
                                float res_8462;
                                
                                y_8454 = res_8364 * index_primexp_8592;
                                x_8455 = res_8408 - y_8454;
                                res_8456 = x_8455 / res_8363;
                                x_8457 = res_8409 - index_primexp_8592;
                                x_8458 = fpow32(x_8457, 2.0F);
                                x_8459 = res_8410 - res_8456;
                                y_8460 = fpow32(x_8459, 2.0F);
                                arg_8461 = x_8458 + y_8460;
                                res_8462 = futrts_sqrt32(arg_8461);
                                res_8453 = res_8462;
                            } else {
                                float res_8463;
                                
                                if (cond_8424) {
                                    float y_8464;
                                    float x_8465;
                                    float res_8466;
                                    float x_8467;
                                    float x_8468;
                                    float x_8469;
                                    float y_8470;
                                    float arg_8471;
                                    float res_8472;
                                    
                                    y_8464 = res_8364 * index_primexp_8574;
                                    x_8465 = res_8408 - y_8464;
                                    res_8466 = x_8465 / res_8363;
                                    x_8467 = res_8415 - index_primexp_8574;
                                    x_8468 = fpow32(x_8467, 2.0F);
                                    x_8469 = res_8416 - res_8466;
                                    y_8470 = fpow32(x_8469, 2.0F);
                                    arg_8471 = x_8468 + y_8470;
                                    res_8472 = futrts_sqrt32(arg_8471);
                                    res_8463 = res_8472;
                                } else {
                                    float res_8473;
                                    
                                    if (res_8425) {
                                        float y_8474;
                                        float x_8475;
                                        float res_8476;
                                        float x_8477;
                                        float x_8478;
                                        float x_8479;
                                        float y_8480;
                                        float arg_8481;
                                        float res_8482;
                                        
                                        y_8474 = res_8364 * index_primexp_8574;
                                        x_8475 = res_8408 - y_8474;
                                        res_8476 = x_8475 / res_8363;
                                        x_8477 = res_8409 - index_primexp_8574;
                                        x_8478 = fpow32(x_8477, 2.0F);
                                        x_8479 = res_8410 - res_8476;
                                        y_8480 = fpow32(x_8479, 2.0F);
                                        arg_8481 = x_8478 + y_8480;
                                        res_8482 = futrts_sqrt32(arg_8481);
                                        res_8473 = res_8482;
                                    } else {
                                        float res_8483;
                                        
                                        if (cond_8430) {
                                            float y_8484;
                                            float x_8485;
                                            float res_8486;
                                            float x_8487;
                                            float x_8488;
                                            float x_8489;
                                            float y_8490;
                                            float arg_8491;
                                            float res_8492;
                                            
                                            y_8484 = res_8363 *
                                                index_primexp_8587;
                                            x_8485 = res_8408 - y_8484;
                                            res_8486 = x_8485 / res_8364;
                                            x_8487 = res_8415 - res_8486;
                                            x_8488 = fpow32(x_8487, 2.0F);
                                            x_8489 = res_8416 -
                                                index_primexp_8587;
                                            y_8490 = fpow32(x_8489, 2.0F);
                                            arg_8491 = x_8488 + y_8490;
                                            res_8492 = futrts_sqrt32(arg_8491);
                                            res_8483 = res_8492;
                                        } else {
                                            float res_8493;
                                            
                                            if (res_8431) {
                                                float y_8494;
                                                float x_8495;
                                                float res_8496;
                                                float x_8497;
                                                float x_8498;
                                                float x_8499;
                                                float y_8500;
                                                float arg_8501;
                                                float res_8502;
                                                
                                                y_8494 = res_8363 *
                                                    index_primexp_8587;
                                                x_8495 = res_8408 - y_8494;
                                                res_8496 = x_8495 / res_8364;
                                                x_8497 = res_8409 - res_8496;
                                                x_8498 = fpow32(x_8497, 2.0F);
                                                x_8499 = res_8410 -
                                                    index_primexp_8587;
                                                y_8500 = fpow32(x_8499, 2.0F);
                                                arg_8501 = x_8498 + y_8500;
                                                res_8502 =
                                                    futrts_sqrt32(arg_8501);
                                                res_8493 = res_8502;
                                            } else {
                                                float res_8503;
                                                
                                                if (cond_8436) {
                                                    float y_8504;
                                                    float x_8505;
                                                    float res_8506;
                                                    float x_8507;
                                                    float x_8508;
                                                    float x_8509;
                                                    float y_8510;
                                                    float arg_8511;
                                                    float res_8512;
                                                    
                                                    y_8504 = res_8363 *
                                                        index_primexp_8568;
                                                    x_8505 = res_8408 - y_8504;
                                                    res_8506 = x_8505 /
                                                        res_8364;
                                                    x_8507 = res_8415 -
                                                        res_8506;
                                                    x_8508 = fpow32(x_8507,
                                                                    2.0F);
                                                    x_8509 = res_8416 -
                                                        index_primexp_8568;
                                                    y_8510 = fpow32(x_8509,
                                                                    2.0F);
                                                    arg_8511 = x_8508 + y_8510;
                                                    res_8512 =
                                                        futrts_sqrt32(arg_8511);
                                                    res_8503 = res_8512;
                                                } else {
                                                    float res_8513;
                                                    
                                                    if (res_8437) {
                                                        float y_8514;
                                                        float x_8515;
                                                        float res_8516;
                                                        float x_8517;
                                                        float x_8518;
                                                        float x_8519;
                                                        float y_8520;
                                                        float arg_8521;
                                                        float res_8522;
                                                        
                                                        y_8514 = res_8363 *
                                                            index_primexp_8568;
                                                        x_8515 = res_8408 -
                                                            y_8514;
                                                        res_8516 = x_8515 /
                                                            res_8364;
                                                        x_8517 = res_8409 -
                                                            res_8516;
                                                        x_8518 = fpow32(x_8517,
                                                                        2.0F);
                                                        x_8519 = res_8410 -
                                                            index_primexp_8568;
                                                        y_8520 = fpow32(x_8519,
                                                                        2.0F);
                                                        arg_8521 = x_8518 +
                                                            y_8520;
                                                        res_8522 =
                                                            futrts_sqrt32(arg_8521);
                                                        res_8513 = res_8522;
                                                    } else {
                                                        float x_8523;
                                                        float x_8524;
                                                        float x_8525;
                                                        float y_8526;
                                                        float arg_8527;
                                                        float res_8528;
                                                        
                                                        x_8523 = res_8415 -
                                                            res_8409;
                                                        x_8524 = fpow32(x_8523,
                                                                        2.0F);
                                                        x_8525 = res_8416 -
                                                            res_8410;
                                                        y_8526 = fpow32(x_8525,
                                                                        2.0F);
                                                        arg_8527 = x_8524 +
                                                            y_8526;
                                                        res_8528 =
                                                            futrts_sqrt32(arg_8527);
                                                        res_8513 = res_8528;
                                                    }
                                                    res_8503 = res_8513;
                                                }
                                                res_8493 = res_8503;
                                            }
                                            res_8483 = res_8493;
                                        }
                                        res_8473 = res_8483;
                                    }
                                    res_8463 = res_8473;
                                }
                                res_8453 = res_8463;
                            }
                            res_8443 = res_8453;
                        }
                        res_8442 = res_8443;
                    }
                    x_8529 = res_8408 - res_7345;
                    arg_8530 = x_8529 / res_7347;
                    arg_8531 = futrts_round32(arg_8530);
                    res_8532 = fptosi_f32_i32(arg_8531);
                    res_8533 = x_8389 + res_8532;
                    y_8534 = *(__global
                               float *) &projections_mem_8598[res_8533 * 4];
                    res_8535 = res_8442 * y_8534;
                    res_8537 = acc_8402 + res_8535;
                    
                    float acc_tmp_8635 = res_8537;
                    
                    acc_8402 = acc_tmp_8635;
                }
            } else {
                for (int32_t i_8401 = 0; i_8401 < chunk_sizze_8395; i_8401++) {
                    int32_t convop_x_8560;
                    float res_8406;
                    float y_8407;
                    float res_8408;
                    float res_8409;
                    float res_8410;
                    float res_8415;
                    float res_8416;
                    bool cond_8421;
                    bool res_8422;
                    bool x_8423;
                    bool cond_8424;
                    bool res_8425;
                    bool x_8426;
                    bool x_8427;
                    bool y_8428;
                    bool cond_8429;
                    bool cond_8430;
                    bool res_8431;
                    bool x_8432;
                    bool x_8433;
                    bool y_8434;
                    bool cond_8435;
                    bool cond_8436;
                    bool res_8437;
                    bool x_8438;
                    bool x_8439;
                    bool y_8440;
                    bool cond_8441;
                    float res_8442;
                    float x_8529;
                    float arg_8530;
                    float arg_8531;
                    int32_t res_8532;
                    int32_t res_8533;
                    float y_8534;
                    float res_8535;
                    float res_8537;
                    
                    convop_x_8560 = chunk_offset_8396 + i_8401;
                    res_8406 = sitofp_i32_f32(convop_x_8560);
                    y_8407 = res_7347 * res_8406;
                    res_8408 = res_8385 + y_8407;
                    if (res_8388) {
                        float x_8411;
                        float res_8412;
                        
                        x_8411 = res_8408 - y_8390;
                        res_8412 = x_8411 / res_8363;
                        res_8409 = res_8355;
                        res_8410 = res_8412;
                    } else {
                        float x_8413;
                        float res_8414;
                        
                        x_8413 = res_8408 - y_8391;
                        res_8414 = x_8413 / res_8364;
                        res_8409 = res_8414;
                        res_8410 = res_8356;
                    }
                    if (res_8388) {
                        float x_8417;
                        float res_8418;
                        
                        x_8417 = res_8408 - y_8392;
                        res_8418 = x_8417 / res_8363;
                        res_8415 = res_8359;
                        res_8416 = res_8418;
                    } else {
                        float x_8419;
                        float res_8420;
                        
                        x_8419 = res_8408 - y_8393;
                        res_8420 = x_8419 / res_8364;
                        res_8415 = res_8420;
                        res_8416 = res_8360;
                    }
                    cond_8421 = res_8409 < index_primexp_8592;
                    res_8422 = res_8415 < index_primexp_8592;
                    x_8423 = cond_8421 && res_8422;
                    cond_8424 = index_primexp_8574 < res_8409;
                    res_8425 = index_primexp_8574 < res_8415;
                    x_8426 = cond_8424 && res_8425;
                    x_8427 = !x_8423;
                    y_8428 = x_8426 && x_8427;
                    cond_8429 = x_8423 || y_8428;
                    cond_8430 = res_8410 < index_primexp_8587;
                    res_8431 = res_8416 < index_primexp_8587;
                    x_8432 = cond_8430 && res_8431;
                    x_8433 = !cond_8429;
                    y_8434 = x_8432 && x_8433;
                    cond_8435 = cond_8429 || y_8434;
                    cond_8436 = index_primexp_8568 < res_8410;
                    res_8437 = index_primexp_8568 < res_8416;
                    x_8438 = cond_8436 && res_8437;
                    x_8439 = !cond_8435;
                    y_8440 = x_8438 && x_8439;
                    cond_8441 = cond_8435 || y_8440;
                    if (cond_8441) {
                        res_8442 = 0.0F;
                    } else {
                        float res_8443;
                        
                        if (cond_8421) {
                            float y_8444;
                            float x_8445;
                            float res_8446;
                            float x_8447;
                            float x_8448;
                            float x_8449;
                            float y_8450;
                            float arg_8451;
                            float res_8452;
                            
                            y_8444 = res_8364 * index_primexp_8592;
                            x_8445 = res_8408 - y_8444;
                            res_8446 = x_8445 / res_8363;
                            x_8447 = res_8415 - index_primexp_8592;
                            x_8448 = fpow32(x_8447, 2.0F);
                            x_8449 = res_8416 - res_8446;
                            y_8450 = fpow32(x_8449, 2.0F);
                            arg_8451 = x_8448 + y_8450;
                            res_8452 = futrts_sqrt32(arg_8451);
                            res_8443 = res_8452;
                        } else {
                            float res_8453;
                            
                            if (res_8422) {
                                float y_8454;
                                float x_8455;
                                float res_8456;
                                float x_8457;
                                float x_8458;
                                float x_8459;
                                float y_8460;
                                float arg_8461;
                                float res_8462;
                                
                                y_8454 = res_8364 * index_primexp_8592;
                                x_8455 = res_8408 - y_8454;
                                res_8456 = x_8455 / res_8363;
                                x_8457 = res_8409 - index_primexp_8592;
                                x_8458 = fpow32(x_8457, 2.0F);
                                x_8459 = res_8410 - res_8456;
                                y_8460 = fpow32(x_8459, 2.0F);
                                arg_8461 = x_8458 + y_8460;
                                res_8462 = futrts_sqrt32(arg_8461);
                                res_8453 = res_8462;
                            } else {
                                float res_8463;
                                
                                if (cond_8424) {
                                    float y_8464;
                                    float x_8465;
                                    float res_8466;
                                    float x_8467;
                                    float x_8468;
                                    float x_8469;
                                    float y_8470;
                                    float arg_8471;
                                    float res_8472;
                                    
                                    y_8464 = res_8364 * index_primexp_8574;
                                    x_8465 = res_8408 - y_8464;
                                    res_8466 = x_8465 / res_8363;
                                    x_8467 = res_8415 - index_primexp_8574;
                                    x_8468 = fpow32(x_8467, 2.0F);
                                    x_8469 = res_8416 - res_8466;
                                    y_8470 = fpow32(x_8469, 2.0F);
                                    arg_8471 = x_8468 + y_8470;
                                    res_8472 = futrts_sqrt32(arg_8471);
                                    res_8463 = res_8472;
                                } else {
                                    float res_8473;
                                    
                                    if (res_8425) {
                                        float y_8474;
                                        float x_8475;
                                        float res_8476;
                                        float x_8477;
                                        float x_8478;
                                        float x_8479;
                                        float y_8480;
                                        float arg_8481;
                                        float res_8482;
                                        
                                        y_8474 = res_8364 * index_primexp_8574;
                                        x_8475 = res_8408 - y_8474;
                                        res_8476 = x_8475 / res_8363;
                                        x_8477 = res_8409 - index_primexp_8574;
                                        x_8478 = fpow32(x_8477, 2.0F);
                                        x_8479 = res_8410 - res_8476;
                                        y_8480 = fpow32(x_8479, 2.0F);
                                        arg_8481 = x_8478 + y_8480;
                                        res_8482 = futrts_sqrt32(arg_8481);
                                        res_8473 = res_8482;
                                    } else {
                                        float res_8483;
                                        
                                        if (cond_8430) {
                                            float y_8484;
                                            float x_8485;
                                            float res_8486;
                                            float x_8487;
                                            float x_8488;
                                            float x_8489;
                                            float y_8490;
                                            float arg_8491;
                                            float res_8492;
                                            
                                            y_8484 = res_8363 *
                                                index_primexp_8587;
                                            x_8485 = res_8408 - y_8484;
                                            res_8486 = x_8485 / res_8364;
                                            x_8487 = res_8415 - res_8486;
                                            x_8488 = fpow32(x_8487, 2.0F);
                                            x_8489 = res_8416 -
                                                index_primexp_8587;
                                            y_8490 = fpow32(x_8489, 2.0F);
                                            arg_8491 = x_8488 + y_8490;
                                            res_8492 = futrts_sqrt32(arg_8491);
                                            res_8483 = res_8492;
                                        } else {
                                            float res_8493;
                                            
                                            if (res_8431) {
                                                float y_8494;
                                                float x_8495;
                                                float res_8496;
                                                float x_8497;
                                                float x_8498;
                                                float x_8499;
                                                float y_8500;
                                                float arg_8501;
                                                float res_8502;
                                                
                                                y_8494 = res_8363 *
                                                    index_primexp_8587;
                                                x_8495 = res_8408 - y_8494;
                                                res_8496 = x_8495 / res_8364;
                                                x_8497 = res_8409 - res_8496;
                                                x_8498 = fpow32(x_8497, 2.0F);
                                                x_8499 = res_8410 -
                                                    index_primexp_8587;
                                                y_8500 = fpow32(x_8499, 2.0F);
                                                arg_8501 = x_8498 + y_8500;
                                                res_8502 =
                                                    futrts_sqrt32(arg_8501);
                                                res_8493 = res_8502;
                                            } else {
                                                float res_8503;
                                                
                                                if (cond_8436) {
                                                    float y_8504;
                                                    float x_8505;
                                                    float res_8506;
                                                    float x_8507;
                                                    float x_8508;
                                                    float x_8509;
                                                    float y_8510;
                                                    float arg_8511;
                                                    float res_8512;
                                                    
                                                    y_8504 = res_8363 *
                                                        index_primexp_8568;
                                                    x_8505 = res_8408 - y_8504;
                                                    res_8506 = x_8505 /
                                                        res_8364;
                                                    x_8507 = res_8415 -
                                                        res_8506;
                                                    x_8508 = fpow32(x_8507,
                                                                    2.0F);
                                                    x_8509 = res_8416 -
                                                        index_primexp_8568;
                                                    y_8510 = fpow32(x_8509,
                                                                    2.0F);
                                                    arg_8511 = x_8508 + y_8510;
                                                    res_8512 =
                                                        futrts_sqrt32(arg_8511);
                                                    res_8503 = res_8512;
                                                } else {
                                                    float res_8513;
                                                    
                                                    if (res_8437) {
                                                        float y_8514;
                                                        float x_8515;
                                                        float res_8516;
                                                        float x_8517;
                                                        float x_8518;
                                                        float x_8519;
                                                        float y_8520;
                                                        float arg_8521;
                                                        float res_8522;
                                                        
                                                        y_8514 = res_8363 *
                                                            index_primexp_8568;
                                                        x_8515 = res_8408 -
                                                            y_8514;
                                                        res_8516 = x_8515 /
                                                            res_8364;
                                                        x_8517 = res_8409 -
                                                            res_8516;
                                                        x_8518 = fpow32(x_8517,
                                                                        2.0F);
                                                        x_8519 = res_8410 -
                                                            index_primexp_8568;
                                                        y_8520 = fpow32(x_8519,
                                                                        2.0F);
                                                        arg_8521 = x_8518 +
                                                            y_8520;
                                                        res_8522 =
                                                            futrts_sqrt32(arg_8521);
                                                        res_8513 = res_8522;
                                                    } else {
                                                        float x_8523;
                                                        float x_8524;
                                                        float x_8525;
                                                        float y_8526;
                                                        float arg_8527;
                                                        float res_8528;
                                                        
                                                        x_8523 = res_8415 -
                                                            res_8409;
                                                        x_8524 = fpow32(x_8523,
                                                                        2.0F);
                                                        x_8525 = res_8416 -
                                                            res_8410;
                                                        y_8526 = fpow32(x_8525,
                                                                        2.0F);
                                                        arg_8527 = x_8524 +
                                                            y_8526;
                                                        res_8528 =
                                                            futrts_sqrt32(arg_8527);
                                                        res_8513 = res_8528;
                                                    }
                                                    res_8503 = res_8513;
                                                }
                                                res_8493 = res_8503;
                                            }
                                            res_8483 = res_8493;
                                        }
                                        res_8473 = res_8483;
                                    }
                                    res_8463 = res_8473;
                                }
                                res_8453 = res_8463;
                            }
                            res_8443 = res_8453;
                        }
                        res_8442 = res_8443;
                    }
                    x_8529 = res_8408 - res_7345;
                    arg_8530 = x_8529 / res_7347;
                    arg_8531 = futrts_round32(arg_8530);
                    res_8532 = fptosi_f32_i32(arg_8531);
                    res_8533 = x_8389 + res_8532;
                    y_8534 = *(__global
                               float *) &projections_mem_8598[res_8533 * 4];
                    res_8535 = res_8442 * y_8534;
                    res_8537 = acc_8402 + res_8535;
                    
                    float acc_tmp_8636 = res_8537;
                    
                    acc_8402 = acc_tmp_8636;
                }
            }
        }
        res_8399 = acc_8402;
        x_8397 = res_8399;
        res_8394 = x_8397;
        if (slt32(gtid_8260, arg_7360) && slt32(gtid_8283, sizze_7338)) {
            *(__global float *) &mem_8610[gtid_8260 * 4] = res_8394;
        }
    }
}
__kernel void segred_small_8284(__local volatile int64_t *red_arr_mem_aligned_0,
                                int32_t sizze_7338, int32_t sizze_7343,
                                float res_7345, float res_7347,
                                int32_t res_7359, int32_t arg_7360,
                                int32_t y_7373, int32_t arg_7398,
                                int32_t num_groups_8346, __global
                                unsigned char *angles_mem_8594, __global
                                unsigned char *projections_mem_8598, __global
                                unsigned char *mem_8610)
{
    __local volatile char *restrict red_arr_mem_8639 = red_arr_mem_aligned_0;
    int32_t wave_sizze_8637;
    int32_t group_sizze_8638;
    int32_t gtid_8260;
    int32_t gtid_8283;
    int32_t global_tid_8284;
    int32_t local_tid_8285;
    int32_t group_id_8286;
    
    global_tid_8284 = get_global_id(0);
    local_tid_8285 = get_local_id(0);
    group_sizze_8638 = get_local_size(0);
    wave_sizze_8637 = LOCKSTEP_WIDTH;
    group_id_8286 = get_group_id(0);
    for (int32_t i_8642 = 0; i_8642 < squot32(squot32(arg_7360 +
                                                      squot32(group_sizze_8336,
                                                              sizze_7338) - 1,
                                                      squot32(group_sizze_8336,
                                                              sizze_7338)) -
                                              group_id_8286 + num_groups_8346 -
                                              1, num_groups_8346); i_8642++) {
        gtid_8260 = squot32(local_tid_8285, sizze_7338) + (group_id_8286 +
                                                           i_8642 *
                                                           num_groups_8346) *
            squot32(group_sizze_8336, sizze_7338);
        gtid_8283 = srem32(local_tid_8285, sizze_7338);
        // apply map function if in bounds
        {
            if (slt32(gtid_8260, arg_7360) && slt32(local_tid_8285, sizze_7338 *
                                                    squot32(group_sizze_8336,
                                                            sizze_7338))) {
                int32_t binop_y_8588;
                int32_t binop_y_8589;
                int32_t binop_x_8590;
                int32_t convop_x_8591;
                float index_primexp_8592;
                float res_8355;
                int32_t convop_x_8586;
                float index_primexp_8587;
                float res_8356;
                float index_primexp_8584;
                float index_primexp_8578;
                float index_primexp_8574;
                float res_8359;
                float index_primexp_8568;
                float res_8360;
                float res_8362;
                float res_8363;
                float res_8364;
                float y_8365;
                float res_8366;
                float y_8367;
                float res_8368;
                float x_8369;
                float y_8370;
                float res_8371;
                float x_8372;
                float arg_8373;
                int32_t res_8374;
                float res_8375;
                bool res_8376;
                float res_8377;
                float y_8384;
                float res_8385;
                float res_8386;
                float res_8387;
                bool res_8388;
                int32_t x_8389;
                float y_8390;
                float y_8391;
                float y_8392;
                float y_8393;
                
                binop_y_8588 = sdiv32(gtid_8260, sizze_7343);
                binop_y_8589 = sizze_7343 * binop_y_8588;
                binop_x_8590 = gtid_8260 - binop_y_8589;
                convop_x_8591 = binop_x_8590 - y_7373;
                index_primexp_8592 = sitofp_i32_f32(convop_x_8591);
                res_8355 = index_primexp_8592;
                convop_x_8586 = binop_y_8588 - y_7373;
                index_primexp_8587 = sitofp_i32_f32(convop_x_8586);
                res_8356 = index_primexp_8587;
                index_primexp_8584 = 0.5F + index_primexp_8592;
                index_primexp_8578 = 0.5F + index_primexp_8587;
                index_primexp_8574 = 1.0F + index_primexp_8592;
                res_8359 = index_primexp_8574;
                index_primexp_8568 = 1.0F + index_primexp_8587;
                res_8360 = index_primexp_8568;
                res_8362 = *(__global float *) &angles_mem_8594[gtid_8283 * 4];
                res_8363 = futrts_sin32(res_8362);
                res_8364 = futrts_cos32(res_8362);
                y_8365 = 0.70710677F * res_8364;
                res_8366 = index_primexp_8584 - y_8365;
                y_8367 = 0.70710677F * res_8363;
                res_8368 = index_primexp_8578 - y_8367;
                x_8369 = res_8364 * res_8366;
                y_8370 = res_8363 * res_8368;
                res_8371 = x_8369 + y_8370;
                x_8372 = res_8371 - res_7345;
                arg_8373 = x_8372 / res_7347;
                res_8374 = fptosi_f32_i32(arg_8373);
                res_8375 = sitofp_i32_f32(res_8374);
                res_8376 = 0.0F <= arg_8373;
                if (res_8376) {
                    bool res_8378;
                    float res_8379;
                    
                    res_8378 = res_8375 < arg_8373;
                    if (res_8378) {
                        int32_t res_8380;
                        float res_8381;
                        
                        res_8380 = 1 + res_8374;
                        res_8381 = sitofp_i32_f32(res_8380);
                        res_8379 = res_8381;
                    } else {
                        res_8379 = arg_8373;
                    }
                    res_8377 = res_8379;
                } else {
                    bool res_8382;
                    float res_8383;
                    
                    res_8382 = arg_8373 < res_8375;
                    if (res_8382) {
                        res_8383 = res_8375;
                    } else {
                        res_8383 = arg_8373;
                    }
                    res_8377 = res_8383;
                }
                y_8384 = res_7347 * res_8377;
                res_8385 = res_7345 + y_8384;
                res_8386 = (float) fabs(res_8363);
                res_8387 = (float) fabs(res_8364);
                res_8388 = res_8387 <= res_8386;
                x_8389 = arg_7398 * gtid_8283;
                y_8390 = res_8364 * index_primexp_8592;
                y_8391 = res_8363 * index_primexp_8587;
                y_8392 = res_8364 * index_primexp_8574;
                y_8393 = res_8363 * index_primexp_8568;
                
                float res_8394;
                float x_8397 = 0.0F;
                int32_t chunk_sizze_8395;
                int32_t chunk_offset_8396 = 0;
                
                chunk_sizze_8395 = res_7359;
                
                float res_8399;
                float acc_8402 = x_8397;
                int32_t groupstream_mapaccum_dummy_chunk_sizze_8400;
                
                groupstream_mapaccum_dummy_chunk_sizze_8400 = 1;
                if (chunk_sizze_8395 == res_7359) {
                    for (int32_t i_8401 = 0; i_8401 < res_7359; i_8401++) {
                        int32_t convop_x_8560;
                        float res_8406;
                        float y_8407;
                        float res_8408;
                        float res_8409;
                        float res_8410;
                        float res_8415;
                        float res_8416;
                        bool cond_8421;
                        bool res_8422;
                        bool x_8423;
                        bool cond_8424;
                        bool res_8425;
                        bool x_8426;
                        bool x_8427;
                        bool y_8428;
                        bool cond_8429;
                        bool cond_8430;
                        bool res_8431;
                        bool x_8432;
                        bool x_8433;
                        bool y_8434;
                        bool cond_8435;
                        bool cond_8436;
                        bool res_8437;
                        bool x_8438;
                        bool x_8439;
                        bool y_8440;
                        bool cond_8441;
                        float res_8442;
                        float x_8529;
                        float arg_8530;
                        float arg_8531;
                        int32_t res_8532;
                        int32_t res_8533;
                        float y_8534;
                        float res_8535;
                        float res_8537;
                        
                        convop_x_8560 = chunk_offset_8396 + i_8401;
                        res_8406 = sitofp_i32_f32(convop_x_8560);
                        y_8407 = res_7347 * res_8406;
                        res_8408 = res_8385 + y_8407;
                        if (res_8388) {
                            float x_8411;
                            float res_8412;
                            
                            x_8411 = res_8408 - y_8390;
                            res_8412 = x_8411 / res_8363;
                            res_8409 = res_8355;
                            res_8410 = res_8412;
                        } else {
                            float x_8413;
                            float res_8414;
                            
                            x_8413 = res_8408 - y_8391;
                            res_8414 = x_8413 / res_8364;
                            res_8409 = res_8414;
                            res_8410 = res_8356;
                        }
                        if (res_8388) {
                            float x_8417;
                            float res_8418;
                            
                            x_8417 = res_8408 - y_8392;
                            res_8418 = x_8417 / res_8363;
                            res_8415 = res_8359;
                            res_8416 = res_8418;
                        } else {
                            float x_8419;
                            float res_8420;
                            
                            x_8419 = res_8408 - y_8393;
                            res_8420 = x_8419 / res_8364;
                            res_8415 = res_8420;
                            res_8416 = res_8360;
                        }
                        cond_8421 = res_8409 < index_primexp_8592;
                        res_8422 = res_8415 < index_primexp_8592;
                        x_8423 = cond_8421 && res_8422;
                        cond_8424 = index_primexp_8574 < res_8409;
                        res_8425 = index_primexp_8574 < res_8415;
                        x_8426 = cond_8424 && res_8425;
                        x_8427 = !x_8423;
                        y_8428 = x_8426 && x_8427;
                        cond_8429 = x_8423 || y_8428;
                        cond_8430 = res_8410 < index_primexp_8587;
                        res_8431 = res_8416 < index_primexp_8587;
                        x_8432 = cond_8430 && res_8431;
                        x_8433 = !cond_8429;
                        y_8434 = x_8432 && x_8433;
                        cond_8435 = cond_8429 || y_8434;
                        cond_8436 = index_primexp_8568 < res_8410;
                        res_8437 = index_primexp_8568 < res_8416;
                        x_8438 = cond_8436 && res_8437;
                        x_8439 = !cond_8435;
                        y_8440 = x_8438 && x_8439;
                        cond_8441 = cond_8435 || y_8440;
                        if (cond_8441) {
                            res_8442 = 0.0F;
                        } else {
                            float res_8443;
                            
                            if (cond_8421) {
                                float y_8444;
                                float x_8445;
                                float res_8446;
                                float x_8447;
                                float x_8448;
                                float x_8449;
                                float y_8450;
                                float arg_8451;
                                float res_8452;
                                
                                y_8444 = res_8364 * index_primexp_8592;
                                x_8445 = res_8408 - y_8444;
                                res_8446 = x_8445 / res_8363;
                                x_8447 = res_8415 - index_primexp_8592;
                                x_8448 = fpow32(x_8447, 2.0F);
                                x_8449 = res_8416 - res_8446;
                                y_8450 = fpow32(x_8449, 2.0F);
                                arg_8451 = x_8448 + y_8450;
                                res_8452 = futrts_sqrt32(arg_8451);
                                res_8443 = res_8452;
                            } else {
                                float res_8453;
                                
                                if (res_8422) {
                                    float y_8454;
                                    float x_8455;
                                    float res_8456;
                                    float x_8457;
                                    float x_8458;
                                    float x_8459;
                                    float y_8460;
                                    float arg_8461;
                                    float res_8462;
                                    
                                    y_8454 = res_8364 * index_primexp_8592;
                                    x_8455 = res_8408 - y_8454;
                                    res_8456 = x_8455 / res_8363;
                                    x_8457 = res_8409 - index_primexp_8592;
                                    x_8458 = fpow32(x_8457, 2.0F);
                                    x_8459 = res_8410 - res_8456;
                                    y_8460 = fpow32(x_8459, 2.0F);
                                    arg_8461 = x_8458 + y_8460;
                                    res_8462 = futrts_sqrt32(arg_8461);
                                    res_8453 = res_8462;
                                } else {
                                    float res_8463;
                                    
                                    if (cond_8424) {
                                        float y_8464;
                                        float x_8465;
                                        float res_8466;
                                        float x_8467;
                                        float x_8468;
                                        float x_8469;
                                        float y_8470;
                                        float arg_8471;
                                        float res_8472;
                                        
                                        y_8464 = res_8364 * index_primexp_8574;
                                        x_8465 = res_8408 - y_8464;
                                        res_8466 = x_8465 / res_8363;
                                        x_8467 = res_8415 - index_primexp_8574;
                                        x_8468 = fpow32(x_8467, 2.0F);
                                        x_8469 = res_8416 - res_8466;
                                        y_8470 = fpow32(x_8469, 2.0F);
                                        arg_8471 = x_8468 + y_8470;
                                        res_8472 = futrts_sqrt32(arg_8471);
                                        res_8463 = res_8472;
                                    } else {
                                        float res_8473;
                                        
                                        if (res_8425) {
                                            float y_8474;
                                            float x_8475;
                                            float res_8476;
                                            float x_8477;
                                            float x_8478;
                                            float x_8479;
                                            float y_8480;
                                            float arg_8481;
                                            float res_8482;
                                            
                                            y_8474 = res_8364 *
                                                index_primexp_8574;
                                            x_8475 = res_8408 - y_8474;
                                            res_8476 = x_8475 / res_8363;
                                            x_8477 = res_8409 -
                                                index_primexp_8574;
                                            x_8478 = fpow32(x_8477, 2.0F);
                                            x_8479 = res_8410 - res_8476;
                                            y_8480 = fpow32(x_8479, 2.0F);
                                            arg_8481 = x_8478 + y_8480;
                                            res_8482 = futrts_sqrt32(arg_8481);
                                            res_8473 = res_8482;
                                        } else {
                                            float res_8483;
                                            
                                            if (cond_8430) {
                                                float y_8484;
                                                float x_8485;
                                                float res_8486;
                                                float x_8487;
                                                float x_8488;
                                                float x_8489;
                                                float y_8490;
                                                float arg_8491;
                                                float res_8492;
                                                
                                                y_8484 = res_8363 *
                                                    index_primexp_8587;
                                                x_8485 = res_8408 - y_8484;
                                                res_8486 = x_8485 / res_8364;
                                                x_8487 = res_8415 - res_8486;
                                                x_8488 = fpow32(x_8487, 2.0F);
                                                x_8489 = res_8416 -
                                                    index_primexp_8587;
                                                y_8490 = fpow32(x_8489, 2.0F);
                                                arg_8491 = x_8488 + y_8490;
                                                res_8492 =
                                                    futrts_sqrt32(arg_8491);
                                                res_8483 = res_8492;
                                            } else {
                                                float res_8493;
                                                
                                                if (res_8431) {
                                                    float y_8494;
                                                    float x_8495;
                                                    float res_8496;
                                                    float x_8497;
                                                    float x_8498;
                                                    float x_8499;
                                                    float y_8500;
                                                    float arg_8501;
                                                    float res_8502;
                                                    
                                                    y_8494 = res_8363 *
                                                        index_primexp_8587;
                                                    x_8495 = res_8408 - y_8494;
                                                    res_8496 = x_8495 /
                                                        res_8364;
                                                    x_8497 = res_8409 -
                                                        res_8496;
                                                    x_8498 = fpow32(x_8497,
                                                                    2.0F);
                                                    x_8499 = res_8410 -
                                                        index_primexp_8587;
                                                    y_8500 = fpow32(x_8499,
                                                                    2.0F);
                                                    arg_8501 = x_8498 + y_8500;
                                                    res_8502 =
                                                        futrts_sqrt32(arg_8501);
                                                    res_8493 = res_8502;
                                                } else {
                                                    float res_8503;
                                                    
                                                    if (cond_8436) {
                                                        float y_8504;
                                                        float x_8505;
                                                        float res_8506;
                                                        float x_8507;
                                                        float x_8508;
                                                        float x_8509;
                                                        float y_8510;
                                                        float arg_8511;
                                                        float res_8512;
                                                        
                                                        y_8504 = res_8363 *
                                                            index_primexp_8568;
                                                        x_8505 = res_8408 -
                                                            y_8504;
                                                        res_8506 = x_8505 /
                                                            res_8364;
                                                        x_8507 = res_8415 -
                                                            res_8506;
                                                        x_8508 = fpow32(x_8507,
                                                                        2.0F);
                                                        x_8509 = res_8416 -
                                                            index_primexp_8568;
                                                        y_8510 = fpow32(x_8509,
                                                                        2.0F);
                                                        arg_8511 = x_8508 +
                                                            y_8510;
                                                        res_8512 =
                                                            futrts_sqrt32(arg_8511);
                                                        res_8503 = res_8512;
                                                    } else {
                                                        float res_8513;
                                                        
                                                        if (res_8437) {
                                                            float y_8514;
                                                            float x_8515;
                                                            float res_8516;
                                                            float x_8517;
                                                            float x_8518;
                                                            float x_8519;
                                                            float y_8520;
                                                            float arg_8521;
                                                            float res_8522;
                                                            
                                                            y_8514 = res_8363 *
                                                                index_primexp_8568;
                                                            x_8515 = res_8408 -
                                                                y_8514;
                                                            res_8516 = x_8515 /
                                                                res_8364;
                                                            x_8517 = res_8409 -
                                                                res_8516;
                                                            x_8518 =
                                                                fpow32(x_8517,
                                                                       2.0F);
                                                            x_8519 = res_8410 -
                                                                index_primexp_8568;
                                                            y_8520 =
                                                                fpow32(x_8519,
                                                                       2.0F);
                                                            arg_8521 = x_8518 +
                                                                y_8520;
                                                            res_8522 =
                                                                futrts_sqrt32(arg_8521);
                                                            res_8513 = res_8522;
                                                        } else {
                                                            float x_8523;
                                                            float x_8524;
                                                            float x_8525;
                                                            float y_8526;
                                                            float arg_8527;
                                                            float res_8528;
                                                            
                                                            x_8523 = res_8415 -
                                                                res_8409;
                                                            x_8524 =
                                                                fpow32(x_8523,
                                                                       2.0F);
                                                            x_8525 = res_8416 -
                                                                res_8410;
                                                            y_8526 =
                                                                fpow32(x_8525,
                                                                       2.0F);
                                                            arg_8527 = x_8524 +
                                                                y_8526;
                                                            res_8528 =
                                                                futrts_sqrt32(arg_8527);
                                                            res_8513 = res_8528;
                                                        }
                                                        res_8503 = res_8513;
                                                    }
                                                    res_8493 = res_8503;
                                                }
                                                res_8483 = res_8493;
                                            }
                                            res_8473 = res_8483;
                                        }
                                        res_8463 = res_8473;
                                    }
                                    res_8453 = res_8463;
                                }
                                res_8443 = res_8453;
                            }
                            res_8442 = res_8443;
                        }
                        x_8529 = res_8408 - res_7345;
                        arg_8530 = x_8529 / res_7347;
                        arg_8531 = futrts_round32(arg_8530);
                        res_8532 = fptosi_f32_i32(arg_8531);
                        res_8533 = x_8389 + res_8532;
                        y_8534 = *(__global
                                   float *) &projections_mem_8598[res_8533 * 4];
                        res_8535 = res_8442 * y_8534;
                        res_8537 = acc_8402 + res_8535;
                        
                        float acc_tmp_8643 = res_8537;
                        
                        acc_8402 = acc_tmp_8643;
                    }
                } else {
                    for (int32_t i_8401 = 0; i_8401 < chunk_sizze_8395;
                         i_8401++) {
                        int32_t convop_x_8560;
                        float res_8406;
                        float y_8407;
                        float res_8408;
                        float res_8409;
                        float res_8410;
                        float res_8415;
                        float res_8416;
                        bool cond_8421;
                        bool res_8422;
                        bool x_8423;
                        bool cond_8424;
                        bool res_8425;
                        bool x_8426;
                        bool x_8427;
                        bool y_8428;
                        bool cond_8429;
                        bool cond_8430;
                        bool res_8431;
                        bool x_8432;
                        bool x_8433;
                        bool y_8434;
                        bool cond_8435;
                        bool cond_8436;
                        bool res_8437;
                        bool x_8438;
                        bool x_8439;
                        bool y_8440;
                        bool cond_8441;
                        float res_8442;
                        float x_8529;
                        float arg_8530;
                        float arg_8531;
                        int32_t res_8532;
                        int32_t res_8533;
                        float y_8534;
                        float res_8535;
                        float res_8537;
                        
                        convop_x_8560 = chunk_offset_8396 + i_8401;
                        res_8406 = sitofp_i32_f32(convop_x_8560);
                        y_8407 = res_7347 * res_8406;
                        res_8408 = res_8385 + y_8407;
                        if (res_8388) {
                            float x_8411;
                            float res_8412;
                            
                            x_8411 = res_8408 - y_8390;
                            res_8412 = x_8411 / res_8363;
                            res_8409 = res_8355;
                            res_8410 = res_8412;
                        } else {
                            float x_8413;
                            float res_8414;
                            
                            x_8413 = res_8408 - y_8391;
                            res_8414 = x_8413 / res_8364;
                            res_8409 = res_8414;
                            res_8410 = res_8356;
                        }
                        if (res_8388) {
                            float x_8417;
                            float res_8418;
                            
                            x_8417 = res_8408 - y_8392;
                            res_8418 = x_8417 / res_8363;
                            res_8415 = res_8359;
                            res_8416 = res_8418;
                        } else {
                            float x_8419;
                            float res_8420;
                            
                            x_8419 = res_8408 - y_8393;
                            res_8420 = x_8419 / res_8364;
                            res_8415 = res_8420;
                            res_8416 = res_8360;
                        }
                        cond_8421 = res_8409 < index_primexp_8592;
                        res_8422 = res_8415 < index_primexp_8592;
                        x_8423 = cond_8421 && res_8422;
                        cond_8424 = index_primexp_8574 < res_8409;
                        res_8425 = index_primexp_8574 < res_8415;
                        x_8426 = cond_8424 && res_8425;
                        x_8427 = !x_8423;
                        y_8428 = x_8426 && x_8427;
                        cond_8429 = x_8423 || y_8428;
                        cond_8430 = res_8410 < index_primexp_8587;
                        res_8431 = res_8416 < index_primexp_8587;
                        x_8432 = cond_8430 && res_8431;
                        x_8433 = !cond_8429;
                        y_8434 = x_8432 && x_8433;
                        cond_8435 = cond_8429 || y_8434;
                        cond_8436 = index_primexp_8568 < res_8410;
                        res_8437 = index_primexp_8568 < res_8416;
                        x_8438 = cond_8436 && res_8437;
                        x_8439 = !cond_8435;
                        y_8440 = x_8438 && x_8439;
                        cond_8441 = cond_8435 || y_8440;
                        if (cond_8441) {
                            res_8442 = 0.0F;
                        } else {
                            float res_8443;
                            
                            if (cond_8421) {
                                float y_8444;
                                float x_8445;
                                float res_8446;
                                float x_8447;
                                float x_8448;
                                float x_8449;
                                float y_8450;
                                float arg_8451;
                                float res_8452;
                                
                                y_8444 = res_8364 * index_primexp_8592;
                                x_8445 = res_8408 - y_8444;
                                res_8446 = x_8445 / res_8363;
                                x_8447 = res_8415 - index_primexp_8592;
                                x_8448 = fpow32(x_8447, 2.0F);
                                x_8449 = res_8416 - res_8446;
                                y_8450 = fpow32(x_8449, 2.0F);
                                arg_8451 = x_8448 + y_8450;
                                res_8452 = futrts_sqrt32(arg_8451);
                                res_8443 = res_8452;
                            } else {
                                float res_8453;
                                
                                if (res_8422) {
                                    float y_8454;
                                    float x_8455;
                                    float res_8456;
                                    float x_8457;
                                    float x_8458;
                                    float x_8459;
                                    float y_8460;
                                    float arg_8461;
                                    float res_8462;
                                    
                                    y_8454 = res_8364 * index_primexp_8592;
                                    x_8455 = res_8408 - y_8454;
                                    res_8456 = x_8455 / res_8363;
                                    x_8457 = res_8409 - index_primexp_8592;
                                    x_8458 = fpow32(x_8457, 2.0F);
                                    x_8459 = res_8410 - res_8456;
                                    y_8460 = fpow32(x_8459, 2.0F);
                                    arg_8461 = x_8458 + y_8460;
                                    res_8462 = futrts_sqrt32(arg_8461);
                                    res_8453 = res_8462;
                                } else {
                                    float res_8463;
                                    
                                    if (cond_8424) {
                                        float y_8464;
                                        float x_8465;
                                        float res_8466;
                                        float x_8467;
                                        float x_8468;
                                        float x_8469;
                                        float y_8470;
                                        float arg_8471;
                                        float res_8472;
                                        
                                        y_8464 = res_8364 * index_primexp_8574;
                                        x_8465 = res_8408 - y_8464;
                                        res_8466 = x_8465 / res_8363;
                                        x_8467 = res_8415 - index_primexp_8574;
                                        x_8468 = fpow32(x_8467, 2.0F);
                                        x_8469 = res_8416 - res_8466;
                                        y_8470 = fpow32(x_8469, 2.0F);
                                        arg_8471 = x_8468 + y_8470;
                                        res_8472 = futrts_sqrt32(arg_8471);
                                        res_8463 = res_8472;
                                    } else {
                                        float res_8473;
                                        
                                        if (res_8425) {
                                            float y_8474;
                                            float x_8475;
                                            float res_8476;
                                            float x_8477;
                                            float x_8478;
                                            float x_8479;
                                            float y_8480;
                                            float arg_8481;
                                            float res_8482;
                                            
                                            y_8474 = res_8364 *
                                                index_primexp_8574;
                                            x_8475 = res_8408 - y_8474;
                                            res_8476 = x_8475 / res_8363;
                                            x_8477 = res_8409 -
                                                index_primexp_8574;
                                            x_8478 = fpow32(x_8477, 2.0F);
                                            x_8479 = res_8410 - res_8476;
                                            y_8480 = fpow32(x_8479, 2.0F);
                                            arg_8481 = x_8478 + y_8480;
                                            res_8482 = futrts_sqrt32(arg_8481);
                                            res_8473 = res_8482;
                                        } else {
                                            float res_8483;
                                            
                                            if (cond_8430) {
                                                float y_8484;
                                                float x_8485;
                                                float res_8486;
                                                float x_8487;
                                                float x_8488;
                                                float x_8489;
                                                float y_8490;
                                                float arg_8491;
                                                float res_8492;
                                                
                                                y_8484 = res_8363 *
                                                    index_primexp_8587;
                                                x_8485 = res_8408 - y_8484;
                                                res_8486 = x_8485 / res_8364;
                                                x_8487 = res_8415 - res_8486;
                                                x_8488 = fpow32(x_8487, 2.0F);
                                                x_8489 = res_8416 -
                                                    index_primexp_8587;
                                                y_8490 = fpow32(x_8489, 2.0F);
                                                arg_8491 = x_8488 + y_8490;
                                                res_8492 =
                                                    futrts_sqrt32(arg_8491);
                                                res_8483 = res_8492;
                                            } else {
                                                float res_8493;
                                                
                                                if (res_8431) {
                                                    float y_8494;
                                                    float x_8495;
                                                    float res_8496;
                                                    float x_8497;
                                                    float x_8498;
                                                    float x_8499;
                                                    float y_8500;
                                                    float arg_8501;
                                                    float res_8502;
                                                    
                                                    y_8494 = res_8363 *
                                                        index_primexp_8587;
                                                    x_8495 = res_8408 - y_8494;
                                                    res_8496 = x_8495 /
                                                        res_8364;
                                                    x_8497 = res_8409 -
                                                        res_8496;
                                                    x_8498 = fpow32(x_8497,
                                                                    2.0F);
                                                    x_8499 = res_8410 -
                                                        index_primexp_8587;
                                                    y_8500 = fpow32(x_8499,
                                                                    2.0F);
                                                    arg_8501 = x_8498 + y_8500;
                                                    res_8502 =
                                                        futrts_sqrt32(arg_8501);
                                                    res_8493 = res_8502;
                                                } else {
                                                    float res_8503;
                                                    
                                                    if (cond_8436) {
                                                        float y_8504;
                                                        float x_8505;
                                                        float res_8506;
                                                        float x_8507;
                                                        float x_8508;
                                                        float x_8509;
                                                        float y_8510;
                                                        float arg_8511;
                                                        float res_8512;
                                                        
                                                        y_8504 = res_8363 *
                                                            index_primexp_8568;
                                                        x_8505 = res_8408 -
                                                            y_8504;
                                                        res_8506 = x_8505 /
                                                            res_8364;
                                                        x_8507 = res_8415 -
                                                            res_8506;
                                                        x_8508 = fpow32(x_8507,
                                                                        2.0F);
                                                        x_8509 = res_8416 -
                                                            index_primexp_8568;
                                                        y_8510 = fpow32(x_8509,
                                                                        2.0F);
                                                        arg_8511 = x_8508 +
                                                            y_8510;
                                                        res_8512 =
                                                            futrts_sqrt32(arg_8511);
                                                        res_8503 = res_8512;
                                                    } else {
                                                        float res_8513;
                                                        
                                                        if (res_8437) {
                                                            float y_8514;
                                                            float x_8515;
                                                            float res_8516;
                                                            float x_8517;
                                                            float x_8518;
                                                            float x_8519;
                                                            float y_8520;
                                                            float arg_8521;
                                                            float res_8522;
                                                            
                                                            y_8514 = res_8363 *
                                                                index_primexp_8568;
                                                            x_8515 = res_8408 -
                                                                y_8514;
                                                            res_8516 = x_8515 /
                                                                res_8364;
                                                            x_8517 = res_8409 -
                                                                res_8516;
                                                            x_8518 =
                                                                fpow32(x_8517,
                                                                       2.0F);
                                                            x_8519 = res_8410 -
                                                                index_primexp_8568;
                                                            y_8520 =
                                                                fpow32(x_8519,
                                                                       2.0F);
                                                            arg_8521 = x_8518 +
                                                                y_8520;
                                                            res_8522 =
                                                                futrts_sqrt32(arg_8521);
                                                            res_8513 = res_8522;
                                                        } else {
                                                            float x_8523;
                                                            float x_8524;
                                                            float x_8525;
                                                            float y_8526;
                                                            float arg_8527;
                                                            float res_8528;
                                                            
                                                            x_8523 = res_8415 -
                                                                res_8409;
                                                            x_8524 =
                                                                fpow32(x_8523,
                                                                       2.0F);
                                                            x_8525 = res_8416 -
                                                                res_8410;
                                                            y_8526 =
                                                                fpow32(x_8525,
                                                                       2.0F);
                                                            arg_8527 = x_8524 +
                                                                y_8526;
                                                            res_8528 =
                                                                futrts_sqrt32(arg_8527);
                                                            res_8513 = res_8528;
                                                        }
                                                        res_8503 = res_8513;
                                                    }
                                                    res_8493 = res_8503;
                                                }
                                                res_8483 = res_8493;
                                            }
                                            res_8473 = res_8483;
                                        }
                                        res_8463 = res_8473;
                                    }
                                    res_8453 = res_8463;
                                }
                                res_8443 = res_8453;
                            }
                            res_8442 = res_8443;
                        }
                        x_8529 = res_8408 - res_7345;
                        arg_8530 = x_8529 / res_7347;
                        arg_8531 = futrts_round32(arg_8530);
                        res_8532 = fptosi_f32_i32(arg_8531);
                        res_8533 = x_8389 + res_8532;
                        y_8534 = *(__global
                                   float *) &projections_mem_8598[res_8533 * 4];
                        res_8535 = res_8442 * y_8534;
                        res_8537 = acc_8402 + res_8535;
                        
                        float acc_tmp_8644 = res_8537;
                        
                        acc_8402 = acc_tmp_8644;
                    }
                }
                res_8399 = acc_8402;
                x_8397 = res_8399;
                res_8394 = x_8397;
                // save results to be reduced
                {
                    *(__local float *) &red_arr_mem_8639[local_tid_8285 * 4] =
                        res_8394;
                }
                // save map-out results
                { }
            } else {
                *(__local float *) &red_arr_mem_8639[local_tid_8285 * 4] = 0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // perform segmented scan to imitate reduction
        {
            float x_8352;
            float x_8353;
            int32_t index_i_8645;
            int32_t index_j_8646;
            int32_t index_i_8647;
            int32_t index_j_8648;
            float x_8649;
            float x_8650;
            
            index_i_8645 = local_tid_8285;
            
            int32_t skip_threads_8652;
            
            if (slt32(local_tid_8285, sizze_7338 * squot32(group_sizze_8336,
                                                           sizze_7338))) {
                x_8353 = *(volatile __local
                           float *) &red_arr_mem_8639[local_tid_8285 *
                                                      sizeof(float)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_8652 = 1;
                while (slt32(skip_threads_8652, 32)) {
                    if (sle32(skip_threads_8652, local_tid_8285 -
                              squot32(local_tid_8285, 32) * 32) &&
                        slt32(local_tid_8285, sizze_7338 *
                              squot32(group_sizze_8336, sizze_7338))) {
                        // read operands
                        {
                            x_8352 = *(volatile __local
                                       float *) &red_arr_mem_8639[(local_tid_8285 -
                                                                   skip_threads_8652) *
                                                                  sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_8285, sizze_7338),
                                       local_tid_8285 - (local_tid_8285 -
                                                         skip_threads_8652))) {
                                float res_8354 = x_8352 + x_8353;
                                
                                x_8353 = res_8354;
                            }
                        }
                    }
                    if (sle32(wave_sizze_8637, skip_threads_8652)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_8652, local_tid_8285 -
                              squot32(local_tid_8285, 32) * 32) &&
                        slt32(local_tid_8285, sizze_7338 *
                              squot32(group_sizze_8336, sizze_7338))) {
                        // write result
                        {
                            *(volatile __local
                              float *) &red_arr_mem_8639[local_tid_8285 *
                                                         sizeof(float)] =
                                x_8353;
                        }
                    }
                    if (sle32(wave_sizze_8637, skip_threads_8652)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_8652 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_8285 - squot32(local_tid_8285, 32) * 32) == 31 &&
                    slt32(local_tid_8285, sizze_7338 * squot32(group_sizze_8336,
                                                               sizze_7338))) {
                    *(volatile __local
                      float *) &red_arr_mem_8639[squot32(local_tid_8285, 32) *
                                                 sizeof(float)] = x_8353;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
            {
                int32_t skip_threads_8653;
                
                if (squot32(local_tid_8285, 32) == 0 && slt32(local_tid_8285,
                                                              sizze_7338 *
                                                              squot32(group_sizze_8336,
                                                                      sizze_7338))) {
                    x_8650 = *(volatile __local
                               float *) &red_arr_mem_8639[local_tid_8285 *
                                                          sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_8653 = 1;
                    while (slt32(skip_threads_8653, 32)) {
                        if (sle32(skip_threads_8653, local_tid_8285 -
                                  squot32(local_tid_8285, 32) * 32) &&
                            (squot32(local_tid_8285, 32) == 0 &&
                             slt32(local_tid_8285, sizze_7338 *
                                   squot32(group_sizze_8336, sizze_7338)))) {
                            // read operands
                            {
                                x_8649 = *(volatile __local
                                           float *) &red_arr_mem_8639[(local_tid_8285 -
                                                                       skip_threads_8653) *
                                                                      sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_8285 * 32 + 32 - 1,
                                                  sizze_7338), local_tid_8285 *
                                           32 + 32 - 1 - ((local_tid_8285 -
                                                           skip_threads_8653) *
                                                          32 + 32 - 1))) {
                                    float res_8651 = x_8649 + x_8650;
                                    
                                    x_8650 = res_8651;
                                }
                            }
                        }
                        if (sle32(wave_sizze_8637, skip_threads_8653)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_8653, local_tid_8285 -
                                  squot32(local_tid_8285, 32) * 32) &&
                            (squot32(local_tid_8285, 32) == 0 &&
                             slt32(local_tid_8285, sizze_7338 *
                                   squot32(group_sizze_8336, sizze_7338)))) {
                            // write result
                            {
                                *(volatile __local
                                  float *) &red_arr_mem_8639[local_tid_8285 *
                                                             sizeof(float)] =
                                    x_8650;
                            }
                        }
                        if (sle32(wave_sizze_8637, skip_threads_8653)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_8653 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_8285, 32) == 0 || !slt32(local_tid_8285,
                                                                 sizze_7338 *
                                                                 squot32(group_sizze_8336,
                                                                         sizze_7338)))) {
                    // read operands
                    {
                        x_8352 = *(volatile __local
                                   float *) &red_arr_mem_8639[(squot32(local_tid_8285,
                                                                       32) -
                                                               1) *
                                                              sizeof(float)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32(local_tid_8285, sizze_7338),
                                   local_tid_8285 - (squot32(local_tid_8285,
                                                             32) * 32 - 1))) {
                            float res_8354 = x_8352 + x_8353;
                            
                            x_8353 = res_8354;
                        }
                    }
                    // write final result
                    {
                        *(volatile __local
                          float *) &red_arr_mem_8639[local_tid_8285 *
                                                     sizeof(float)] = x_8353;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_8285, 32) == 0) {
                    *(volatile __local
                      float *) &red_arr_mem_8639[local_tid_8285 *
                                                 sizeof(float)] = x_8353;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_8286 + i_8642 * num_groups_8346) *
                      squot32(group_sizze_8336, sizze_7338) + local_tid_8285,
                      arg_7360) && slt32(local_tid_8285,
                                         squot32(group_sizze_8336,
                                                 sizze_7338))) {
                *(__global float *) &mem_8610[((group_id_8286 + i_8642 *
                                                num_groups_8346) *
                                               squot32(group_sizze_8336,
                                                       sizze_7338) +
                                               local_tid_8285) * 4] = *(__local
                                                                        float *) &red_arr_mem_8639[((local_tid_8285 +
                                                                                                     1) *
                                                                                                    sizze_7338 -
                                                                                                    1) *
                                                                                                   4];
            }
        }
    }
}
