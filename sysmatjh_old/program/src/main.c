#include <stdio.h>
#include <stdlib.h>
#include "intersections.h"

int main(int argc, char* argv[]) {
    
    float delta;
    int grid_size, line_count, num_batches;
    float scan_start, scan_end, scan_step;

    if (argc < 7) {
        grid_size   = 2000;
        delta       = 1.0;
        line_count  = 100;
        scan_start  = 0;
        scan_end    = 180;
        scan_step   = 1.0;
        num_batches = 2;
    } else {
        grid_size   = atoi(argv[1]);
        line_count  = atoi(argv[2]);
        scan_start  = atof(argv[3]);
        scan_end    = atof(argv[4]);
        scan_step   = atof(argv[5]);
        num_batches = atof(argv[6]);        
    }

    int batch_offset = (int)((scan_end - scan_start) / scan_step) / num_batches;

    struct futhark_context_config *cfg = futhark_context_config_new();
    struct futhark_context *ctx = futhark_context_new(cfg);
    for (int i = 0; i < num_batches; i++) {
        float scan_start_local = scan_start + batch_offset * i;
        float scan_end_local   = scan_start_local + batch_offset;
        int angles             = (int)((scan_end_local - scan_start_local) / scan_step);
        int arr_size           = (2 * grid_size - 1) * sizeof(float) * angles * (2 * line_count + 1);
        float *values          = malloc(arr_size);
        
        struct futhark_f32_3d* lengths;
        struct futhark_i32_3d* indicies;
        futhark_main(ctx, &lengths, &indicies, grid_size, delta, line_count, scan_start_local, scan_end_local, scan_step);

        futhark_context_sync(ctx);
        futhark_values_f32_3d(ctx, lengths, values);
        // printf("Result: %f\n", values[0]);

        futhark_free_f32_3d(ctx, lengths);
        futhark_free_i32_3d(ctx, indicies);
        free(values);
    }

    futhark_context_free(ctx);
    futhark_context_config_free(cfg);

    return 0;    
}