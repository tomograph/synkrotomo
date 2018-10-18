#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "intersections.h"

int main () {
  int grid_size    = 2000;
  int line_count_g = 100;
  int delta        = 1;

  float scan_start = 15;
  float scan_end   = 25;
  float scan_step  = 1;
  
  printf("About to write all the floats\n");
  for(int i=0; i < 10; i++) {
    int line_count = line_count_g * i;

    char filename[sizeof "./data/data_result001.bin"];
    sprintf(filename, "./data/data_result%03d.bin", i+1);
    FILE *f = fopen(filename, "wb");
    
    struct futhark_context_config *cfg = futhark_context_config_new();
    struct futhark_context *ctx = futhark_context_new(cfg);

    int angles         = (int)((scan_end - scan_start) / scan_step);
    int arr_size       = (2 * grid_size - 1) * angles * (2 * line_count + 1);
    float *lengths_arr = malloc(arr_size * sizeof(float));
    int *indicies_arr  = malloc(arr_size * sizeof(float));  

    struct futhark_f32_3d* lengths;
    struct futhark_i32_3d* indicies;
    futhark_main(ctx, &lengths, &indicies, grid_size, delta, line_count, scan_start, scan_end, scan_step);

    futhark_context_sync(ctx);
    futhark_values_f32_3d(ctx, lengths, lengths_arr);
    futhark_values_i32_3d(ctx, indicies, indicies_arr);  

    fwrite(lengths_arr, sizeof(lengths_arr[0]), arr_size, f);
    fwrite(indicies_arr, sizeof(indicies_arr[0]), arr_size, f);  

    futhark_free_f32_3d(ctx, lengths);
    futhark_free_i32_3d(ctx, indicies);
    free(lengths_arr);
    free(indicies_arr);

    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
  }
  printf("Phew, we made it captain!\n");

  return 0;    
}