#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "intersections.h"

int read_data(int i) {
    char filename[sizeof "./data/data_result001.bin"];
    sprintf(filename, "./data/data_result%03d.bin", i+1);
    FILE* f = fopen(filename, "rb");
    
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = malloc(fsize);
    fread(data, fsize, 1, f);

    fclose(f);
    free(data);

    return 0;
}

int main (int argc, char* argv[]) {
    printf("Do you even bench?\n");
    FILE *f = fopen("results.txt", "w");
    
    int i_grid_size       = 2000;
    int grid_size_delta   = 0;
    int i_line_count      = 1000;
    int line_count_delta  = 0;
    int i_num_batches     = 0;
    int num_batches_delta = 0;

    float scan_start      = 15;
    float scan_end        = 25;
    float i_scan_step     = 1;
    float scan_step_delta = 0;

    int num_of_runs         = 10;    // Number of increases on input values
    int instance_multiplier = 10;    // Number of instances used for timing average

    long times[num_of_runs];   
    for ( int i = 0; i < num_of_runs; i++ ) {
        long time_accumulator = 0;
        for ( int j = 0; j < instance_multiplier; j++ ) {
            long start, end;
            struct timeval timecheck;

            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

            read_data(i);

            gettimeofday(&timecheck, NULL);
            end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
            time_accumulator += end - start;
        }

        times[i] = time_accumulator / instance_multiplier;
        fprintf(f, "%i,%ld\n", i+1, times[i]);  
    }
    fclose(f);      
}