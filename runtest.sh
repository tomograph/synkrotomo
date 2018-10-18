#!/bin/bash
echo "compiling"
cd futhark
# futhark-pyopencl --library intersections.fut
futhark-test --compiler=futhark-opencl test.fut
cd ..
# echo "running"
# python system_matrix_test.py
