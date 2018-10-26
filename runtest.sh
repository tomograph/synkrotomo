#!/bin/bash
echo "compiling"
cd futhark
# futhark-pyopencl --library intersections.fut
futhark-opencl ba.fut
echo "running"
cd ..
