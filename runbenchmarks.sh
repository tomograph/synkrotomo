echo "compiling"
cd futhark
# futhark-pyopencl --library backprojection.fut
futhark-pyopencl --library forwardprojection.fut
echo "running benchmarks"
cd ..
python benchmark.py
