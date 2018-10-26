echo "compiling"
cd futhark
futhark-pyopencl --library backprojection.fut
futhark-pyopencl --library forwardprojection.fut
cd ..
python projection_test2d.py
