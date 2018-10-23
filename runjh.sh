echo "compiling"
cd sysmatjh
futhark-pyopencl --library intersections.fut
futhark-pyopencl --library entrypoints.fut
cd ..
python system_matrix_test.py
