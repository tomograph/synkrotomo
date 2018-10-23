echo "compiling"
cd sysmatjh
futhark-pyopencl --library intersections.fut
cd ..
python system_matrix_test.py
