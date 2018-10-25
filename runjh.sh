echo "compiling"
cd sysmatjh
futhark-pyopencl --library intersections.fut
cd ..
cd sysmatas
futhark-pyopencl --library algorithm.fut
cd ..
python system_matrix_test.py
