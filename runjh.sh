echo "compiling"
cd sysmatjh
futhark-pyopencl --library intersections.fut
cd ..
echo "running"
python system_matrix_test.py
