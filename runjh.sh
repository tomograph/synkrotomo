echo "compiling"
cd sysmatjh
futhark-pyopencl --library intersections.fut
futhark-pyopencl --library intersections_jhversion.fut
cd ..
python system_matrix_test.py
