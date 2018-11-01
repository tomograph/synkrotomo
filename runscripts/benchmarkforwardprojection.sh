echo "generating data"
cd ..
python forwardprojectiondata.py
echo "compiling"
cd futhark

futhark-opencl forwardprojection_jh.fut
futhark-opencl forwardprojection_doubleparallel.fut
futhark-opencl forwardprojection_map.fut
echo "running benchmarks"

./projectionmatrix_jh -D < ../data/fpinputf32rad256 1> /dev/null 2> ../../output/matrix
./projectionmatrix_doubleparallel -D < ../data/fpinputf32rad256 1> /dev/null 2>>../../output/matrix
./projectionmatrix_map -D < ../data/fpinputf32rad256 1> /dev/null 2>> ../../output/matrix

futhark-bench --runs=10 --skip-compilation forwardprojection_jh.fut
futhark-bench --runs=10 --skip-compilation forwardprojection_doubleparallel.fut
futhark-bench --runs=10 --skip-compilation forwardprojection_map.fut
