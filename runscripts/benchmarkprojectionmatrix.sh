echo "generating data"
cd ..
python projectionmatrixdata.py
echo "compiling"
cd futhark

futhark-opencl projectionmatrix_jh.fut
futhark-opencl projectionmatrix_doubleparallel.fut
futhark-opencl projectionmatrix_map.fut
echo "running benchmarks"

./projectionmatrix_jh -D < ../data/matrixinputf32rad256 1> /dev/null 2> ../../output/matrix
./projectionmatrix_doubleparallel -D < ../data/matrixinputf32rad256 1> /dev/null 2>>../../output/matrix
./projectionmatrix_map -D < ../data/matrixinputf32rad256 1> /dev/null 2>> ../../output/matrix

futhark-bench --runs=10 --skip-compilation projectionmatrix_jh.fut
futhark-bench --runs=10 --skip-compilation projectionmatrix_doubleparallel.fut
futhark-bench --runs=10 --skip-compilation projectionmatrix_map.fut
