echo "benchmarkprojectionmatrix"
echo "generating data"
cd ..
python projectionmatrixdata.py
echo "compiling"
cd futhark

futhark-opencl projectionmatrix_jh.fut
futhark-opencl projectionmatrix_doubleparallel.fut
futhark-opencl projectionmatrix_map.fut
echo "running benchmarks memory"

./projectionmatrix_jh -D < ../data/matrixinputf32rad256 1> /dev/null 2> ../output/projectionmatrix_jh
./projectionmatrix_doubleparallel -D < ../data/matrixinputf32rad256 1> /dev/null 2>../output/projectionmatrix_doubleparallel
./projectionmatrix_map -D < ../data/matrixinputf32rad256 1> /dev/null 2> ../output/projectionmatrix_map
echo "running benchmarks"

futhark-bench --runs=1 --skip-compilation projectionmatrix_jh.fut > ../output/projectionmatrix_jh_benchmark
futhark-bench --runs=1 --skip-compilation projectionmatrix_doubleparallel.fut > ../output/projectionmatrix_doubleparallel_benchmark
futhark-bench --runs=1 --skip-compilation projectionmatrix_map.fut > ../output/projectionmatrix_map_benchmark
