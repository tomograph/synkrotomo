echo "generating data"
cd ..
python forwardprojectiondata.py
echo "compiling"
cd futhark

futhark-opencl forwardprojection_jh.fut
futhark-opencl forwardprojection_doubleparallel.fut
futhark-opencl forwardprojection_map.fut
echo "running benchmarks"

./projectionmatrix_jh -D < ../data/fpinputf32rad256_32 1> /dev/null 2> ../output/forwardprojection_jh
./projectionmatrix_doubleparallel -D < ../data/fpinputf32rad256_32 1> /dev/null 2>../output/forwardprojection_doubleparallel
./projectionmatrix_map -D < ../data/fpinputf32rad256_32 1> /dev/null 2> ../output/forwardprojection_map

futhark-bench --runs=10 --skip-compilation forwardprojection_jh.fut > ../output/forwardprojection_jh_benchmark
futhark-bench --runs=10 --skip-compilation forwardprojection_doubleparallel.fut  > ../output/forwardprojection_doubleparallel_benchmark
futhark-bench --runs=10 --skip-compilation forwardprojection_map.fut > ../output/forwardprojection_map_benchmark
