echo "generating data"
cd ..
python backprojectiondata.py
echo "compiling"
cd futhark

futhark-opencl backprojection_jh.fut
futhark-opencl backprojection_doubleparallel.fut
futhark-opencl backprojection_map.fut
echo "running benchmarks"

./projectionmatrix_jh -D < ../data/bpinputf32rad256_32 1> /dev/null 2> ../output/backprojection_jh
./projectionmatrix_doubleparallel -D < ../data/bpinputf32rad256_32 1> /dev/null 2> ../output/backprojection_doubleparallel
./projectionmatrix_map -D < ../data/bpinputf32rad256_32 1> /dev/null 2> ../output/backprojection_map

futhark-bench --runs=10 --skip-compilation backprojection_jh.fut > ../output/backprojection_jh_benchmark
futhark-bench --runs=10 --skip-compilation backprojection_doubleparallel.fut > ../output/backprojection_doubleparallel_benchmark
futhark-bench --runs=10 --skip-compilation backprojection_map.fut > ../output/backprojection_map_benchmark
