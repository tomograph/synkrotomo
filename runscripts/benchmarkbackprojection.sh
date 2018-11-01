numruns=1
echo "benchmarkbackprojection"
echo "generating data"
cd ..
python backprojectiondata.py
echo "compiling"
cd futhark

futhark-opencl backprojection_jh.fut
futhark-opencl backprojection_doubleparallel.fut
futhark-opencl backprojection_map.fut
echo "running benchmarks memory"

./projectionmatrix_jh -D < ../data/bpinputf32rad256_32 1> /dev/null 2> ../output/backprojection_jh
./projectionmatrix_doubleparallel -D < ../data/bpinputf32rad256_32 1> /dev/null 2> ../output/backprojection_doubleparallel
./projectionmatrix_map -D < ../data/bpinputf32rad256_32 1> /dev/null 2> ../output/backprojection_map

echo "running benchmarks"
futhark-bench --runs=$numruns --skip-compilation backprojection_jh.fut > ../output/backprojection_jh_benchmark
futhark-bench --runs=$numruns --skip-compilation backprojection_doubleparallel.fut > ../output/backprojection_doubleparallel_benchmark
futhark-bench --runs=$numruns --skip-compilation backprojection_map.fut > ../output/backprojection_map_benchmark
