echo "benchmarkbackprojection"
echo "generating data"
cd ..
python backprojectiondata.py
echo "compiling"
cd futhark

futhark-opencl backprojection_jh.fut
futhark-opencl backprojection_doubleparallel.fut
futhark-opencl backprojection_map.fut
futhark-opencl backprojection_semiflat.fut
futhark-opencl backprojection_bpintegrated.fut
echo "running benchmarks memory"

./backprojection_jh -D < ../data/bpinputf32rad256_64 1> /dev/null 2> ../output/backprojection_jh
./backprojection_doubleparallel -D < ../data/bpinputf32rad256_64 1> /dev/null 2> ../output/backprojection_doubleparallel
./backprojection_map -D < ../data/bpinputf32rad256_64 1> /dev/null 2> ../output/backprojection_map
./backprojection_semiflat -D < ../data/bpinputf32rad256_64 1> /dev/null 2> ../output/backprojection_semiflat

echo "running benchmarks"
futhark-bench --runs=1 --compiler=futhark-opencl --pass-option=-d#0 backprojection_jh.fut > ../output/backprojection_jh_benchmark_par
futhark-bench --runs=1 --compiler=futhark-opencl --pass-option=-d#1 backprojection_doubleparallel.fut > ../output/backprojection_doubleparallel_benchmark_par
futhark-bench --runs=1 --compiler=futhark-opencl --pass-option=-d#1 backprojection_map.fut > ../output/backprojection_map_benchmark_par
futhark-bench --runs=1 --compiler=futhark-opencl --pass-option=-d#1 backprojection_semiflat.fut > ../output/backprojection_semiflat_benchmark_par
futhark-bench --runs=1 --compiler=futhark-opencl --pass-option=-d#1 backprojection_bpintegrated.fut > ../output/backprojection_bpintegrated_benchmark_par
