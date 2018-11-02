echo "benchmarkforwardprojection"
echo "generating data"
cd ..
# python forwardprojectiondata.py
echo "compiling"
cd futhark

futhark-opencl forwardprojection_jh.fut
futhark-opencl forwardprojection_doubleparallel.fut
futhark-opencl forwardprojection_map.fut
futhark-opencl forwardprojection_semiflat.fut
futhark-opencl forwardprojection_dpintegrated.fut
echo "running benchmarks memory"

./forwardprojection_jh -D < ../data/fpinputf32rad256_64 1> /dev/null 2> ../output/forwardprojection_jh
./forwardprojection_doubleparallel -D < ../data/fpinputf32rad256_64 1> /dev/null 2>../output/forwardprojection_doubleparallel
./forwardprojection_map -D < ../data/fpinputf32rad256_64 1> /dev/null 2> ../output/forwardprojection_map
./forwardprojection_semiflat -D < ../data/fpinputf32rad256_64 1> /dev/null 2> ../output/forwardprojection_semiflat
./forwardprojection_dpintegrated -D < ../data/fpinputf32rad256_64 1> /dev/null 2> ../output/forwardprojection_dpintegrated
echo "running benchmarks"

futhark-bench --runs=10 --skip-compilation forwardprojection_jh.fut > ../output/forwardprojection_jh_benchmark
futhark-bench --runs=10 --skip-compilation forwardprojection_doubleparallel.fut  > ../output/forwardprojection_doubleparallel_benchmark
futhark-bench --runs=10 --skip-compilation forwardprojection_map.fut > ../output/forwardprojection_map_benchmark
futhark-bench --runs=10 --skip-compilation forwardprojection_semiflat.fut > ../output/forwardprojection_semiflat_benchmark
futhark-bench --runs=10 --skip-compilation forwardprojection_dpintegrated.fut > ../output/forwardprojection_dpintegrated_benchmark
