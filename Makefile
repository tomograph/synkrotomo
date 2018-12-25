lib:
	futhark-pyopencl --library ./futhark/SIRT.fut
	futhark-pyopencl --library ./futhark/backprojection.fut
	futhark-pyopencl --library ./futhark/forwardprojection.fut

libc:
	futhark-c --library ./futhark/SIRT.fut
	futhark-c --library ./futhark/backprojection.fut
	futhark-c --library ./futhark/forwardprojection.fut

opencl:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/SIRT.fut

runopencl: opencl
	./SIRT

runpytest: lib
	python testsirt.py

runpytest-c: lib
	python testsirt.py

data:
	python sirtdata.py

benchfp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/forwardprojection.fut
	futhark-bench --runs=10 --skip-compilation ./futhark/forwardprojection.fut > ./output/fp_benchmark

benchbp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/backprojection.fut
	futhark-bench --runs=10 --skip-compilation ./futhark/backprojection.fut > ./output/bp_benchmark

benchsirt:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/SIRT.fut
	futhark-bench --runs=1 --skip-compilation ./futhark/SIRT.fut > ./output/sirt_benchmark &

benchall:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/SIRT.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/forwardprojection.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/backprojection.fut
	futhark-bench --runs=1 --skip-compilation ./futhark/forwardprojection.fut > ./output/forwardprojection_benchmark
	futhark-bench --runs=1 --skip-compilation ./futhark/backprojection.fut > ./output/backprojection_benchmark
	futhark-bench --runs=1 --skip-compilation ./futhark/SIRT.fut > ./output/sirt_benchmark
