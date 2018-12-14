lib:
	futhark-pyopencl --library ./futhark/SIRT.fut
	futhark-pyopencl --library ./futhark/backprojection.fut
	futhark-pyopencl --library ./futhark/forwardprojection.fut

opencl:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/SIRT.fut

runopencl: opencl
	./SIRT

runpytest: lib
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
	futhark-bench --runs=1 --skip-compilation ./futhark/SIRT.fut > ./output/sirt_benchmark
