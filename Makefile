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

databp:
	python testsirtdata.py

bench:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/backprojection.fut
	futhark-bench --runs=10 --skip-compilation ./futhark/backprojection.fut
