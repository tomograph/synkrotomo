lib:
	futhark pyopencl --library ./futhark/SIRT.fut
	futhark pyopencl --library ./futhark/SIRT3D.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark pyopencl --library ./futhark/backprojection.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark pyopencl --library ./futhark/backprojection_test.fut
	futhark pyopencl --library ./futhark/forwardprojection.fut

libc:
	futhark c --library ./futhark/SIRT.fut
	futhark c --library ./futhark/SIRT3D.fut
	futhark c --library ./futhark/backprojection.fut
	futhark c --library ./futhark/forwardprojection.fut

benchfp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/forwardprojection.fut
	futhark bench --runs=10 --skip-compilation ./futhark/forwardprojection.fut > ./output/benchmarks/fp

benchbp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection.fut

benchbptest:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection_test.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection_test.fut

benchbptest2:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection_test_2.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection_test_2.fut

benchsirt:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT.fut

benchsirttest:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT_test.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT_test.fut

benchsirt3d:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT3D.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT3D.fut
