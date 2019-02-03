lib:
	futhark pyopencl --library ./futhark/SIRT.fut
	futhark pyopencl --library ./futhark/SIRT3D.fut
	futhark pyopencl --library ./futhark/backprojection.fut
	futhark pyopencl --library ./futhark/forwardprojection.fut

libc:
	futhark c --library ./futhark/SIRT.fut
	futhark c --library ./futhark/SIRT3D.fut
	futhark c --library ./futhark/backprojection.fut
	futhark c --library ./futhark/forwardprojection.fut

benchall:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/forwardprojection.fut
	futhark bench --runs=1 --skip-compilation ./futhark/forwardprojection.fut > ./output/benchmarks/fp
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection.fut
	futhark bench --runs=1 --skip-compilation ./futhark/backprojection.fut > ./output/benchmarks/bp
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT.fut > ./output/benchmarks/sirt
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT3D.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT3D.fut > ./output/benchmarks/sirt3d
