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

benchfp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/forwardprojection.fut
	futhark bench --runs=10 --skip-compilation ./futhark/forwardprojection.fut > ./output/benchmarks/fp
	cat output/benchmarks/fp

benchbp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection.fut

benchbp_test:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection_test.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection_test.fut

bench_cur_best:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection_cur_best.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection_cur_best.fut

lbench_cur_best:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/backprojection_cur_best.fut
	futhark-bench --runs=10 --skip-compilation ./futhark/backprojection_cur_best.fut

lbenchbp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/backprojection.fut
	futhark-bench --runs=10 --skip-compilation ./futhark/backprojection.fut

lbenchbp_test:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/backprojection_test.fut
	futhark-bench --runs=10 --skip-compilation ./futhark/backprojection_test.fut

lcompare: lbench_cur_best lbenchbp_test
compare: bench_cur_best benchbp_test

benchsirt:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT.fut > ./output/benchmarks/sirt

benchsirt3d:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT3D.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT3D.fut > ./output/benchmarks/sirt3d

bpbetter:
	cp output/benchmarks/bp output/benchmarks/bestbp
