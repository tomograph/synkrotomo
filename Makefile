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
	futhark opencl ./futhark/forwardprojection.fut
	-futhark bench --runs=10 --skip-compilation ./futhark/forwardprojection.fut > ./output/benchmarks/fp

benchbp:
	futhark opencl ./futhark/backprojection.fut
	-futhark bench --runs=10 --skip-compilation ./futhark/backprojection.fut > ./output/benchmarks/bp

benchbp_test:
	futhark opencl ./futhark/backprojection_test.fut
	-futhark bench --runs=10 --skip-compilation ./futhark/backprojection_test.fut > ./output/benchmarks/bp_test

benchbp_expand:
	futhark opencl ./futhark/backprojection_test.fut
	-futhark bench --runs=10 --skip-compilation ./futhark/backprojection_test.fut > ./output/benchmarks/bp_expand

bench_cur_best:
	futhark opencl ./futhark/backprojection_cur_best.fut
	-futhark bench --runs=10 --skip-compilation ./futhark/backprojection_cur_best.fut > ./output/benchmarks/bp_cur_best

smallbench: benchbp benchbp_test benchbp_expand bench_cur_best

benchsirt:
	futhark opencl ./futhark/SIRT.fut
	-futhark bench --runs=1 --skip-compilation ./futhark/SIRT.fut > ./output/benchmarks/sirt

benchsirt3d:
	futhark opencl ./futhark/SIRT3D.fut
	-futhark bench --runs=1 --skip-compilation ./futhark/SIRT3D.fut > ./output/benchmarks/sirt3d

benchsirtcb:
	futhark opencl ./futhark/SIRTcb.fut
	-futhark bench --runs=1 --skip-compilation ./futhark/SIRTcb.fut > ./output/benchmarks/sirtcb

benchsirt3dcb:
	futhark opencl ./futhark/SIRT3Dcb.fut
	-futhark bench --runs=1 --skip-compilation ./futhark/SIRT3Dcb.fut > ./output/benchmarks/sirt3dcb

benchsirt_expand:
	futhark opencl ./futhark/SIRT_expand.fut
	-futhark bench --runs=1 --skip-compilation ./futhark/SIRT_expand.fut > ./output/benchmarks/sirt_expand

benchsirt3d_expand:
	futhark opencl ./futhark/SIRT3D_expand.fut
	-futhark bench --runs=1 --skip-compilation ./futhark/SIRT3D_expand.fut > ./output/benchmarks/sirt3d_expand

benchall: benchfp benchbp benchbp_test bench_cur_best benchsirt benchsirt3d benchsirtcb benchsirt3dcb benchbp_expand benchsirt_expand benchsirt3d_expand

clean:
	rm -f ./futhark/*.c ./futhark/backprojection ./futhark/backprojection_test ./futhark/backprojection_cur_best ./futhark/backprojection_expand ./futhark/forwardprojection ./futhark/SIRT ./futhark/SIRTcb ./futhark/SIRT_expand ./futhark/SIRT3D ./futhark/SIRT3Dcb ./futhark/SIRT3D_expand
