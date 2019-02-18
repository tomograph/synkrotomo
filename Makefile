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

sanityData:
	-mkdir sd
	futhark opencl ./futhark/forwardprojection.fut
	futhark opencl ./futhark/backprojection.fut
	futhark opencl ./futhark/SIRT.fut
	futhark opencl ./futhark/SIRT3D.fut
	-./futhark/forwardprojection < ./data/fpinputf32rad64    > ./sd/fpsanity64
	-./futhark/forwardprojection < ./data/fpinputf32rad128   > ./sd/fpsanity128
	-./futhark/forwardprojection < ./data/fpinputf32rad256   > ./sd/fpsanity256
	-./futhark/forwardprojection < ./data/fpinputf32rad512   > ./sd/fpsanity512
	-./futhark/forwardprojection < ./data/fpinputf32rad1024  > ./sd/fpsanity1024
	-./futhark/forwardprojection < ./data/fpinputf32rad1500  > ./sd/fpsanity1500
	-./futhark/forwardprojection < ./data/fpinputf32rad2000  > ./sd/fpsanity2000
	-./futhark/forwardprojection < ./data/fpinputf32rad2048  > ./sd/fpsanity2048
	-./futhark/forwardprojection < ./data/fpinputf32rad2500  > ./sd/fpsanity2500
	-./futhark/forwardprojection < ./data/fpinputf32rad3000  > ./sd/fpsanity3000
	-./futhark/forwardprojection < ./data/fpinputf32rad3500  > ./sd/fpsanity3500
	-./futhark/forwardprojection < ./data/fpinputf32rad4000  > ./sd/fpsanity4000
	-./futhark/forwardprojection < ./data/fpinputf32rad4096  > ./sd/fpsanity4096
	-./futhark/backprojection < ./data/bpinputf32rad64    > ./sd/bpsanity64
	-./futhark/backprojection < ./data/bpinputf32rad128   > ./sd/bpsanity128
	-./futhark/backprojection < ./data/bpinputf32rad256   > ./sd/bpsanity256
	-./futhark/backprojection < ./data/bpinputf32rad512   > ./sd/bpsanity512
	-./futhark/backprojection < ./data/bpinputf32rad1024  > ./sd/bpsanity1024
	-./futhark/backprojection < ./data/bpinputf32rad1500  > ./sd/bpsanity1500
	-./futhark/backprojection < ./data/bpinputf32rad2000  > ./sd/bpsanity2000
	-./futhark/backprojection < ./data/bpinputf32rad2048  > ./sd/bpsanity2048
	-./futhark/backprojection < ./data/bpinputf32rad2500  > ./sd/bpsanity2500
	-./futhark/backprojection < ./data/bpinputf32rad3000  > ./sd/bpsanity3000
	-./futhark/backprojection < ./data/bpinputf32rad3500  > ./sd/bpsanity3500
	-./futhark/backprojection < ./data/bpinputf32rad4000  > ./sd/bpsanity4000
	-./futhark/backprojection < ./data/bpinputf32rad4096  > ./sd/bpsanity4096
	-./futhark/SIRT < ./data/sirtinputf32rad64    > ./sd/sirtsanity64
	-./futhark/SIRT < ./data/sirtinputf32rad128   > ./sd/sirtsanity128
	-./futhark/SIRT < ./data/sirtinputf32rad256   > ./sd/sirtsanity256
	-./futhark/SIRT < ./data/sirtinputf32rad512   > ./sd/sirtsanity512
	-./futhark/SIRT < ./data/sirtinputf32rad1024  > ./sd/sirtsanity1024
	-./futhark/SIRT < ./data/sirtinputf32rad1500  > ./sd/sirtsanity1500
	-./futhark/SIRT < ./data/sirtinputf32rad2000  > ./sd/sirtsanity2000
	-./futhark/SIRT < ./data/sirtinputf32rad2048  > ./sd/sirtsanity2048
	-./futhark/SIRT < ./data/sirtinputf32rad2500  > ./sd/sirtsanity2500
	-./futhark/SIRT < ./data/sirtinputf32rad3000  > ./sd/sirtsanity3000
	-./futhark/SIRT < ./data/sirtinputf32rad3500  > ./sd/sirtsanity3500
	-./futhark/SIRT < ./data/sirtinputf32rad4000  > ./sd/sirtsanity4000
	-./futhark/SIRT < ./data/sirtinputf32rad4096  > ./sd/sirtsanity4096
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad64    > ./sd/sirt3Dsanity64
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad128   > ./sd/sirt3Dsanity128
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad256   > ./sd/sirt3Dsanity256
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad512   > ./sd/sirt3Dsanity512
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad1024  > ./sd/sirt3Dsanity1024
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad1500  > ./sd/sirt3Dsanity1500
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad2000  > ./sd/sirt3Dsanity2000
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad2048  > ./sd/sirt3Dsanity2048
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad2500  > ./sd/sirt3Dsanity2500
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad3000  > ./sd/sirt3Dsanity3000
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad3500  > ./sd/sirt3Dsanity3500
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad4000  > ./sd/sirt3Dsanity4000
	-./futhark/SIRT3D < ./data/sirt3Dinputf32rad4096  > ./sd/sirt3Dsanity4096
