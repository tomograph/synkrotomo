lib:
	futhark pyopencl --library ./futhark/SIRT.fut
	futhark pyopencl --library ./futhark/backprojection.fut
	futhark pyopencl --library ./futhark/forwardprojection.fut

libc:
	futhark c --library ./futhark/SIRT.fut
	futhark c --library ./futhark/backprojection.fut
	futhark c --library ./futhark/forwardprojection.fut

benchfp:
	-futhark bench --runs=10 --backend=opencl ./futhark/forwardprojection.fut

benchbp:
	-futhark bench --runs=10 --backend=opencl ./futhark/backprojection.fut

benchsirt:
	-futhark bench --runs=1 --backend=opencl ./futhark/SIRT.fut

benchall: benchfp benchbp benchsirt
