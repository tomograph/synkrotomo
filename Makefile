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
	-futhark bench --runs=10 --backend=opencl ./futhark/forwardprojection.fut

benchbp:
	-futhark bench --runs=10 --backend=opencl ./futhark/backprojection.fut

benchbptest:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection_test.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection_test.fut

benchbptest2:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection_test_2.fut
	futhark bench --runs=10 --skip-compilation ./futhark/backprojection_test_2.fut

benchbptest3:
	futhark bench --runs=10 --backend=opencl ./futhark/backprojection_test.fut

benchbptest4:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/backprojection_test_2.fut
	futhark bench --runs=10 --backend=opencl ./futhark/backprojection_test_2.fut


benchsirt:
	-futhark bench --runs=1 --backend=opencl ./futhark/SIRT.fut

benchsirttest:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT_test.fut
	futhark bench --runs=1 --skip-compilation ./futhark/SIRT_test.fut

benchsirt3d:
	-futhark bench --runs=1 --backend=opencl ./futhark/SIRT3D.fut

benchall: benchfp benchbp benchsirt benchsirt3d

test:
	futhark opencl ./futhark/forwardprojection_test.fut
	# ./futhark/forwardprojection_test < data/fpinputf32rad256 >test

clean:
	rm -f ./futhark/*.c ./futhark/backprojection ./futhark/forwardprojection ./futhark/SIRT ./futhark/SIRT3D ./futhark/sanityCheck ./futhark/forwardprojection_test
