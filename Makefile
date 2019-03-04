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
	./futhark/forwardprojection_test < data/fpinputf32rad64 1>test.out
	# ./futhark/forwardprojection_test -D --dump-opencl dumpcl.out < data/fpinputf32rad512 1>test.out 2>dump.out

benchtest:
	futhark bench --runs=10 --backend=opencl ./futhark/forwardprojection_test.fut

comdat:
	futhark opencl ./futhark/forwardprojection.fut
	./futhark/forwardprojection < data/fpinputf32rad256 >sanity.out
	futhark opencl ./futhark/forwardprojection_test2.fut
	./futhark/forwardprojection_test2 < data/fpinputf32rad256 >>sanity.out
	# futhark opencl ./futhark/forwardprojection_best.fut
	# ./futhark/forwardprojection_best < data/fpinputf32rad256 >>sanity.out

tools:
	futhark opencl ./futhark/sanityCheck.fut
	futhark opencl ./futhark/mse.fut
	futhark opencl ./futhark/diff.fut

compareTest: comdat
	./futhark/sanityCheck < sanity.out > compare.out
	./futhark/mse < sanity.out >> compare.out
	./futhark/diff < sanity.out >> compare.out
	rm -f sanity.out
	# data/fpinputf32rad64

clean:
	rm -f ./futhark/*.c ./futhark/backprojection ./futhark/forwardprojection ./futhark/SIRT ./futhark/SIRT3D ./futhark/sanityCheck ./futhark/forwardprojection_test out ./futhark/mse
