lib:
	futhark pyopencl --library ./futhark/SIRT.fut
	futhark pyopencl --library ./futhark/backprojection.fut
	futhark pyopencl --library ./futhark/forwardprojection_transpose_L.fut
	futhark pyopencl --library ./futhark/SIRT_transpose.fut
	futhark pyopencl --library ./futhark/backprojection_transpose.fut
	futhark pyopencl --library ./futhark/forwardprojection_transpose.fut

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
	# FUTHARK_INCREMENTAL_FLATTENING=1 futhark opencl ./futhark/SIRT_test.fut
	-futhark bench --runs=1 --backend=opencl ./futhark/SIRT_test.fut

comparesirt: benchsirttest benchsirt

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
	./futhark/forwardprojection < data/fpinputf32rad128 >sanity128.out
	./futhark/forwardprojection < data/fpinputf32rad256 >sanity256.out
	./futhark/forwardprojection < data/fpinputf32rad512 >sanity512.out
	futhark opencl ./futhark/forwardprojection_test.fut
	./futhark/forwardprojection_test < data/fpinputf32rad128 >>sanity128.out
	./futhark/forwardprojection_test < data/fpinputf32rad256 >>sanity256.out
	./futhark/forwardprojection_test < data/fpinputf32rad512 >>sanity512.out

comdatsirt:
	futhark opencl ./futhark/SIRT.fut
	./futhark/SIRT < data/sirtinputf32rad128 >sanity128.out
	./futhark/SIRT < data/sirtinputf32rad256 >sanity256.out
	./futhark/SIRT < data/sirtinputf32rad512 >sanity512.out
	futhark opencl ./futhark/SIRT_test.fut
	./futhark/SIRT_test < data/sirtinputf32rad128 >>sanity128.out
	./futhark/SIRT_test < data/sirtinputf32rad256 >>sanity256.out
	./futhark/SIRT_test < data/sirtinputf32rad512 >>sanity512.out

tools:
	futhark opencl ./futhark/sanityCheck.fut
	futhark opencl ./futhark/mse.fut
	futhark opencl ./futhark/diff.fut

compareTest: comdat
	./futhark/sanityCheck < sanity128.out > compare.out
	./futhark/mse < sanity128.out >> compare.out
	./futhark/diff < sanity128.out >> compare.out
	./futhark/sanityCheck < sanity256.out >> compare.out
	./futhark/mse < sanity256.out >> compare.out
	./futhark/diff < sanity256.out >> compare.out
	./futhark/sanityCheck < sanity512.out >> compare.out
	./futhark/mse < sanity512.out >> compare.out
	./futhark/diff < sanity512.out >> compare.out
	rm -f sanity.out

compareSIRT: comdatsirt
	./futhark/sanityCheck < sanity128.out > compare.out
	./futhark/mse < sanity128.out >> compare.out
	./futhark/diff < sanity128.out >> compare.out
	./futhark/sanityCheck < sanity256.out >> compare.out
	./futhark/mse < sanity256.out >> compare.out
	./futhark/diff < sanity256.out >> compare.out
	./futhark/sanityCheck < sanity512.out >> compare.out
	./futhark/mse < sanity512.out >> compare.out
	./futhark/diff < sanity512.out >> compare.out
	rm -f sanity.out
	# data/fpinputf32rad64

clean:
	rm -f ./futhark/*.c ./futhark/backprojection ./futhark/forwardprojection ./futhark/SIRT ./futhark/SIRT3D ./futhark/sanityCheck ./futhark/forwardprojection_test ./futhark/forwardprojection_best out ./futhark/mse
