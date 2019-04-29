lib:
	futhark pyopencl --library ./futhark/SIRT.fut
	futhark pyopencl --library ./futhark/poc.fut
	futhark pyopencl --library ./futhark/SIRT3Dtest.fut
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

# comdat:
# 	futhark opencl ./futhark/forwardprojection.fut
# 	./futhark/forwardprojection < data/fpinputf32rad128 >sanity128.out
# 	./futhark/forwardprojection < data/fpinputf32rad256 >sanity256.out
# 	./futhark/forwardprojection < data/fpinputf32rad512 >sanity512.out
# 	futhark opencl ./futhark/forwardprojection_test.fut
# 	./futhark/forwardprojection_test < data/fpinputf32rad128 >>sanity128.out
# 	./futhark/forwardprojection_test < data/fpinputf32rad256 >>sanity256.out
# 	./futhark/forwardprojection_test < data/fpinputf32rad512 >>sanity512.out
#
# comdatsirt:
# 	futhark opencl ./futhark/SIRT.fut
# 	./futhark/SIRT < data/sirtinputf32rad128 >sanity128.out
# 	./futhark/SIRT < data/sirtinputf32rad256 >sanity256.out
# 	./futhark/SIRT < data/sirtinputf32rad512 >sanity512.out
# 	futhark opencl ./futhark/SIRT_test.fut
# 	./futhark/SIRT_test < data/sirtinputf32rad128 >>sanity128.out
# 	./futhark/SIRT_test < data/sirtinputf32rad256 >>sanity256.out
# 	./futhark/SIRT_test < data/sirtinputf32rad512 >>sanity512.out

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
