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

opencl:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/SIRT.fut

runopencl: opencl
	./SIRT

runpytest: lib
	python samples/futhark_bp.py
	python samples/futhark_fp.py
	python samples/futhark_SIRT.py
	python samples/futhark_SIRT3D.py

benchfp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/forwardprojection.fut
	nohup futhark-bench --runs=10 --skip-compilation ./futhark/forwardprojection.fut > ./output/benchmarks/fp &>/dev/null &

benchbp:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/backprojection.fut
	nohup futhark-bench --runs=10 --skip-compilation ./futhark/backprojection.fut > ./output/benchmarks/bp &>/dev/null &

benchsirt:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/SIRT.fut
	nohup futhark-bench --runs=1 --skip-compilation ./futhark/SIRT.fut > ./output/benchmarks/sirt &>/dev/null &

benchsirt3d:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl ./futhark/SIRT3D.fut
	nohup futhark-bench --runs=1 --skip-compilation ./futhark/SIRT3D.fut > ./output/benchmarks/sirt3d &>/dev/null &

benchall:
	benchfp
	benchbp
	benchsirt
	benchsirt3d
