echo "compiling"
cd ../futhark
FUTHARK_INCREMENTAL_FLATTENING=1 futhark-opencl  projdiff.fut
cd ..
python testprojdiff.py
