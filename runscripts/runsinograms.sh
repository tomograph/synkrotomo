echo "compiling"
cd ..
cd futhark

futhark-pyopencl --library forwardprojection_jh.fut
futhark-pyopencl --library forwardprojection_doubleparallel.fut
futhark-pyopencl --library forwardprojection_map.fut
echo "saving sinograms"

cd ..
python generatesinograms.py
