echo "compiling"
cd ..
cd futhark

futhark-pyopencl --library backprojection_jh.fut
futhark-pyopencl --library backprojection_doubleparallel.fut
futhark-pyopencl --library backprojection_map.fut
echo "saving sinograms"

cd ..
python generatebackprojections.py
