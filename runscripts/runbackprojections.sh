echo "runbackprojections"
echo "compiling"
cd ..
cd futhark
futhark-pyopencl --library backprojection_jh.fut
futhark-pyopencl --library backprojection_doubleparallel.fut
futhark-pyopencl --library backprojection_map.fut
# futhark-pyopencl --library backprojection_semiflat.fut
echo "saving backprojections"
cd ..
python generatebackprojections.py
