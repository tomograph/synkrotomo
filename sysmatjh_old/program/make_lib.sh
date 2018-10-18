[[ ":$PATH:" != *":./lib/:"* ]] && PATH="./lib/:${PATH}"
[[ ":$CPATH:" != *":$HOME/.local/include/:/usr/local/cuda/include:"* ]] && CPATH="$HOME/.local/include/:/usr/local/cuda/include:${CPATH}"
[[ ":$LIBRARY_PATH:" != *":$HOME/.local/lib:/usr/local/cuda/lib64:"* ]] && LIBRARY_PATH="$HOME/.local/lib:/usr/local/cuda/lib64:${LIBRARY_PATH}"
[[ ":$LD_LIBRARY_PATH:" != *":$HOME/.local/lib:/usr/local/cuda/lib64/:"* ]] && LD_LIBRARY_PATH="$HOME/.local/lib:/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}"


LIB_PATH="./lib"
LIBRARIES="-lm -fPIC -shared -L$LIB_PATH -I$LIB_PATH"
if [ "$2" = "linux" ]; then 
    OPENCL_LIB="-lOpenCL" 
else 
    OPENCL_LIB="-framework OpenCL" 
fi


case "$1" in
"opencl" )
    echo "⚙️  compiling to OpenCL"
    futhark-opencl --library src/intersections.fut -o lib/intersections
    echo "⛓  linking the library"
    echo "#include <stdint.h>"|cat - ./lib/intersections.c > /tmp/out && mv /tmp/out ./lib/intersections.c
    gcc -std=c11 lib/intersections.c -o lib/libintersections.so $LIBRARIES $OPENCL_LIB
    ;;
"c")
    echo "⚙️  compiling to C"
    futhark-c --library src/intersections.fut -o lib/intersections
    echo "⛓  linking the library"        
    gcc -std=c11 lib/intersections.c -o lib/libintersections.so $LIBRARIES
    ;;
"pyopencl")
    echo "⚙️  compiling to PyOpenCL"
    futhark-pyopencl src/intersections.fut -o lib/intersections.py
    ;;
"python")
    echo "⚙️  compiling to Python"
    futhark-py src/intersections.fut -o lib/intersections.py
    ;;
*)
    echo "❌  Unknown compile target!"
    exit 1;
    ;;
esac

cp -R lib/. benchmarking/load_time/lib/
echo "🙌 done!"
