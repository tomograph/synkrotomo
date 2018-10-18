echo "🎁 compiling C files"

LIB_PATH="../../lib"
LIBRARIES="-lm -lintersections -L$LIB_PATH -I$LIB_PATH"
if [ "$2" = "linux" ]; then 
    OPENCL_LIB="-lOpenCL" 
else 
    OPENCL_LIB="-framework OpenCL" 
fi

case "$1" in
"opencl" )
    gcc src/bench_load_time.c -o bin/bench_load_time $LIBRARIES $OPENCL_LIB
    gcc src/make_results.c -o bin/make_results $LIBRARIES $OPENCL_LIB
    ;;
"c")
    gcc src/bench_load_time.c -o bin/bench_load_time $LIBRARIES
    gcc src/make_results.c -o bin/make_results $LIBRARIES   
    ;;
esac

echo "🙌 done!";
