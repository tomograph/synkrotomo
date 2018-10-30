echo "compiling"
cd futhark
futhark-opencl forwardprojection.fut
echo "running benchmarks"
cat ../test | ./forwardprojection -D -b -t /dev/stderr 1>/dev/null 2>mem_usage_out
cd ..
