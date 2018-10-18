## Purpose
The purpose of the project is to speed up a mathematical modelling of a subject scanned using a synchrotron scanner. This will be attempted by exploiting the parallelisation capabilities of the GPU. The functional programming language Futhark has been chosen as it provides features that fit the project well.

## Installation
Compiling the source code requires installation of the Futhark language as described [here](https://futhark.readthedocs.io/en/latest/installation.html).


## Running the program
Navigate to the program folder:

```cd program```

Then compile the program as a library:

```futhark-py intersections.fut --library```

Execute the program my running the `main.py` script:

```python main.py```

## Benchmarking (WIP)
For openCL, compile program:
```futhark-opencl --library intersections.fut```
Link the library:
```gcc intersections.c -o libintersections.so -fPIC -shared -framework OpenCL```
Compile the program (main.c) that calls our futhark code:
```gcc main.c -o main -lm -lintersections -L./ -framework OpenCL```
Run the resulting file:
```time ./main```

or you could just do 
```sh make.sh [opencl | c] [filename]```