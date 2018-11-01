-- ==
-- input@../data/matrixinputf32rad64
-- input@../data/matrixinputf32rad128
-- input@../data/matrixinputf32rad256
-- input@../data/matrixinputf32rad512
-- input@../data/matrixinputf32rad1028
-- input@../data/matrixinputf32rad2048
-- input@../data/matrixinputf32rad4096
import "matrix_lib"
open Matrix
let main  (angles: []f32)
          (rays: []f32)
          (gridsize: i32) : [][](f32,i32) =
     weights_doublepar angles rays gridsize
