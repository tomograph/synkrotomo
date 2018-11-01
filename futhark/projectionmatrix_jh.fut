-- ==
-- input@../data/matrixinputf32rad128
-- input@../data/matrixinputf32rad256
import "matrix_lib"
open Matrix
let main  (angles: []f32)
         (rays: []f32)
         (gridsize: i32) : [][](f32,i32) =
    weights_jh angles rays gridsize
