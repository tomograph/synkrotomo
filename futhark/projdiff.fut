-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
import "matrix_lib"
open Matrix
let main  [n](angles : []f32)
          (rays : []f32)
          (image : [n][n]f32)
          (projections: *[]f32)
          (iterations : i32) : []f32 =
          projection_difference angles rays image projections
