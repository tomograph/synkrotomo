-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- input@../data/sirtinputf32rad1024
-- input@../data/sirtinputf32rad2048
-- input@../data/sirtinputf32rad4096
import "projection_lib"
open Projection

let main  [n](angles : []f32)
          (rhos : []f32)
          (image : [n][n]f32)
          (projections: []f32)
          (iterations : i32) : [n][n]f32 =
          SIRT angles rhos image projections iterations
