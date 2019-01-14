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

let main  [n][p][a][r](angles : [a]f32)
          (rhos : [r]f32)
          (volume : *[n]f32)
          (projections: [p]f32)
          (iterations : i32)
          (size: i32) : [n]f32 =
          map(\i -> SIRT angles rhos image[i*size*size:(i+1)*size*size] projections[i*r*a:(i+1)*r*a] iterations) (iota size)
