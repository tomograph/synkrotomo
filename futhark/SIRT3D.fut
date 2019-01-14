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

let inverse [n](values: [n]f32) : [n]f32 =
     map(\v -> if v == 0.0 then 0.0 else 1/v) values

let main  [n][p](angles : []f32)
          (rhos : []f32)
          (volume : *[n]f32)
          (projections: [p]f32)
          (iterations : i32)
          (size: i32) : [n]f32 =
          map(\image -> SIRT angles rhos image projections iterations) (unflatten volume n/size size)
