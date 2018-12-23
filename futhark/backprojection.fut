-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- input@../data/sirtinputf32rad1024
-- input@../data/sirtinputf32rad2048
import "projection_lib"
open Projection

let main  [p](angles : []f32)
          (rhos : []f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          in back_projection angles rhozero deltarho size projections
