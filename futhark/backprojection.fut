-- ==
-- input@../data/bpinputf32rad64
-- input@../data/bpinputf32rad128
-- input@../data/bpinputf32rad256
-- input@../data/bpinputf32rad512
-- input@../data/bpinputf32rad1024

import "projection_lib"
open Projection

let main  [p](angles : []f32)
          (rhos : []f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          in back_projection_expand angles rhozero deltarho size projections
