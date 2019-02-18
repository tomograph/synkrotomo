-- ==
-- input@../data/bpinputf32rad64
-- input@../data/bpinputf32rad128
-- input@../data/bpinputf32rad256
-- input@../data/bpinputf32rad512
-- input@../data/bpinputf32rad1024
-- input@../data/bpinputf32rad1500
-- input@../data/bpinputf32rad2000
-- input@../data/bpinputf32rad2048
-- input@../data/bpinputf32rad2500
-- input@../data/bpinputf32rad3000
-- input@../data/bpinputf32rad3500
-- input@../data/bpinputf32rad4000
-- input@../data/bpinputf32rad4096

import "projection_lib"
open Projection

let main  [p](angles : []f32)
          (rhos : []f32)
          (size : i32)
          (projections: [p]f32): []f32 =
  let rhozero = unsafe rhos[0]
  let deltarho = unsafe rhos[1]-rhozero
  in cur_best angles rhozero deltarho size projections
