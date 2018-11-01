-- ==
-- input@../data/fpinputf32rad64
-- input@../data/fpinputf32rad128
-- input@../data/fpinputf32rad256
-- input@../data/fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- input@../data/fpinputf32rad2048
-- input@../data/fpinputf32rad4096
import "projection_lib"
open Projection

let main  (angles: []f32)
          (rays: []f32)
          (voxels: []f32)
          (stepsize: i32) : []f32 =
          forwardprojection_jh angles rays voxels stepsize
