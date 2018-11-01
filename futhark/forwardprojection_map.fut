-- ==
-- input@../test
import "projection_lib"
open Projection

let main  (angles: []f32)
          (rays: []f32)
          (voxels: []f32)
          (stepsize: i32) : []f32 =
          forwardprojection_map angles rays voxels stepsize
