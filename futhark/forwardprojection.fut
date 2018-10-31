-- ==
-- input@../test
import "projection_lib"
open Projection

let main  (rays: []f32)
          (angles: []f32)
          (voxels: []f32) : []f32 =
          forwardprojection rays angles voxels
