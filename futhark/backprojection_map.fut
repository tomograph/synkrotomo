-- ==
-- input@../test
import "projection_lib"
open Projection

let main  (rays : []f32)
          (angles : []f32)
          (projections : []f32)
          (gridsize: i32)
          (stepSize : i32) : []f32 =
          backprojection_map rays angles projections gridsize stepSize
