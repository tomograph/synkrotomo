import "projection_lib"
open Projection
let main  (angles: []f32)
          (rays: []f32)
          (voxels: [][]f32)
          (projections: []f32) : []f32 =
          projection_difference_base angles rays voxels projections
