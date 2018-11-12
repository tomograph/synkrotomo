import "matrix_lib"
open Matrix
let main  (angles: []f32)
          (rays: []f32)
          (voxels: [][]f32)
          (projections: []f32) : []f32 =
          projection_difference angles rays voxels projections
