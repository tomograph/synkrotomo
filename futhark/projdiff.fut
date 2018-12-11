-- currently does a forwardprojection using what will become calculation of the differnece between the projections and the forwardprojection
import "matrix_lib"
open Matrix
let main  [n](angles : []f32)
          (rays : []f32)
          (size: i32)
          (image : *[n]f32)
          (iterations : i32) : []f32 =
          projection_difference angles rays (size/2) image
