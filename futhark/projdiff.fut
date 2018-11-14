import "matrix_lib"
open Matrix
let main  [n](angles : []f32)
          (rays : []f32)
          (image : *[n]f32)
          (iterations : i32) : []f32 =
          projection_difference angles rays image
