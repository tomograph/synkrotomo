import "projection_lib"
open Projection

let main  (rays: []f32)
          (angles: []f32)
          (sinogram: []f32)
          (size: i32): []f32 =
          backprojection rays angles sinogram size
