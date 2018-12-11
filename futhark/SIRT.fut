-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
import "projection_lib"
open Projection

let main  [n](angles : []f32)
          (rhos : []f32)
          (size: i32)
          (image : *[n]f32)
          (projections: []f32)
          (iterations : i32) : [n]f32 =
          let res = loop (image) = (image) for iter < iterations do
               let fpdiff = map2 (-) projections (forward_projection angles rhos (size/2) image)
               -- use stepsize = n for now (i.e one angle at a time)
               let bp = backprojection angles rhos fpdiff size size
               in (image with [0:n] = (map2 (+) image bp))
          in res
