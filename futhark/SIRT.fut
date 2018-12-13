-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
import "projection_lib"
open Projection

let main  [n][p](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : [n]f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let size = t32(f32.sqrt(r32(n)))
          let halfsize = size/2
          let inverserowsums = map2 (/) (replicate p 1) (forward_projection angles rhos halfsize (replicate n 1))
          let inversecolumnsums = map2 (/) (replicate n 1) (back_projection angles rhozero deltarho size (replicate p 1))
          let res = loop (image) = (image) for iter < iterations do
               (image with [0:n] = (map2 (+) image (map2 (*) inversecolumnsums (back_projection angles rhozero deltarho size (map2 (*) inverserowsums (map2 (-) projections (forward_projection angles rhos halfsize image)))))))
          in res
