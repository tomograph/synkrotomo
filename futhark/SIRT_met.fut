-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- input@../data/sirtinputf32rad1024

import "projection_lib"
open Projection

let inverse [n](values: [n]f32) : [n]f32 =
     map(\v -> if v == 0.0 then 0.0 else 1/v) values

let SIRT [n][p][r](angles : []f32)
          (rhos : [r]f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : [n]f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let size = t32(f32.sqrt(r32(n)))
          let halfsize = size/2
          let inverserowsums = inverse (forward_projection angles rhos halfsize (replicate n 1))
          let inversecolumnsums = inverse (back_projection angles rhozero deltarho size (replicate p 1))
          let lines = preprocess angles
          let res = loop (image) = (image) for iter < iterations do
               -- let fp = forward_projection angles rhos halfsize image
               -- let fp_diff = map2 (-) projections fp
               -- let fp_weighted = map2 (*) inverserowsums fp_diff
               -- let steep = bp_steep lines.2 rhozero deltarho rhosprpixel r halfsize fp_weighted
               -- let flat = bp_flat lines.1 rhozero deltarho rhosprpixel r halfsize fp_weighted
               -- let bp = map2 (+) steep flat
               -- let bp_weighted = map2 (*) inversecolumnsums bp
               -- in image with [0:n] = map2 (+) image bp_weighted

               (image with [0:n] = (map2 (+) image (map2 (*) inversecolumnsums (back_projection_met lines rhozero deltarho rhosprpixel r halfsize (map2 (*) inverserowsums (map2 (-) projections (forward_projection angles rhos halfsize image)))))))
          in res

let main  [n][p](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : [n]f32 =
          SIRT angles rhos image projections iterations
