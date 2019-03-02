-- ==
-- input@../data/sirtinputf32rad64
-- output@../sd/sanity-sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- output@../sd/sanity-sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- output@../sd/sanity-sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- output@../sd/sanity-sirtinputf32rad512
-- input@../data/sirtinputf32rad1024
-- output@../sd/sanity-sirtinputf32rad1024
-- input@../data/sirtinputf32rad2048
-- output@../sd/sanity-sirtinputf32rad2048


import "projection_lib"
open Projection

let inverse [n](values: [n]f32) : [n]f32 =
     map(\v -> if v == 0.0 then 0.0 else 1/v) values

let SIRT [n][p](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : [n]f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let size = t32(f32.sqrt(r32(n)))
          let halfsize = size/2
          let inverserowsums = inverse (forward_projection angles rhos halfsize (replicate n 1))
          let inversecolumnsums = inverse (back_projection angles rhozero deltarho size (replicate p 1))
          let res = loop (image) = (image) for iter < iterations do
               --(image with [0:n] = map(\v -> f32.min 1.0 v)(map(\v -> f32.max 0.0 v)(map2 (+) image (map2 (*) inversecolumnsums (back_projection angles rhozero deltarho size (map2 (*) inverserowsums (map2 (-) projections (forward_projection angles rhos halfsize image))))))))
               (image with [0:n] = (map2 (+) image (map2 (*) inversecolumnsums (back_projection angles rhozero deltarho size (map2 (*) inverserowsums (map2 (-) projections (forward_projection angles rhos halfsize image)))))))
          in res

let main  [n][p](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : [n]f32 =
          SIRT angles rhos image projections iterations
