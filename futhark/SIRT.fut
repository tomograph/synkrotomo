-- ==
-- input@data/sirtinputf32rad64
-- output@data/sirtoutputf32rad
-- input@data/sirtinputf32rad128
-- output@data/sirtoutputf32rad128
-- input@data/sirtinputf32rad256
-- input@data/sirtinputf32rad512
-- input@data/sirtinputf32rad1024
-- input@data/sirtinputf32rad1500
-- input@data/sirtinputf32rad2000
-- input@data/sirtinputf32rad2048
-- input@data/sirtinputf32rad3000
-- input@data/sirtinputf32rad3500
-- input@data/sirtinputf32rad4000
-- input@data/sirtinputf32rad4096

import "backprojection"
open bplib
import "forwardprojection"
open fplib

let SIRT [n][p][a](angles : [a]f32)
    (rhozero : f32)
    (deltarho: f32)
    (image : *[n]f32)
    (projections: [p]f32)
    (iterations : i32) : [n]f32 =
          let size = t32(f32.sqrt(r32(n)))
          let numrhos = p/a
          let inverserowsums = inverse (forwardprojection angles rhozero deltarho numrhos (replicate n 1))
          let inversecolumnsums = inverse (backprojection angles rhozero deltarho size (replicate p 1))
          let res = loop (image) = (image) for iter < iterations do
               (image with [0:n] = (map2 (+) image (map2 (*) inversecolumnsums (backprojection angles rhozero deltarho size (map2 (*) inverserowsums (map2 (-) projections (forwardprojection angles rhozero deltarho numrhos image)))))))
          in res

let main  [n][p][a](angles : [a]f32)
      (rhozero : f32)
      (deltarho: f32)
      (image : *[n]f32)
      (projections: [p]f32)
      (iterations : i32) : [n]f32 =
      SIRT angles rhozero deltarho image projections iterations
