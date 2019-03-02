-- ==
-- input@../data/fpinputf32rad64
-- input@../data/fpinputf32rad128
-- input@../data/fpinputf32rad256
-- input@../data/fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- input@../data/fpinputf32rad2048
-- input@../data/fpinputf32rad4096

import "projection_lib"
open Projection
-- import "preprocessing"

let main  [n][r] (angles : []f32)
          (rhos : [r]f32)
          (image : *[n]f32) =
          let size = t32(f32.sqrt(r32(n)))
          let halfsize = size/2
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let numrhos = r
          let lines = preprocess_2 angles
          -- let (lines, rhozero, deltarho, numrhos) = preprocessing angles rhos
          let steep = forwardprojection_steep lines.2 rhozero deltarho numrhos halfsize image
          let flat = forwardprojection_flat lines.1 rhozero deltarho numrhos halfsize image
          in steep ++ flat
          -- let dat = steep ++ flat
          -- let vals = map (\(v, _) -> v) dat
          -- let inds = map (\(_, i) -> i) dat
          -- in scatter (replicate ((length steep) + (length flat)) 0.0f32) inds vals
