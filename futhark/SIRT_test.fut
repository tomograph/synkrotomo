-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- input@../data/sirtinputf32rad1024
-- input@../data/sirtinputf32rad2048
-- input@../data/sirtinputf32rad4049


import "testlib"
open testlib

let inverse [n](values: [n]f32) : [n]f32 =
     map(\v -> if v == 0.0 then 0.0 else 1/v) values

let SIRT [n] [p] (angles : []f32)
  (rhozero : f32)
  (deltaho: f32)
  (image : *[n]f32)
  (projections: [p]f32)
  (iterations : i32) : [n]f32 =
  let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let (proj_flat, proj_steep) = fix_projections projections angles r
  let lines = preprocess angles

  let rowsums_steep = forwardprojection lines.2 rhozero deltarho r halfsize (replicate n 1)
  let rowsums_flat = forwardprojection lines.1 rhozero deltarho r halfsize (replicate n 1)

  let colsums_steep = bp lines.2 rhozero deltarho rhosprpixel r halfsize (replicate p 1)
  let colsums_flat = bp lines.1 rhozero deltarho rhosprpixel r halfsize (replicate p 1)

  -- hack to always do this!
  let imageT = if (size < 10000)
             then flatten <| transpose <| (unflatten size size image)
             else image

  let res_steep = loop (image) = (image) for iter < iterations do
    let fp = forwardprojection lines.2 rhozero deltarho r halfsize image
    let fp_diff = map2 (-) proj_steep fp
    let fp_weighted = map2 (*) rowsums fp_diff
    let bp = bp lines.2 rhozero deltarho rhosprpixel r halfsize fp_weighted
    let bp_weighted = map2 (*) colsums_steep bp
    in image with [0:n] = map2 (+) image bp_weighted

  let res_flat = loop (imageT) = (imageT) for iter < iterations do
    let fp = forwardprojection lines.1 rhozero deltarho r halfsize imageT
    let fp_diff = map2 (-) proj_flat fp
    let fp_weighted = map2 (*) rowsums_flat fp_diff
    let bp = bp_flat lines.1 rhozero deltarho rhosprpixel r halfsize fp_weighted
    let bp_weighted = map2 (*) colsums_flat bp
    in imageT with [0:n] = map2 (+) imageT bp_weighted

  let imageUT = if (size < 10000)
                then flatten <| transpose <| (unflatten size size imageT)
                else imageT

  in map2 (+) res_steep imageUT

let main  [n][p](angles : []f32)
          (rhozero : f32)
          (deltaho: f32)
          (numrhos:i32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : [n]f32 =
          SIRT angles rhozero deltaho numrhos image projections iterations
