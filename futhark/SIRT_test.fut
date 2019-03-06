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

let SIRT [n] [p] [r] (angles : []f32)
  (rhos : [r]f32)
  (image : *[n]f32)
  (projections: [p]f32)
  (iterations : i32) : [n]f32 =
  let rhozero = unsafe rhos[0]
  let deltarho = unsafe rhos[1]-rhozero
  let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let (proj_flat, proj_steep) = fix_projections projections angles r
  let lines = preprocess angles

  let rowsums_steep = forwardprojection_steep lines.2 rhozero deltarho r halfsize (replicate n 1)
  let rowsums_flat = forwardprojection_flat lines.1 rhozero deltarho r halfsize (replicate n 1)
  let inverserowsums = inverse (rowsums_steep ++ rowsums_flat)

  let colsums_steep = bp_steep lines.2 rhozero deltarho rhosprpixel r halfsize (replicate p 1)
  let colsums_flat = bp_flat lines.1 rhozero deltarho rhosprpixel r halfsize (replicate p 1)
  let inversecolumnsums = inverse <| map2 (+) colsums_steep colsums_flat

  let imgT = copy(image)

  let res_steep = loop (image) = (image) for iter < iterations do
    let fp_steep = forwardprojection_steep lines.2 rhozero deltarho r halfsize image
    let fp_diff = map2 (-) proj_steep fp_steep
    let fp_weighted = map2 (*) rowsums_steep fp_diff
    let bp_steep = bp_steep lines.2 rhozero deltarho rhosprpixel r halfsize    fp_weighted
    let bp_weighted = map2 (*) inversecolumnsums bp_steep
    in image with [0:n] = map2 (+) image bp_weighted

  let res_flat = loop (imgT) = (imgT) for iter < iterations do
    let fp_flat = forwardprojection_flat lines.1 rhozero deltarho r halfsize imgT
    let fp_diff = map2 (-) proj_flat fp_flat
    let fp_weighted = map2 (*) rowsums_flat fp_diff
    let bp_flat = bp_flat lines.1 rhozero deltarho rhosprpixel r halfsize    fp_weighted
    let bp_weighted = map2 (*) inversecolumnsums bp_flat
    in imgT with [0:n] = map2 (+) imgT bp_weighted

  in map2 (+) res_steep res_flat

let main  [n][p](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : [n]f32 =
          SIRT angles rhos image projections iterations
