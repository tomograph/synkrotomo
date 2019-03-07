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
  (deltarho: f32)
  (numrhos: i32)
  (image : *[n]f32)
  (projections: [p]f32)
  (iterations : i32) : []f32 =
  let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let (proj_flat, proj_steep) = fix_projections projections angles numrhos
  let lines = preprocess angles

  let rowsums_steep = inverse (forwardprojection lines.2 rhozero deltarho numrhos halfsize (replicate n 1.0f32))
  let rowsums_flat = inverse (forwardprojection lines.1 rhozero deltarho numrhos halfsize (replicate n 1.0f32))

  let colsums_steep = inverse (bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize (replicate p 1.0f32))
  let colsums_flat = inverse (bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize (replicate p 1.0f32))

  -- hack to always do this!
  let imageT =  if (size < 10000)
                then flatten <| transpose <| copy (unflatten size size image)
                else (replicate n 1.0f32)

  let res_steep = loop (image) = (image) for iter < 100 do
    let fp_s = forwardprojection lines.2 rhozero deltarho numrhos halfsize image
    let fp_f = forwardprojection lines.1 rhozero deltarho numrhos halfsize imageT
    let fp_diff_s = map2 (-) proj_steep fp_s
    let fp_diff_f = map2 (-) proj_flat fp_f
    let fp_weighted_s = map2 (*) rowsums_steep fp_diff_s
    let fp_weighted_f = map2 (*) rowsums_flat fp_diff_f
    let bp_s = bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize fp_weighted_s
    let bp_f = bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize fp_weighted_f
    let bp_weighted_s = map2 (*) colsums_steep bp_s
    let bp_weighted_f = map2 (*) colsums_flat bp_f
    let bpwf = if (size < 10000) then flatten <| transpose <| copy (unflatten size size bp_weighted_f) else (replicate n 1.0f32)
    let resimg = map2 (+) bpwf bp_weighted_s
    in image with [0:n] = map2 (+) image resimg
  --
  -- let res_flat = loop (imageT) = (imageT) for iter < 100 do
  --   in imageT with [0:n] = map2 (+) imageT bp_weighted


  in res_steep
  -- in rowsums_flat
--
-- let main  [n][p](angles : []f32)
--           (rhozero : f32)
--           (deltarho: f32)
--           (numrhos:i32)
--           (image : *[n]f32)
--           (projections: [p]f32)
--           (iterations : i32) : []f32 =
--           SIRT angles rhozero deltarho numrhos image projections iterations
let main  [n][p][r](angles : []f32)
          (rhos : [r]f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations : i32) : []f32 =
          let rhozero = rhos[0]
          let deltarho = rhos[1]-rhozero
          in SIRT angles rhozero deltarho r image projections iterations

          -- SIRT angles rhos image projections iterations
