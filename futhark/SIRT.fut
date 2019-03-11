-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- input@../data/sirtinputf32rad1024
-- input@../data/sirtinputf32rad2048
-- input@../data/sirtinputf32rad4049


import "sirtlib"
open sirtlib

let inverse [n](values: [n]f32) : [n]f32 =
     map(\v -> if v == 0.0 then 0.0 else 1/v) values

let SIRT [n][p](angles : []f32)
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

  let colsums_steep = inverse (bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize (replicate (length proj_steep) 1.0f32))
  let colsums_flat = inverse (bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize (replicate (length proj_flat) 1.0f32))

  -- hack to always do this!
  let imageT =  if (size < 10000)
                then flatten <| transpose <| copy (unflatten size size image)
                else (replicate n 1.0f32)

  let res_steep = loop (image) = (copy image) for iter < iterations do
    let fp = forwardprojection lines.2 rhozero deltarho numrhos halfsize image
    let fp_diff = map2 (-) proj_steep fp
    let fp_weighted = map2 (*) rowsums_steep fp_diff
    let bp = bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize fp_weighted
    let bp_weighted = map2 (*) colsums_steep bp
    in image with [0:n] = map2 (+) image bp_weighted

  let res_flat = loop (imageT) = (copy imageT) for iter < iterations do
    let fp = forwardprojection lines.1 rhozero deltarho numrhos halfsize imageT
    let fp_diff = map2 (-) proj_flat fp
    let fp_weighted = map2 (*) rowsums_flat fp_diff
    let bp = bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize fp_weighted
    let bp_weighted = map2 (*) colsums_flat bp
    in imageT with [0:n] = map2 (+) imageT bp_weighted

  let imageUT = if (size < 10000)
                then flatten <| transpose <| unflatten size size (reverse res_flat)
                else (replicate n 1.0f32)

  in map2 (+) (reverse res_steep) imageUT

let main  [n][p](angles : []f32)
           (rhozero : f32)
           (deltarho: f32)
           (numrhos:i32)
           (image : *[n]f32)
           (projections: [p]f32)
           (iterations : i32) : [n]f32 =
           SIRT angles rhozero deltarho numrhos image projections iterations
