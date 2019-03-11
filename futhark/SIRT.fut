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

let safe_inverse (value: f32) : f32 =
     if value == 0.0 then 0.0 else 1/value

let inverse (values: []f32) : []f32 =
     map(\v -> safe_inverse v) values

let SIRT [n][p][a](angles : [a]f32)
  (rhozero : f32)
  (deltarho: f32)
  (image : *[n]f32)
  (projections: [p]f32)
  (iterations : i32) : [n]f32 =
  let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
  let numrhos = p//a-1
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let (proj_flat, proj_steep) = fix_projections projections angles numrhos
  let lines = preprocess angles

  --let rowsums_steep = fp lines.2 rhozero deltarho numrhos halfsize (replicate n 1.0f32)
  --let rowsums_flat = fp lines.1 rhozero deltarho numrhos halfsize (replicate n 1.0f32)

  --let colsums_steep = inverse (bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize (replicate (length proj_steep) 1.0f32))
  --let colsums_flat = inverse (bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize (replicate (length proj_flat) 1.0f32))
  let rowsums_steep = fp lines.2 rhozero deltarho numrhos halfsize (replicate n 1)
  let rowsums_flat = fp lines.1 rhozero deltarho numrhos halfsize (replicate n 1)
  let rowsums = postprocess_fp angles rowsums_steep rowsums_flat numrhos

  let inverserowsums = inverse rowsums

  let colsums_steep = bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize (replicate (length proj_steep) 1.0f32)
  let colsums_flat = bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize  (replicate (length proj_flat) 1.0f32)
  let colsums_flatT =  if (size < 10000)
               then flatten <| transpose <| unflatten size size colsums_flat
               else (replicate (size**2) 1.0f32)
  let colsums = map2 (+) colsums_steep colsums_flatT
  let inversecolumnsums = inverse colsums

  -- hack to always do this!
  let imageT =  if (size < 10000)
                then flatten <| transpose <| copy (unflatten size size image)
                else (replicate n 1.0f32)

  let res = loop (image) = (image) for iter < iterations do
      let fp_steep = fp lines.2 rhozero deltarho numrhos halfsize image
      let fp_flat = fp lines.1 rhozero deltarho numrhos halfsize imageT
      let fp = postprocess_fp angles fp_steep fp_flat numrhos
      let fp_diff = map2 (-) projections fp
      let fp_weighted = map2 (*) inverserowsums fp_diff
      let (flat_proj, steep_proj) = fix_projections fp_weighted angles numrhos
      let bp_steep = bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize steep_proj
      let bp_flat = bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize flat_proj
      --untranspose in flat case
      let bp_flatT =  if (size < 10000)
                   then flatten <| transpose <| unflatten size size bp_flat
                   else (replicate (size**2) 1.0f32)
      let bp = map2 (+) bp_steep bp_flatT
      let bp_weighted = map2(*) inversecolumnsums bp
      in image with [0:n] = map2 (+) image bp_weighted

  -- let res_steep = loop (image) = (image) for iter < iterations do
  --    let fp_steep = fp lines.2 rhozero deltarho numrhos halfsize image
  --    let fp_diff = map2 (-) proj_steep fp_steep
  --    let fp_weighted = map2 (*) rowsums_steep fp_diff
  --    let bp_steep = bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize fp_weighted
  --    let bp_weighted = map2 (*) colsums_steep bp_steep
  --    in image with [0:n] = map2 (+) image bp_weighted
  --
  -- let res_flat = loop (imageT) = (copy imageT) for iter < iterations do
  --    let fp_flat = fp lines.1 rhozero deltarho numrhos halfsize imageT
  --    let fp_diff = map2 (-) proj_flat fp_flat
  --    let fp_weighted = map2 (*) rowsums_flat fp_diff
  --    let bp_flat = bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize fp_weighted
  --    let bp_weighted = map2 (*) colsums_flat bp_flat
  --    in imageT with [0:n] = map2 (+) imageT bp_weighted

     -- let imageUT = if (size < 10000)
     --            then flatten <| transpose <| unflatten size size res_flat
     --            else (replicate n 1.0f32)

  in res--map2 (+) res_steep imageUT

let main  [n][p][a](angles : [a]f32)
           (rhozero : f32)
           (deltarho: f32)
           (image : *[n]f32)
           (projections: [p]f32)
           (iterations : i32) : [n]f32 =
           SIRT angles rhozero deltarho image projections iterations
