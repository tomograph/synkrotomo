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
  (numrhos: i32)
  (image : *[n]f32)
  (projections: [p]f32)
  (iterations : i32) : [n]f32 =
  let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let (proj_flat, proj_steep) = fix_projections projections angles numrhos
  let lines = preprocess angles

  let rowsums_steep = inverse (fp lines.2 rhozero deltarho numrhos halfsize (replicate n 1.0f32))
  let rowsums_flat = inverse (fp lines.1 rhozero deltarho numrhos halfsize (replicate n 1.0f32))

  let colsums_steep = inverse (bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize (replicate (length proj_steep) 1.0f32))
  let colsums_flat = inverse (bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize (replicate (length proj_flat) 1.0f32))

  -- hack to always do this!
  -- let imageT =  if (size < 10000)
  --               then flatten <| transpose <| copy (unflatten size size image)
  --               else (replicate n 1.0f32)

      let res = loop (image) = (image) for iter < iterations do
           --(image with [0:n] = map(\v -> f32.min 1.0 v)(map(\v -> f32.max 0.0 v)(map2 (+) image (map2 (*) inversecolumnsums (back_projection angles rhozero deltarho size (map2 (*) inverserowsums (map2 (-) projections (forward_projection angles rhos halfsize image))))))))
           (image with [0:n] = (map2 (+) image (map2 (*) colsums_steep (bp lines.2 rhozero deltarho rhosprpixel numrhos halfsize (map2 (*) rowsums_steep (map2 (-) proj_steep (fp lines.2 rhozero deltarho numrhos halfsize image)))))))
      in res

  -- let res_flat = loop (imageT) = (copy imageT) for iter < iterations do
  --    (imageT with [0:n] = (map2 (+) imageT (map2 (*) colsums_flat (bp lines.1 rhozero deltarho rhosprpixel numrhos halfsize (map2 (*) rowsums_flat (map2 (-) proj_flat (fp lines.1 rhozero deltarho numrhos halfsize imageT)))))))
  -- in res_flat

  -- let imageUT = if (size < 10000)
  --               then flatten <| transpose <| unflatten size size res_flat
  --               else (replicate n 1.0f32)

  in res --imageUT

let main  [n][p][a](angles : [a]f32)
           (rhozero : f32)
           (deltarho: f32)
           (numrhos:i32)
           (image : *[n]f32)
           (projections: [p]f32)
           (iterations : i32) : [n]f32 =
           SIRT angles rhozero deltarho numrhos image projections iterations
