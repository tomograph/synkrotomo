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
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2
  let numrhos = p/a
  let (steep_lines, flat_lines, is_flat, _) = preprocess angles numrhos
  let (steep_proj, flat_proj) = fix_projections projections is_flat

  let rowsums_steep = inverse (fp steep_lines rhozero deltarho numrhos halfsize (replicate n 1.0f32))
  let rowsums_flat = inverse (fp flat_lines rhozero deltarho numrhos halfsize (replicate n 1.0f32))

  let inversecolumnsums = inverse (backprojection (replicate (length steep_proj) 1.0f32) (replicate (length flat_proj) 1.0f32) steep_lines flat_lines rhozero deltarho rhosprpixel numrhos halfsize)

  let res = loop (image) = (image) for iter < iterations do
      let imageT =  if (size < 10000)
                     then flatten <| transpose <| unflatten size size image
                     else (replicate n 1.0f32)
      let fp_steep = fp steep_lines rhozero deltarho numrhos halfsize image
      let fp_flat = fp flat_lines rhozero deltarho numrhos halfsize imageT
      let fp_diff_steep = map2 (-) steep_proj fp_steep
      let fp_diff_flat = map2 (-) flat_proj fp_flat
      let fp_weighted_steep = map2 (*) rowsums_steep fp_diff_steep
      let fp_weighted_flat = map2 (*) rowsums_flat fp_diff_flat
      let bp = backprojection fp_weighted_steep fp_weighted_flat steep_lines flat_lines rhozero deltarho rhosprpixel numrhos halfsize
      let bp_weighted = map2 (*) inversecolumnsums bp
      in image with [0:n] = map2 (+) image bp_weighted

  in res

let main  [n][p][a](angles : [a]f32)
           (rhozero : f32)
           (deltarho: f32)
           (image : *[n]f32)
           (projections: [p]f32)
           (iterations : i32) : [n]f32 =
           SIRT angles rhozero deltarho image projections iterations
