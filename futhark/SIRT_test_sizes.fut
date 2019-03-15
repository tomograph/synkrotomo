-- ==
-- input@data/sirtinputf32rad64
-- input@data/sirtinputf32rad128
-- input@data/sirtinputf32rad256
-- input@data/sirtinputf32rad512
-- input@data/sirtinputf32rad1024

import "forwardprojection"
open fpTlib
import "backprojection"
open bpTlib

let SIRT [n][p][a](angles : [a]f32)
  (rhozero : f32)
  (deltarho: f32)
  (image : *[n]f32)
  (projections: [p]f32)
  (iterations : i32) : (i32, i32, i32) =
  let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2
  let numrhos = p/a
  let (steep_lines, flat_lines, proj_division, _) = preprocess angles numrhos
  let (steep_proj, flat_proj) = fix_projections projections proj_division

  let rowsums_steep = inverse (fp steep_lines rhozero deltarho numrhos halfsize (replicate n 1.0f32))
  let rowsums_flat = inverse (fp flat_lines rhozero deltarho numrhos halfsize (replicate n 1.0f32))

  let inversecolumnsums = inverse (backprojection (replicate (length steep_proj) 1.0f32) (replicate (length flat_proj) 1.0f32) steep_lines flat_lines rhozero deltarho rhosprpixel numrhos halfsize)

  let imageT =  if (size < 10000)
                 then flatten <| transpose <| unflatten size size image
                 else (replicate n 1.0f32)
  let fp_steep = fp steep_lines rhozero deltarho numrhos halfsize image
  let fp_flat = fp flat_lines rhozero deltarho numrhos halfsize imageT

  -- do bool first then partition
  let angle_test_bool_first = flatten <| map(\a-> replicate numrhos (is_flat (f32.cos a) (f32.sin a)))(angles)
  let angle_parts_bool_first = partition(\b-> b)angle_test_bool_first

  -- do bool last partition
  let angle_test_bool_last = flatten <| map(\a-> replicate numrhos a)(angles)
  let angle_parts_bool_last = partition(\a-> (is_flat (f32.cos a) (f32.sin a)))angle_test_bool_last
  in ((length angle_parts_bool_first.1), (length angle_parts_bool_last.1), ((length flat_lines)*640))
  --in is_flat
  --in ((length parts.1), (length parts.2))
  --in ((length fp_steep), (length fp_flat), (length steep_proj), (length flat_proj), a, numrhos, (length steep_lines), (length flat_lines), (length is_flat))

let main  [n][p][a](angles : [a]f32)
           (rhozero : f32)
           (deltarho: f32)
           (image : *[n]f32)
           (projections: [p]f32)
           (iterations : i32) :(i32, i32, i32) =
           SIRT angles rhozero deltarho image projections iterations
