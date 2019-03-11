-- ==
-- input@../data/bpinputf32rad64
-- input@../data/bpinputf32rad128
-- input@../data/bpinputf32rad256
-- input@../data/bpinputf32rad512
-- input@../data/bpinputf32rad1024
-- input@../data/bpinputf32rad2048
-- input@../data/bpinputf32rad4096


import "sirtlib"
open sirtlib

let main  [p][a](angles : [a]f32)
          (rhozero : f32)
          (deltarho : f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let halfsize = size/2
          let numrhos = p/a
          let (steep_lines, flat_lines, is_flat, projection_indexes) = preprocess angles numrhos

	     let (flat_proj, steep_proj) = fix_projections projections is_flat
          let steep = bp steep_lines rhozero deltarho rhosprpixel numrhos halfsize steep_proj
          let flat = bp flat_lines rhozero deltarho rhosprpixel numrhos halfsize flat_proj
          --untranspose in flat case
          let flatT =  if (size < 10000)
                       then flatten <| transpose <| unflatten size size flat
                       else (replicate (size**2) 1.0f32)
          in map2 (+) steep flatT
