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
          let (steep_lines, flat_lines, is_flat, _) = preprocess angles numrhos

	     let (steep_proj, flat_proj) = fix_projections projections is_flat
          in backprojection steep_proj flat_proj steep_lines flat_lines rhozero deltarho rhosprpixel numrhos halfsize
