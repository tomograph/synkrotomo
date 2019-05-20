-- ==
-- input@data/sparse/bpsparseinputf32rad1
-- input@data/sparse/bpsparseinputf32rad5
-- input@data/sparse/bpsparseinputf32rad10
-- input@data/sparse/bpsparseinputf32rad15
-- input@data/sparse/bpsparseinputf32rad20
-- input@data/sparse/bpsparseinputf32rad25
-- input@data/sparse/bpsparseinputf32rad30
-- input@data/sparse/bpsparseinputf32rad35
-- input@data/sparse/bpsparseinputf32rad40
-- input@data/sparse/bpsparseinputf32rad45
-- input@data/sparse/bpsparseinputf32rad50
-- input@data/sparse/bpsparseinputf32rad55
-- input@data/sparse/bpsparseinputf32rad60
-- input@data/sparse/bpsparseinputf32rad65
-- input@data/sparse/bpsparseinputf32rad70
-- input@data/sparse/bpsparseinputf32rad75
-- input@data/sparse/bpsparseinputf32rad80
-- input@data/sparse/bpsparseinputf32rad85
-- input@data/sparse/bpsparseinputf32rad90
-- input@data/sparse/bpsparseinputf32rad95

import "backprojection"
open bpTlib

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
