-- ==
-- input@data/sparse/fpsparseinputf32rad1
-- input@data/sparse/fpsparseinputf32rad5
-- input@data/sparse/fpsparseinputf32rad10
-- input@data/sparse/fpsparseinputf32rad15
-- input@data/sparse/fpsparseinputf32rad20
-- input@data/sparse/fpsparseinputf32rad25
-- input@data/sparse/fpsparseinputf32rad30
-- input@data/sparse/fpsparseinputf32rad35
-- input@data/sparse/fpsparseinputf32rad40
-- input@data/sparse/fpsparseinputf32rad45
-- input@data/sparse/fpsparseinputf32rad50
-- input@data/sparse/fpsparseinputf32rad55
-- input@data/sparse/fpsparseinputf32rad60
-- input@data/sparse/fpsparseinputf32rad65
-- input@data/sparse/fpsparseinputf32rad70
-- input@data/sparse/fpsparseinputf32rad75
-- input@data/sparse/fpsparseinputf32rad80
-- input@data/sparse/fpsparseinputf32rad85
-- input@data/sparse/fpsparseinputf32rad90
-- input@data/sparse/fpsparseinputf32rad95

import "forwardprojection"
open fpTlib

let main  [n][a] (angles : *[a]f32)
          (rhozero : f32)
          (deltarho : f32)
          (numrhos : i32)
          (image : *[n]f32) =
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let (steep_lines, flat_lines, _, projection_indexes) = preprocess angles numrhos
  -- hack to always do this!
  let imageT =  if (size < 10000)
                then flatten <| transpose <| copy (unflatten size size image)
                else (replicate n 1.0f32)

  in forwardprojection steep_lines flat_lines projection_indexes rhozero deltarho numrhos halfsize image imageT
