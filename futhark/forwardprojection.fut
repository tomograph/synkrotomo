-- ==
-- input@../data/fpinputf32rad64
-- input@../data/fpinputf32rad128
-- input@../data/fpinputf32rad256
-- input@../data/fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- input@../data/fpinputf32rad2048


import "sirtlib"
open sirtlib

let main  [n][r][a] (angles : [a]f32)
          (rhozero : f32)
          (deltarho : f32)
          (numrhos : i32)
          (image : *[n]f32) =
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let lines = preprocess angles
  -- hack to always do this!
  let imageT =  if (size < 10000)
                then flatten <| transpose <| copy (unflatten size size image)
                else (replicate n 1.0f32)

  let steep = forwardprojection lines.2 rhozero deltarho numrhos halfsize image
  let flat = forwardprojection lines.1 rhozero deltarho numrhos halfsize imageT
  in postprocess_fp angles flat steep numrhos
