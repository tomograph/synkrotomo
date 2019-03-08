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

let main  [p][r](angles : []f32)
          (rhozero : f32)
          (deltarho : f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let halfsize = size/2
          let lines = preprocess angles
          let steep = bp lines.2 rhozero deltarho rhosprpixel r halfsize projections
          let flat = bp lines.1 rhozero deltarho rhosprpixel r halfsize projections
          --untranspose in flat case
          let flatT =  if (size < 10000)
                        then flatten <| transpose <| copy (unflatten size size flat)
                        else (replicate n 1.0f32)
          in map2 (+) steep flat
