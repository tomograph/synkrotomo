-- ==
-- input@../data/bpinputf32rad64
-- output@../sd/sanity-bpinputf32rad64
-- input@../data/bpinputf32rad128
-- output@../sd/sanity-bpinputf32rad128
-- input@../data/bpinputf32rad256
-- output@../sd/sanity-bpinputf32rad256
-- input@../data/bpinputf32rad512
-- output@../sd/sanity-bpinputf32rad512
-- input@../data/bpinputf32rad1024
-- output@../sd/sanity-bpinputf32rad1024
-- input@../data/bpinputf32rad2048
-- output@../sd/sanity-bpinputf32rad2048


import "projection_lib"
open Projection

let main  [p][r](angles : []f32)
          (rhos : [r]f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let halfsize = size/2
          let lines = preprocess_2 angles
          let steep = bp_steep lines.2 rhozero deltarho rhosprpixel r halfsize projections
          let flat = bp_flat lines.1 rhozero deltarho rhosprpixel r halfsize projections
          in map2 (+) steep flat
