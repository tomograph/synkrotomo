-- ==
-- input@../data/bpinputf32rad64
-- input@../data/bpinputf32rad128
-- input@../data/bpinputf32rad256
-- input@../data/bpinputf32rad512
-- input@../data/bpinputf32rad1024
-- input@../data/bpinputf32rad2048
-- input@../data/bpinputf32rad4096


import "testlib"
open testlib

let main  [p][r](angles : []f32)
          (rhos : [r]f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let halfsize = size/2
          let lines = preprocess angles
          let steep = bp_steep lines.2 0 rhozero deltarho rhosprpixel r halfsize projections
          let flat = bp_flat lines.1 (length lines.2) rhozero deltarho rhosprpixel r halfsize projections
          in map2 (+) steep flat
