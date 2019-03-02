-- ==
-- input@../data/fpinputf32rad64
-- output@../sd/sanity-fpinputf32rad64
-- input@../data/fpinputf32rad128
-- output@../sd/sanity-fpinputf32rad128
-- input@../data/fpinputf32rad256
-- output@../sd/sanity-fpinputf32rad256
-- input@../data/fpinputf32rad512
-- output@../sd/sanity-fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- output@../sd/sanity-fpinputf32rad1024
-- input@../data/fpinputf32rad2048
-- output@../sd/sanity-fpinputf32rad2048

import "projection_lib"
open Projection

let main  [n](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32): []f32 =
          let size = t32(f32.sqrt(r32(n)))
          let halfsize = size/2
          in forward_projection angles rhos halfsize image
