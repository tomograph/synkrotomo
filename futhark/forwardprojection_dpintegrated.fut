-- ==
-- input@../data/fpinputf32rad64_32
-- input@../data/fpinputf32rad128_32
-- input@../data/fpinputf32rad256_32
-- input@../data/fpinputf32rad512_32
-- input@../data/fpinputf32rad1024_32
-- input@../data/fpinputf32rad2048_32
-- input@../data/fpinputf32rad4096_32
-- input@../data/fpinputf32rad64_64
-- input@../data/fpinputf32rad128_64
-- input@../data/fpinputf32rad256_64
-- input@../data/fpinputf32rad512_64
-- input@../data/fpinputf32rad1024_64
-- input@../data/fpinputf32rad2048_64
-- input@../data/fpinputf32rad4096_64
-- input@../data/fpinputf32rad64_128
-- input@../data/fpinputf32rad128_128
-- input@../data/fpinputf32rad256_128
-- input@../data/fpinputf32rad512_128
-- input@../data/fpinputf32rad1024_128
-- input@../data/fpinputf32rad2048_128
-- input@../data/fpinputf32rad4096_128
-- input@../data/fpinputf32rad64_256
-- input@../data/fpinputf32rad128_256
-- input@../data/fpinputf32rad256_256
-- input@../data/fpinputf32rad512_256
-- input@../data/fpinputf32rad1024_256
-- input@../data/fpinputf32rad2048_256
-- input@../data/fpinputf32rad4096_256
import "projection_lib"
open Projection

let main  (angles: []f32)
          (rays: []f32)
          (voxels: []f32)
          (stepsize: i32) : []f32 =
          forwardprojection_semiflat angles rays voxels stepsize
