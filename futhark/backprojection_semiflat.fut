-- ==
-- input@../data/bpinputf32rad64_32
-- input@../data/bpinputf32rad128_32
-- input@../data/bpinputf32rad256_32
-- input@../data/bpinputf32rad512_32
-- input@../data/bpinputf32rad1024_32
-- input@../data/bpinputf32rad2048_32
-- input@../data/bpinputf32rad4096_32
-- input@../data/bpinputf32rad64_64
-- input@../data/bpinputf32rad128_64
-- input@../data/bpinputf32rad256_64
-- input@../data/bpinputf32rad512_64
-- input@../data/bpinputf32rad1024_64
-- input@../data/bpinputf32rad2048_64
-- input@../data/bpinputf32rad4096_64
-- input@../data/bpinputf32rad64_128
-- input@../data/bpinputf32rad128_128
-- input@../data/bpinputf32rad256_128
-- input@../data/bpinputf32rad512_128
-- input@../data/bpinputf32rad1024_128
-- input@../data/bpinputf32rad2048_128
-- input@../data/bpinputf32rad4096_128
-- input@../data/bpinputf32rad64_256
-- input@../data/bpinputf32rad128_256
-- input@../data/bpinputf32rad256_256
-- input@../data/bpinputf32rad512_256
-- input@../data/bpinputf32rad1024_256
-- input@../data/bpinputf32rad2048_256
-- input@../data/bpinputf32rad4096_256
import "projection_lib"
open Projection

let main  (angles : []f32)
          (rays : []f32)
          (projections : []f32)
          (gridsize: i32)
          (stepSize : i32) : []f32 =
          backprojection_semiflat angles rays projections gridsize stepSize
