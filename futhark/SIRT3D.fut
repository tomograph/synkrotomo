-- ==
-- input@../data/sirt3Dinputf32rad64
-- input@../data/sirt3Dinputf32rad128
-- input@../data/sirt3Dinputf32rad256
-- input@../data/sirt3Dinputf32rad512
-- input@../data/sirt3Dinputf32rad1024
-- input@../data/sirt3Dinputf32rad1500
-- input@../data/sirt3Dinputf32rad2000
-- input@../data/sirt3Dinputf32rad2048
-- input@../data/sirt3Dinputf32rad2500
-- input@../data/sirt3Dinputf32rad3000
-- input@../data/sirt3Dinputf32rad3500
-- input@../data/sirt3Dinputf32rad4000
-- input@../data/sirt3Dinputf32rad4096

import "SIRT"

let main  [p][a][r](angles : [a]f32)
          (rhos : [r]f32)
          (projections: [p]f32)
          (iterations : i32)
          (size: i32) : []f32 =
          flatten(
               (unsafe map(\i ->
                    let image = replicate (size*size) 0.0
                    let proj = (unsafe projections[i*r*a:(i+1)*r*a])
                    in unsafe(SIRT angles rhos image proj iterations)
               ) (iota size))
          )
