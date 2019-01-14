-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- input@../data/sirtinputf32rad1024
-- input@../data/sirtinputf32rad2048
-- input@../data/sirtinputf32rad4096
import "SIRT"

let main  [n][p][a][r](angles : [a]f32)
          (rhos : [r]f32)
          (volume : *[n]f32)
          (projections: [p]f32)
          (iterations : i32)
          (size: i32) : [n]f32 =
          flatten(
               (unsafe map(\i ->
                    let image = copy (unsafe volume[i*size*size:(i+1)*size*size])
                    let proj = (unsafe projections[i*r*a:(i+1)*r*a])
                    in unsafe(SIRT angles rhos image proj iterations)
               ) (iota size))
          )
