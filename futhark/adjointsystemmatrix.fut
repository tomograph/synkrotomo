-- ==
-- input@../data/sirtinputf32rad64
-- input@../data/sirtinputf32rad128
-- input@../data/sirtinputf32rad256
-- input@../data/sirtinputf32rad512
-- input@../data/sirtinputf32rad1024
-- input@../data/sirtinputf32rad2048

import "line_lib"
open Lines

let main  [n][p][a](angles : [a]f32)
          (rhos : []f32)
          (image : *[n]f32)
          (projections: [p]f32)
          (iterations: i32): [n][p]f32 =
          let size = t32(f32.sqrt(r32(n)))
          in map(\pix ->
               let lowerleft = lowerleftpixelpoint pix size
               in flatten (map(\ang ->
                    let sin = f32.sin(ang)
                    let cos = f32.cos(ang)
                    in (map(\rho->
                              intersectiondistance sin cos rho lowerleft
                         ) rhos)
               ) angles)
          )(iota n)
