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
          (iterations: i32): [][][]f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let size = t32(f32.sqrt(r32(n)))
          let rhosforpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let rhomax = rhozero + deltarho*r32((p/a)) - 1.0
          in map(\pix ->
               let pixcenter = pixelcenter pix size
               in (map(\i ->
                    let ang = unsafe angles[i]
                    let sin = f32.sin(ang)
                    let cos = f32.cos(ang)
                    let minrho = rhomin cos sin pixcenter rhozero deltarho
                    let rhos = getrhos minrho deltarho rhosforpixel
                    in (map(\rho->
                              intersectiondistance sin cos rho pixcenter
                         ) rhos)
               ) (iota a))
          )(iota (size**2))
