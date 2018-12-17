import "matrix_lib"
open Matrix

module Projection = {
     -- calculate forward projection
     let forward_projection [a][r][n](angles: [a]f32) (rhos: [r]f32) (halfsize: i32) (img: [n]f32): []f32 =
          flatten(
               (map(\i->
                    let ang = unsafe angles[i]
                    let sin = f32.sin(ang)
                    let cos = f32.cos(ang)
                    in (map(\r -> forward_projection_value sin cos r halfsize img) rhos)
               ) (iota a)))

     -- get the index into the projection vector based on rho and angleindex
     let getprojectionindex (angleindex: i32) (rhovalue: f32) (deltarho: f32) (rhozero: f32) (numrhos: i32): i32 =
          angleindex*numrhos+t32((rhovalue-rhozero)/deltarho)

     -- calculate back_projection
     let back_projection [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32=
          let rhosforpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let rhomax = rhozero + deltarho*r32((p/a)) - 1.0
          in map(\pix ->
               let pixcenter = pixelcenter pix size
               in reduce (+) 0 <| map(\i ->
                    let ang = unsafe angles[i]
                    let sin = f32.sin(ang)
                    let cos = f32.cos(ang)
                    let minrho = rhomin cos sin pixcenter rhozero deltarho
                    let rhos = getrhos minrho deltarho rhosforpixel
                    in reduce (+) 0 <| (map(\rho->
                              let l = intersectiondistance sin cos rho pixcenter
                              let projectionidx = getprojectionindex i rho deltarho rhozero (p/a)
                              in if rho >= rhozero && rho <= rhomax then l*(unsafe projections[projectionidx]) else 0.0
                         ) rhos)
               ) (iota a)
          )(iota (size**2))
}
