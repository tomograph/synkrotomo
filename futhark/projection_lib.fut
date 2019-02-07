-- ==
-- compiled input {
--    [-1.5f32, -0.5f32, 0.5f32, 1.5f32]
--    [0.0f32, 0.785398f32, 1.5708f32, 2.35619f32]
--    1
--    4
--    10
-- }
-- output {
--    6
-- }
-- compiled input {
--    [-1.5f32, -0.5f32, 0.5f32, 1.5f32]
--    [0.0f32, 0.785398f32, 1.5708f32, 2.35619f32]
--    3
--    4
--    9
-- }
-- output {
--    14
-- }
-- compiled input {
--    [-1.5f32, -0.5f32, 0.5f32, 1.5f32]
--    [0.0f32, 0.785398f32, 1.5708f32, 2.35619f32]
--    3
--    4
--    6
-- }
-- output {
--    13
-- }
-- compiled input {
--    [-1.5f32, -0.5f32, 0.5f32, 1.5f32]
--    [0.0f32, 0.785398f32, 1.5708f32, 2.35619f32]
--    1
--    4
--    5
-- }
-- output {
--    5
-- }

import "lib/github.com/diku-dk/segmented/segmented"
import "matrix_lib"
open Matrix

module Projection = {
  -- calculate forward projection
  let forward_projection [a][r][n](angles: [a]f32) (rhos: [r]f32) (halfsize: i32) (img: [n]f32): []f32 =
    flatten(
      (map(\ang->
        let sin = f32.sin(ang)
        let cos = f32.cos(ang)
        in (map(\r -> forward_projection_value sin cos r halfsize img) rhos)
      ) angles))

  -- get the index into the projection vector based on rho and angleindex
  let getprojectionindex (angleindex: i32) (rhovalue: f32) (deltarho: f32) (rhozero: f32) (numrhos: i32): i32 =
    angleindex*numrhos+(t32(f32.round((rhovalue-rhozero)/deltarho)))

  -- calculate back_projection
  let back_projection [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32=
    let rhosforpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
    --let rhomax = rhozero + deltarho*r32((p/a)) - 1.0f32
    in map(\pix ->
    let lowerleft = lowerleftpixelpoint pix size
    in reduce (+) 0.0f32 <| map(\i ->
      let ang = unsafe angles[i]
      let sin = f32.sin(ang)
      let cos = f32.cos(ang)
      let minrho = rhomin cos sin lowerleft rhozero deltarho
      let rhos = getrhos minrho deltarho rhosforpixel
      in reduce (+) 0.0f32 <| (map(\rho->
        let l = intersectiondistance sin cos rho lowerleft
        let projectionidx = getprojectionindex i rho deltarho rhozero (p/a)
        in l*(unsafe projections[projectionidx])
        ) rhos)
      ) (iota a)
    )(iota (size**2))

  -- calculate back_projection
  let back_projection_testbed [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32=
    let rhosforpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
    --let rhomax = rhozero + deltarho*r32((p/a)) - 1.0f32
    in map(\pix ->
    let lowerleft = lowerleftpixelpoint pix size
    in reduce (+) 0.0f32 <| flatten <| map(\i ->
      let ang = unsafe angles[i]
      let sin = f32.sin(ang)
      let cos = f32.cos(ang)
      let minrho = rhomin cos sin lowerleft rhozero deltarho
      let rhos = getrhos minrho deltarho rhosforpixel
      in (map(\rho->
        let l = intersectiondistance_o sin cos rho lowerleft
        let projectionidx = getprojectionindex i rho deltarho rhozero (p/a)
        in l*(unsafe projections[projectionidx])
        ) rhos)
      ) (iota a)
    )(iota (size**2))

  -- calculate back_projection
  let cur_best [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32=
    let rhosforpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
    --let rhomax = rhozero + deltarho*r32((p/a)) - 1.0f32
    in map(\pix ->
    let lowerleft = lowerleftpixelpoint pix size
    in reduce (+) 0.0f32 <| map(\i ->
      let ang = unsafe angles[i]
      let sin = f32.sin(ang)
      let cos = f32.cos(ang)
      let minrho = rhomin cos sin lowerleft rhozero deltarho
      let rhos = getrhos minrho deltarho rhosforpixel
      in reduce (+) 0.0f32 <| (map(\rho->
        let l = intersectiondistance_o sin cos rho lowerleft
        let projectionidx = getprojectionindex i rho deltarho rhozero (p/a)
        in l*(unsafe projections[projectionidx])
        ) rhos)
      ) (iota a)
    )(iota (size**2))


  -- calculate back_projection
  let back_projection_expand [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32 =
    let rhosforpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
    --let rhomax = rhozero + deltarho*r32((p/a)) - 1.0f32
    let get_val (row:i32) (row_ind:i32) =
      let lowerleft = (r32(row_ind-size/2), r32(row - size/2))
      let pa = p/a
      let vs =flatten <| map (\i ->
        let ang = unsafe angles[i]
        let sin = f32.sin(ang)
        let cos = f32.cos(ang)
        let rhos = getrhos (rhomin cos sin lowerleft rhozero deltarho) deltarho rhosforpixel
        in map (\rho ->
          let l = intersectiondistance_o sin cos rho lowerleft
          let projectionidx = getprojectionindex i rho deltarho rhozero pa
          in l*(unsafe projections[projectionidx])
          ) rhos
        ) (iota a)
      in reduce (+) 0.0f32 vs
    in expand (\_ -> size) get_val (iota size)

  let back_projection_test [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32 =
    back_projection_testbed angles rhozero deltarho size projections

  let back_projection_cur_best [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32 =
    cur_best angles rhozero deltarho size projections
}

open Projection

let main  [r](rhos : [r]f32)
          (angles: []f32)
          (angleindex: i32)
          (size: i32)
          (pixel: i32): i32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          let lowerleft = lowerleftpixelpoint pixel size
          let angle = unsafe angles[angleindex]
          let cost = f32.cos(angle)
          let sint = f32.sin(angle)
          let min = rhomin cost sint lowerleft rhozero deltarho
          let rho = getrhos min deltarho 1
          in getprojectionindex angleindex (unsafe rho[0]) deltarho rhozero r
