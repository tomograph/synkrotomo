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

  let forwardprojection_steep [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
    let fhalfsize = r32(halfsize)
    let size = halfsize*2
    in flatten <| map (\(cos, sin, lbase, _) ->
      map (\r ->
        let rho = (rhozero + r32(r)*deltarho)
        let fpv = map (\i ->
          let ent = (find_x (-fhalfsize) rho cos sin, (-fhalfsize))
          let ext = (find_x fhalfsize rho cos sin, fhalfsize)

          let k = (ext.1 - ent.1)/(ext.2 - ent.2)
          let xmin = k*(r32(i) - ent.2) + ent.1 + (fhalfsize)
          let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + (fhalfsize)
          let Xpixmin = f32.floor(xmin)
          let Xpixplus = f32.floor(xplus)

          let Xpixmax = f32.max Xpixmin Xpixplus
          let xdiff = xplus - xmin

          let b = if f32.abs(Xpixmin - Xpixplus) < 0.4f32 then true else false
          let bmin = if Xpixmin >= (-0.4f32) && Xpixmin < (r32(size) + 0.4f32) then true else false
          let bplus = if (!b) && Xpixplus >= (-0.4f32) && Xpixplus < (r32(size) + 0.4f32) then true else false

          let xminfacttmp = (Xpixmax - xmin)/xdiff
          let xminfact = if b then 1 else xminfacttmp
          let xplusfact = (xplus - Xpixmax)/xdiff

          let lxmin = xminfact*lbase
          let lxplus = xplusfact*lbase

          let pixminval = lxmin*(unsafe img[t32(Xpixmin)+(i+halfsize)*size])
          let pixplusval = lxplus*(unsafe img[t32(Xpixplus)+(i+halfsize)*size])

          let min = if bmin then pixminval else 0.0f32
          let plus = if bplus then pixplusval else 0.0f32

          -- let min = if bmin then lxmin*(unsafe img[t32(Xpixmin)+(i+halfsize)*size]) else 0.0f32
          -- let plus = if bplus then lxplus*(unsafe img[t32(Xpixplus)+(i+halfsize)*size]) else 0.0f32

          in (min+plus)
        ) ((-halfsize)...(halfsize-1))
        in (reduce (+) 0.0f32 fpv)
      ) (iota numrhos)
    ) lines

  let forwardprojection_flat [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
    let fhalfsize = r32(halfsize)
    let size = halfsize*2
    in flatten <| map (\(cos, sin, lbase, _) ->
    map (\r ->
      let rho = rhozero + r32(r)*deltarho
      let fpv = map (\i ->
        let ent = ((-fhalfsize), find_y (-fhalfsize) rho cos sin)
        let ext = (fhalfsize, find_y fhalfsize rho cos sin)

        let k = (ext.2 - ent.2)/(ext.1 - ent.1)
        let ymin = k*(r32(i) - ent.1) + ent.2 + fhalfsize
        let yplus = k*(r32(i) + 1 - ent.1) + ent.2 + fhalfsize
        let Ypixmin = f32.floor(ymin)
        let Ypixplus = f32.floor(yplus)

        let Ypixmax = f32.max Ypixmin Ypixplus
        let ydiff = yplus - ymin

        let t1 = (i+halfsize)+t32(Ypixmin)*size >= 0 && (i+halfsize)+t32(Ypixmin)*size < n
        let t2 = (i+halfsize)+t32(Ypixplus)*size >= 0 && (i+halfsize)+t32(Ypixplus)*size < n

        let b = f32.abs(Ypixmin - Ypixplus) < 0.4f32
        let bmin = t1 && Ypixmin >= (-0.0f32) && Ypixmin < r32(size)
        let bplus = (!b) && t2 && Ypixplus >= (-0.0f32) && Ypixplus < r32(size)

        let yminfacttmp = (Ypixmax - ymin)/ydiff
        let yminfact = if b then 1 else yminfacttmp
        let yplusfact = (yplus - Ypixmax)/ydiff

        let lymin = yminfact*(f32.sqrt(1+k*k))
        let lyplus = yplusfact*(f32.sqrt(1+k*k))

        let pixminval = lymin*(unsafe img[(i+halfsize)+t32(Ypixmin)*size])
        let pixplusval = lyplus*(unsafe img[(i+halfsize)+t32(Ypixplus)*size])

        -- let pixminval = (i+halfsize)+t32(Ypixmin)*size
        -- let pixplusval = (i+halfsize)+t32(Ypixplus)*size

        let min = if bmin then pixminval else 0.0f32
        let plus = if bplus then pixplusval else 0.0f32

        -- let min = if t1 && bmin then lymin*(unsafe img[(i+halfsize)+t32(Ypixmin)*size]) else 0.0f32
        -- let plus = if t2 && bplus then lyplus*(unsafe img[(i+halfsize)+t32(Ypixplus)*size]) else 0.0f32

        in (min+plus)
      ) ((-halfsize)...(halfsize-1))
      in (reduce (+) 0.0f32 fpv)
    ) (iota numrhos)
  ) lines


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

     let preprocess [a](angles: [a]f32): ([](f32,f32,i32),[](f32,f32,i32)) =
          let cossin = map(\i -> let angle = angles[i]
               let cos= f32.cos(angle)
               let sin = f32.sin(angle)
               in (cos, sin, i))
          (iota(a))
          in partition(\(c,s,_) -> is_flat c s  )cossin

     let preprocess_2 [a](angles: [a]f32): ([](f32,f32,f32,i32),[](f32,f32,f32,i32)) =
          let cossin = map(\i -> let angle = angles[i]
               let cos= f32.cos(angle)
               let sin = f32.sin(angle)
               let lcot  = f32.sqrt(1.0+(cos/sin)**2.0f32)
               let ltan = f32.sqrt(1.0+(sin/cos)**2.0f32)
               in (cos, sin, lcot,ltan, i))
          (iota(a))
          let parts = partition(\(c,s,_,_,_) -> is_flat c s )cossin
          in ((map (\(cos, sin, lcot,_, i)-> (cos,sin,lcot,i)) parts.1), (map(\(cos, sin, _,ltan, i)-> (cos,sin,ltan,i)) parts.2))

     let back_projection_met [p] (lines: ([](f32,f32,i32),[](f32,f32,i32))) (rhozero: f32) (deltarho: f32) (rhosprpixel: i32) (numrhos: i32) (halfsize: i32) (projections: [p]f32): []f32 =
           let fact = f32.sqrt(2.0f32)/2.0f32
           in flatten (map(\irow ->
               map(\icolumn ->
                    let xmin = r32(icolumn)
                    let ymin = r32(irow)
                    let flat = reduce (+) 0.0f32 <| map(\(cost,sint,angleidx) ->
                         let cott = cost/sint
                         let lbase = f32.sqrt(1.0+cott**2.0f32)
                         let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
                         let rho = cost*p.1+sint*p.2
                         let s = f32.ceil((rho-rhozero)/deltarho)
                         let ybase = xmin*cott
                         in reduce (+) 0.0f32 <| map(\i ->
                                   let sprime = s+(r32(i))
                                   let r = sprime*deltarho+rhozero
                                   let y_left = (r/sint)-ybase
                                   let y_right = y_left-cott
                                   let maxy = f32.max y_left y_right
                                   let miny = f32.min y_left y_right
                                   let l = (intersect_fact maxy miny ymin (ymin+1.0))*lbase
                                   let projectionidx = angleidx*numrhos+(t32(sprime))
                                   in l*(unsafe projections[projectionidx])
                              )(iota rhosprpixel)
                         )lines.1
                    let steep = reduce (+) 0.0f32 <| map(\(cost,sint,angleidx) ->
                         let tant = sint/cost
                         let lbase = f32.sqrt(1.0+tant**2.0f32)
                         let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
                         let rho = cost*p.1+sint*p.2
                         let s = f32.ceil((rho-rhozero)/deltarho)
                         let xbase = ymin*tant
                         in reduce (+) 0.0f32 <| map(\i ->
                                   let sprime = s+(r32(i))
                                   let r = sprime*deltarho+rhozero
                                   let x_bot = (r/cost)-xbase
                                   let x_top = x_bot-tant
                                   let maxx = f32.max x_bot x_top
                                   let minx = f32.min x_bot x_top
                                   let l = (intersect_fact maxx minx xmin (xmin+1.0))*lbase
                                   let projectionidx = angleidx*numrhos+(t32(sprime))
                                   in l*(unsafe projections[projectionidx])
                              )(iota rhosprpixel)
                         )lines.2
                    in flat + steep
               )((-halfsize)...(halfsize-1))
          )((-halfsize)...(halfsize-1)))

     let bp_steep [p] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (rhosprpixel: i32) (numrhos: i32) (halfsize: i32) (projections: [p]f32): []f32 =
          let fact = f32.sqrt(2.0f32)/2.0f32
          in flatten (map(\irow ->
                  map(\icolumn ->
                        let xmin = r32(icolumn)
                        let ymin = r32(irow)
                        in reduce (+) 0.0f32 <| map(\(cost,sint,lbase,angleidx) ->
                             let tant = sint/cost
                             let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
                             let rho = cost*p.1+sint*p.2
                             let s = f32.ceil((rho-rhozero)/deltarho)
                             let xbase = ymin*tant
                             in reduce (+) 0.0f32 <| map(\i ->
                                       let sprime = s+(r32(i))
                                       let r = sprime*deltarho+rhozero
                                       let x_bot = (r/cost)-xbase
                                       let x_top = x_bot-tant
                                       let maxx = f32.max x_bot x_top
                                       let minx = f32.min x_bot x_top
                                       let l = (intersect_fact maxx minx xmin (xmin+1.0))*(lbase)
                                       let projectionidx = angleidx*numrhos+(t32(sprime))
                                       in l*(unsafe projections[projectionidx])
                                  )(iota rhosprpixel)
                        )lines
              )((-halfsize)...(halfsize-1))
         )((-halfsize)...(halfsize-1)))

        let bp_flat [p] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (rhosprpixel: i32) (numrhos: i32) (halfsize: i32) (projections: [p]f32): []f32 =
              let fact = f32.sqrt(2.0f32)/2.0f32
              in flatten (map(\irow ->
                 map(\icolumn ->
                      let xmin = r32(icolumn)
                      let ymin = r32(irow)
                      in reduce (+) 0.0f32 <| map(\(cost,sint,lbase,angleidx) ->
                           let cott = cost/sint
                           let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
                           let rho = cost*p.1+sint*p.2
                           let s = f32.ceil((rho-rhozero)/deltarho)
                           let ybase = xmin*cott
                           in reduce (+) 0.0f32 <| map(\i ->
                                     let sprime = s+(r32(i))
                                     let r = sprime*deltarho+rhozero
                                     let y_left = (r/sint)-ybase
                                     let y_right = y_left-cott
                                     let maxy = f32.max y_left y_right
                                     let miny = f32.min y_left y_right
                                     let l = (intersect_fact maxy miny ymin (ymin+1.0))*lbase
                                     let projectionidx = angleidx*numrhos+(t32(sprime))
                                     in l*(unsafe projections[projectionidx])
                                )(iota rhosprpixel)
                           )lines
                      )((-halfsize)...(halfsize-1))
                 )((-halfsize)...(halfsize-1)))
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
