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
    flatten <| map (\(cos, sin, lbase, _) ->
      map (\r ->
        let rho = (rhozero + r32(r)*deltarho)
        let fpv = map (\i ->
          let ent = (find_x (-1.0*(r32(halfsize))) rho cost sint, (-1.0*(r32(halfsize))))
          let ext = (find_x (r32(halfsize)) rho cost sint, (r32(halfsize)))

          -- let (ent,ext) = entryexitPoint sin cos rho (r32(halfsize))
          let k = (ext.1 - ent.1)/(ext.2 - ent.2)
          let xmin = k*(r32(i) - ent.2) + ent.1 + (r32(halfsize))
          let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + (r32(halfsize))
          let Xpixmin = t32(f32.floor(xmin))
          let Xpixplus = t32(f32.floor(xplus))
          let baselength = lbase
          let Xpixmax = i32.max Xpixmin Xpixplus
          let xdiff = xplus - xmin
          -- if both equal then l is baselength and we only want one l
          let xminfact = if Xpixmin == Xpixplus then 1 else (r32(Xpixmax) - xmin)/xdiff
          let xplusfact = if Xpixmin == Xpixplus then 0 else (xplus - r32(Xpixmax))/xdiff
          let lxmin = xminfact*baselength
          let lxplus = xplusfact*baselength
          let y = i+halfsize
          -- FIX this is just renaming
          let ((lmin,xmin,ymin),(lplus,xplus,yplus)) = ((lxmin, Xpixmin, y), (lxplus, Xpixplus, y))
          let size = halfsize*2
          let pixmin = xmin+ymin*size
          let pixplus = xplus+yplus*size
          let min = if xmin >= 0 && xmin < size && ymin >=0 && ymin < size then (unsafe lmin*img[pixmin]) else 0.0f32
          let plus = if  xplus >= 0 && xplus < size && yplus >=0 && yplus < size then (unsafe lplus*img[pixplus]) else 0.0f32
          in (min+plus)
        ) ((-halfsize)...(halfsize-1))
        in (reduce (+) 0.0f32 fpv)
      ) (iota numrhos)
    ) lines

    let forwardprojection_flat [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
    flatten <| map (\(cos, sin, lbase, _) ->
      map (\r ->
        let rho = rhozero + r32(r)*deltarho
        let fpv = map (\i ->
          let ent = ((-1.0*(r32(halfsize))), find_y (-1.0*(r32(halfsize))) rho cost sint)
          let ext = ((r32(halfsize)), find_y (r32(halfsize)) rho cost sint)
          -- let (ent,ext) = entryexitPoint sin cos rho (r32(halfsize))
          let k = (ext.2 - ent.2)/(ext.1 - ent.1)
          let ymin = k*(r32(i) - ent.1) + ent.2 + (r32(halfsize))
          let yplus = k*(r32(i) + 1 - ent.1) + ent.2 + (r32(halfsize))
          let Ypixmin = t32(f32.floor(ymin))
          let Ypixplus = t32(f32.floor(yplus))
          -- could be done for all rays of same angle at once
          let baselength = lbase
          let Ypixmax = i32.max Ypixmin Ypixplus
          let ydiff = yplus - ymin
          -- if both equal then l is baselength and we only want one l
          let yminfact = if Ypixmin == Ypixplus then 1 else (r32(Ypixmax) - ymin)/ydiff
          let yplusfact = if Ypixmin == Ypixplus then 0 else (yplus - r32(Ypixmax))/ydiff
          let lymin = yminfact*baselength
          let lyplus = yplusfact*baselength
          let x = i+halfsize
          -- FIX this is just renaming
          let ((lmin,xmin,ymin),(lplus,xplus,yplus)) = ((lymin, x, Ypixmin), (lyplus, x, Ypixplus))
          let size = halfsize*2
          let pixmin = xmin+ymin*size
          let pixplus = xplus+yplus*size
          let min = if xmin >= 0 && xmin < size && ymin >=0 && ymin < size then (unsafe lmin*img[pixmin]) else 0.0f32
          let plus = if  xplus >= 0 && xplus < size && yplus >=0 && yplus < size then (unsafe lplus*img[pixplus]) else 0.0f32
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
