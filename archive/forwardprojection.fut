-- ==
-- input@../futhark/data/fpinputf32rad64
-- input@../futhark/data/fpinputf32rad128
-- input@../futhark/data/fpinputf32rad256
-- input@../futhark/data/fpinputf32rad512
-- input@../futhark/data/fpinputf32rad1024
-- input@../futhark/data/fpinputf32rad1500
-- input@../futhark/data/fpinputf32rad2000
-- input@../futhark/data/fpinputf32rad2048
-- input@../futhark/data/fpinputf32rad3000
-- input@../futhark/data/fpinputf32rad2500
-- input@../futhark/data/fpinputf32rad3500
-- input@../futhark/data/fpinputf32rad4000
-- input@../futhark/data/fpinputf32rad4096


import "line_lib"
open Lines

module fplib = {

  -- calculate intersection length for all intersecting pixels in grid row i with line with slope >= 1
  let intersect_steep (i: i32) (ext: point) (ent: point) (Nhalf: i32): ((f32,i32,i32),(f32,i32,i32)) =
       let k = (ext.1 - ent.1)/(ext.2 - ent.2)
       let xmin = k*(r32(i) - ent.2) + ent.1 + (r32(Nhalf))
       let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + (r32(Nhalf))
       let Xpixmin = t32(f32.floor(xmin))
       let Xpixplus = t32(f32.floor(xplus))
       let baselength = f32.sqrt(1+k*k)
       let Xpixmax = i32.max Xpixmin Xpixplus
       let xdiff = xplus - xmin
       -- if both equal then l is baselength and we only want one l
       let xminfact = if Xpixmin == Xpixplus then 1 else (r32(Xpixmax) - xmin)/xdiff
       let xplusfact = if Xpixmin == Xpixplus then 0 else (xplus - r32(Xpixmax))/xdiff
       let lxmin = xminfact*baselength
       let lxplus = xplusfact*baselength
       let y = i+Nhalf
       in ((lxmin, Xpixmin, y), (lxplus, Xpixplus, y))

  -- calculate intersection length for all intersecting pixels in grid row i with line with slope <= 1
  let intersect_flat (i: i32) (ext: point) (ent: point) (Nhalf: i32): ((f32,i32,i32),(f32,i32,i32)) =
       let k = (ext.2 - ent.2)/(ext.1 - ent.1)
       let ymin = k*(r32(i) - ent.1) + ent.2 + (r32(Nhalf))
       let yplus = k*(r32(i) + 1 - ent.1) + ent.2 + (r32(Nhalf))
       let Ypixmin = t32(f32.floor(ymin))
       let Ypixplus = t32(f32.floor(yplus))
       -- could be done for all rays of same angle at once
       let baselength = f32.sqrt(1+k*k)
       let Ypixmax = i32.max Ypixmin Ypixplus
       let ydiff = yplus - ymin
       -- if both equal then l is baselength and we only want one l
       let yminfact = if Ypixmin == Ypixplus then 1 else (r32(Ypixmax) - ymin)/ydiff
       let yplusfact = if Ypixmin == Ypixplus then 0 else (yplus - r32(Ypixmax))/ydiff
       let lymin = yminfact*baselength
       let lyplus = yplusfact*baselength
       let x = i+Nhalf
       in ((lymin, x, Ypixmin), (lyplus, x, Ypixplus))

  -- calculate the sum of products for intersections between lines of type rho = cos*x+y*sin on row or column i
  let calculate_product [n](sin: f32)
            (cos: f32)
            (rho: f32)
            (i: i32)
            (halfsize: i32)
            (vct: [n]f32) : f32 =
       let (ent,ext) = entryexitPoint sin cos rho (r32(halfsize))
       let flat = is_flat cos sin
       let ((lmin,xmin,ymin),(lplus,xplus,yplus)) = if flat then intersect_flat i ext ent halfsize else intersect_steep i ext ent halfsize
       let size = halfsize*2
       let pixmin = xmin+ymin*size
       let pixplus = xplus+yplus*size
       let min = if xmin >= 0 && xmin < size && ymin >=0 && ymin < size then (unsafe lmin*vct[pixmin]) else 0.0f32
       let plus = if  xplus >= 0 && xplus < size && yplus >=0 && yplus < size then (unsafe lplus*vct[pixplus]) else 0.0f32
       in (min+plus)

  -- calculate one value in the forward projection vector
  let forward_projection_value (sin: f32) (cos: f32) (rho: f32) (halfsize: i32) (img: []f32): f32 =
       reduce (+) 0.0f32 <| map(\i -> calculate_product sin cos rho i halfsize img)((-halfsize)...(halfsize-1))

  let forwardprojection [a][n](angles: [a]f32)
        (rhozero : f32)
        (deltarho : f32)
        (numrhos : i32)
        (img: [n]f32): []f32 =
    let size = t32(f32.sqrt(r32(n)))
    let halfsize = size/2
    let rhos = map(\i -> rhozero + (r32(i))*deltarho)(iota numrhos)
    in flatten <| map(\ang->
        let sin = -f32.sin(ang)
        let cos = f32.cos(ang)
        in (map(\r -> forward_projection_value sin cos r halfsize img) rhos)
      ) angles
}

open fplib
let main  [n][a] (angles : *[a]f32)
          (rhozero : f32)
          (deltarho : f32)
          (numrhos : i32)
          (image : *[n]f32) =
          forwardprojection angles rhozero deltarho numrhos image
