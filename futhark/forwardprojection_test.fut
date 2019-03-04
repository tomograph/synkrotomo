-- ==
-- input@../data/fpinputf32rad64
-- output@../sd/sanity-fpinputf32rad64
-- input@../data/fpinputf32rad128
-- output@../sd/sanity-fpinputf32rad128
-- input@../data/fpinputf32rad256
-- output@../sd/sanity-fpinputf32rad256
-- input@../data/fpinputf32rad512
-- output@../sd/sanity-fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- output@../sd/sanity-fpinputf32rad1024
-- input@../data/fpinputf32rad1500
-- output@../sd/sanity-fpinputf32rad1500
-- input@../data/fpinputf32rad2000
-- output@../sd/sanity-fpinputf32rad2000
-- input@../data/fpinputf32rad2048
-- output@../sd/sanity-fpinputf32rad2048
-- input@../data/fpinputf32rad2500
-- output@../sd/sanity-fpinputf32rad2500
-- input@../data/fpinputf32rad3000
-- output@../sd/sanity-fpinputf32rad3000
-- input@../data/fpinputf32rad3500
-- output@../sd/sanity-fpinputf32rad3500
-- input@../data/fpinputf32rad4000
-- output@../sd/sanity-fpinputf32rad4000
-- input@../data/fpinputf32rad4096
-- output@../sd/sanity-fpinputf32rad4096


-- import "projection_lib"
-- open Projection
-- import "preprocessing"

let is_flat (cos: f32) (sin: f32): bool =
  f32.abs(sin) >= f32.abs(cos)

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

let forwardprojection_steep [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
  let fhalfsize = r32(halfsize)
  let size = halfsize*2
  in flatten <| map (\(cos, sin, lbase, ind) ->
    map (\r ->
      let rho = rhozero + r32(r)*deltarho
      let ent = ((rho-(-fhalfsize)*sin)/cos, (-fhalfsize))
      let ext = ((rho-fhalfsize*sin)/cos, fhalfsize)
      let k = (ext.1 - ent.1)/(ext.2 - ent.2)

      let v = reduce (+) 0.0f32 <| map(\i ->
        let xmin = k*(r32(i) - ent.2) + ent.1 + fhalfsize
        let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + fhalfsize

        let Xpixmin = t32(f32.floor(xmin))
        let Xpixplus = t32(f32.floor(xplus))

        let Xpixmax = i32.max Xpixmin Xpixplus
        let xdiff = xplus - xmin

        let bounds = (i+halfsize) >= 0 && (i+halfsize) < size
        let eq = Xpixmin == Xpixplus
        let bmin = bounds && Xpixmin >= 0 && Xpixmin < size
        let bplus = (!eq) && bounds && Xpixplus >= 0 && Xpixplus < size

        let lxmin = if eq then lbase else ((r32(Xpixmax) - xmin)/xdiff)*lbase
        let lxplus = ((xplus - r32(Xpixmax))/xdiff)*lbase

        let pixmin = Xpixmin+(i+halfsize)*size
        let pixplus = Xpixplus+(i+halfsize)*size

        let min = if bmin then (unsafe lxmin*img[pixmin]) else 0.0f32
        let plus = if bplus then (unsafe lxplus*img[pixplus]) else 0.0f32

        in (min+plus)
      ) ((-halfsize)...(halfsize-1))
      -- in (v, ind*numrhos+r)
      in v
    ) (iota numrhos)
  ) lines

let forwardprojection_flat [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
  let fhalfsize = r32(halfsize)
  let size = halfsize*2
  in flatten <| map (\(cos, sin, lbase, ind) ->
    map (\r ->
      let rho = rhozero + r32(r)*deltarho
      let ent = ((-fhalfsize), (rho-(-fhalfsize)*cos)/sin)
      let ext = (fhalfsize, (rho-fhalfsize*cos)/sin)
      let k = (ext.2 - ent.2)/(ext.1 - ent.1)

      let v = reduce (+) 0.0f32 <| map(\i ->

        let ymin = k*(r32(i) - ent.1) + ent.2 + fhalfsize
        let yplus = k*(r32(i) + 1 - ent.1) + ent.2 + fhalfsize

        let Ypixmin = t32(f32.floor(ymin))
        let Ypixplus = t32(f32.floor(yplus))
        -- could be done for all rays of same angle at once
        let Ypixmax = i32.max Ypixmin Ypixplus
        let ydiff = yplus - ymin

        -- bools
        let bounds = (i+halfsize) >= 0 && (i+halfsize) < size
        let eq = Ypixmin == Ypixplus
        let bmin = bounds && Ypixmin >=0 && Ypixmin < size
        let bplus = (!eq) && bounds && Ypixplus >=0 && Ypixplus < size

        let lymin = if eq then lbase else ((r32(Ypixmax) - ymin)/ydiff)*lbase
        let lyplus = ((yplus - r32(Ypixmax))/ydiff)*lbase

        let pixmin = (i+halfsize)+Ypixmin*size
        let pixplus = (i+halfsize)+Ypixplus*size

        let min = if bmin then (unsafe lymin*img[pixmin]) else 0.0f32
        let plus = if bplus then (unsafe lyplus*img[pixplus]) else 0.0f32

        in (min+plus)

        )((-halfsize)...(halfsize-1))
      -- in (v, ind*numrhos+r)
      in v
  ) (iota numrhos)
) lines

let main  [n][r][a] (angles : [a]f32)
          (rhos : [r]f32)
          (image : *[n]f32) =
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2
  let rhozero = unsafe rhos[0]
  let deltarho = unsafe rhos[1]-rhozero
  let numrhos = r
  let lines = preprocess_2 angles
  -- let (lines, rhozero, deltarho, numrhos) = preprocessing angles rhos
  let steep = forwardprojection_steep lines.2 rhozero deltarho numrhos halfsize image
  let flat = forwardprojection_flat lines.1 rhozero deltarho numrhos halfsize image
  in steep ++ flat
  -- let arr = steep ++ flat
  -- let vals = map (\(v, _) -> v) arr
  -- let inds = map (\(_, i) -> i) arr
  -- in scatter (replicate (a*r) 0.0f32) inds vals
