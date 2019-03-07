-- ==
-- input@../data/fpinputf32rad64
-- input@../data/fpinputf32rad128
-- input@../data/fpinputf32rad256
-- input@../data/fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- input@../data/fpinputf32rad2048


import "testlib"
open testlib

-- let is_flat (cos: f32) (sin: f32): bool =
--   f32.abs(sin) >= f32.abs(cos)
--
-- let preprocess [a](angles: [a]f32): ([](f32, f32, f32), [](f32, f32, f32)) =
--   let cossin = map(\angle ->
--     let cos= f32.cos(angle)
--     let sin = f32.sin(angle)
--     let lcot = f32.sqrt(1.0+(cos/sin)**2.0f32)
--     let ltan = f32.sqrt(1.0+(sin/cos)**2.0f32)
--     in (cos, sin, lcot, ltan)
--   ) angles
--   let parts = partition(\(c,s,_,_) -> is_flat c s ) cossin
--   in (map (\(cos, sin, lcot, _)-> (cos, sin, lcot)) parts.1, map(\(cos, sin, _, ltan)-> (cos, sin, ltan)) parts.2)
--
-- let forwardprojection_steep [n] (lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
--   let fhalfsize = r32(halfsize)
--   let size = halfsize*2
--   in flatten <| map (\(cos, sin, lbase) ->
--     map (\r ->
--       let rho = rhozero + r32(r)*deltarho
--       let ent = ((rho-(-fhalfsize)*sin)/cos, (-fhalfsize))
--       let ext = ((rho-fhalfsize*sin)/cos, fhalfsize)
--       let k = (ext.1 - ent.1)/(ext.2 - ent.2)
--       let kbase = ent.1 + fhalfsize
--
--       in reduce (+) 0.0f32 <| map(\i ->
--         let ih = i+halfsize
--
--         let xmin = k*(r32(i) - ent.2) + kbase
--         let xplus = k*(r32(i) + 1 - ent.2) + kbase
--
--         let Xpixmin = t32(f32.floor(xmin))
--         let Xpixplus = t32(f32.floor(xplus))
--
--         let Xpixmax = r32(i32.max Xpixmin Xpixplus)
--         let xdiff = xplus - xmin
--
--         let bounds = ih >= 0 && ih < size
--         let eq = Xpixmin == Xpixplus
--         let bmin = bounds && Xpixmin >= 0 && Xpixmin < size
--         let bplus = (!eq) && bounds && Xpixplus >= 0 && Xpixplus < size
--
--         let lxminfac = ((Xpixmax - xmin)/xdiff)
--         let lxmin = if eq then lbase else lxminfac*lbase
--         let lxplus = ((xplus - Xpixmax)/xdiff)*lbase
--
--         let pixmin = Xpixmin+ih*size
--         let pixplus = Xpixplus+ih*size
--
--         let min = if bmin then (unsafe lxmin*img[pixmin]) else 0.0f32
--         let plus = if bplus then (unsafe lxplus*img[pixplus]) else 0.0f32
--
--         in (min+plus)
--       ) ((-halfsize)...(halfsize-1))
--     ) (iota numrhos)
--   ) lines
--
-- let forwardprojection_flat [n] (lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
--   let fhalfsize = r32(halfsize)
--   let size = halfsize*2
--   in flatten <| map (\(cos, sin, lbase) ->
--     map (\r ->
--       let rho = rhozero + r32(r)*deltarho
--       let ent = ((-fhalfsize), (rho-(-fhalfsize)*cos)/sin)
--       let ext = (fhalfsize, (rho-fhalfsize*cos)/sin)
--       let k = (ext.2 - ent.2)/(ext.1 - ent.1)
--       let kbase = ent.2 + fhalfsize
--
--       in reduce (+) 0.0f32 <| map(\i ->
--         let ih = i+halfsize
--
--         let ymin = k*(r32(i) - ent.1) + kbase
--         let yplus = k*(r32(i) + 1 - ent.1) + kbase
--
--         let Ypixmin = t32(f32.floor(ymin))
--         let Ypixplus = t32(f32.floor(yplus))
--         let Ypixmax = r32(i32.max Ypixmin Ypixplus)
--         let ydiff = yplus - ymin
--
--         -- bools
--         let bounds = ih >= 0 && ih < size
--         let eq = Ypixmin == Ypixplus
--         let bmin = bounds && Ypixmin >=0 && Ypixmin < size
--         let bplus = (!eq) && bounds && Ypixplus >=0 && Ypixplus < size
--
--         let lyminfac = ((Ypixmax - ymin)/ydiff)
--         let lymin = if eq then lbase else lyminfac*lbase
--         let lyplus = ((yplus - Ypixmax)/ydiff)*lbase
--
--         let pixmin = ih+Ypixmin*size
--         let pixplus = ih+Ypixplus*size
--
--         let min = if bmin then (unsafe lymin*img[pixmin]) else 0.0f32
--         let plus = if bplus then (unsafe lyplus*img[pixplus]) else 0.0f32
--
--         in (min+plus)
--         )((-halfsize)...(halfsize-1))
--   ) (iota numrhos)
-- ) lines

let main  [n][r][a] (angles : [a]f32)
          (rhos : [r]f32)
          (image : *[n]f32) =
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2
  let rhozero = unsafe rhos[0]
  let deltarho = unsafe rhos[1]-rhozero
  let numrhos = r
  let lines = preprocess angles
  -- let (lines, rhozero, deltarho, numrhos) = preprocessing angles rhos
  let steep = forwardprojection lines.2 rhozero deltarho numrhos halfsize image
  let flat = forwardprojection lines.1 rhozero deltarho numrhos halfsize image
  in steep ++ flat
  -- let arr = steep ++ flat
  -- let vals = map (\(v, _) -> v) arr
  -- let inds = map (\(_, i) -> i) arr
  -- in scatter (replicate (a*r) 0.0f32) inds vals
